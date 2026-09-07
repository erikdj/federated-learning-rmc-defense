"""Tenure-Gated Ensemble (TGE) Defense Model.

Core implementation of the TGE model combining:
1. GBDTColdStartExpert — IsolationForest anomaly detector on geometric features
2. LSTMTemporalExpert — LSTM autoencoder for temporal drift detection
3. TenureGatedDecisionRule — Routes scoring based on client tenure

Feature extraction produces a fixed-size vector per client per round from
their model update, shared by both experts. The GBDT handles cold-start
clients (low tenure) and the LSTM handles established clients with enough
temporal history to detect behavioral drift.

Usage:
    model = TGEnsembleModel(num_features=12)
    features = model.extract_features(client_params, all_params)
    score = model.score_client(client_id, features, tenure, server_round)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from sklearn.ensemble import IsolationForest
import logging
import warnings

logger = logging.getLogger(__name__)

# Suppress sklearn convergence/fit warnings during warmup
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Fixed seed for reproducibility
_RNG_SEED = 42


def _round_or_none(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    """Round for logging, preserving None (the null-until-first-update contract
    of the EMA leg's signal-log field)."""
    return None if value is None else round(value, ndigits)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_geometric_features(
    client_flat: np.ndarray,
    all_flat: List[np.ndarray],
    layer_boundaries: Optional[List[int]] = None,
) -> np.ndarray:
    """Extract a fixed-size geometric feature vector from a client update.

    Features (12-dimensional by default for a 3-layer network):
        [0]  z-scored distance from coordinate-wise median
        [1]  cosine similarity to median update
        [2]  L2 norm deviation from median norm (absolute ratio)
        [3..3+L-1]  per-layer norm ratio (client layer norm / median layer norm)
        [3+L]  mean of absolute update values
        [3+L+1]  std of absolute update values
        [3+L+2]  max of absolute update values

    Args:
        client_flat: Flattened parameter vector for the client.
        all_flat: List of flattened parameter vectors for ALL clients this round.
        layer_boundaries: Cumulative sizes for per-layer features. If None,
            treats entire vector as one layer.

    Returns:
        1-D numpy feature vector.
    """
    n = len(all_flat)
    stacked = np.stack(all_flat)
    median = np.median(stacked, axis=0)

    # --- Feature 0: z-scored distance from median ---
    distances = np.array([np.linalg.norm(u - median) for u in all_flat])
    mean_d = distances.mean()
    std_d = distances.std()
    client_dist = np.linalg.norm(client_flat - median)
    if std_d > 1e-12:
        z_distance = (client_dist - mean_d) / std_d
    else:
        z_distance = 0.0

    # --- Feature 1: cosine similarity to median ---
    median_norm = np.linalg.norm(median)
    client_norm = np.linalg.norm(client_flat)
    if median_norm > 1e-12 and client_norm > 1e-12:
        cos_sim = np.dot(client_flat, median) / (client_norm * median_norm)
    else:
        cos_sim = 1.0

    # --- Feature 2: norm deviation ratio ---
    norms = np.array([np.linalg.norm(u) for u in all_flat])
    median_norm_val = np.median(norms)
    if median_norm_val > 1e-12:
        norm_dev = abs(client_norm - median_norm_val) / median_norm_val
    else:
        norm_dev = 0.0

    # --- Features 3..3+L-1: per-layer norm ratios ---
    if layer_boundaries is None:
        layer_boundaries = [len(client_flat)]

    layer_ratios = []
    prev = 0
    for boundary in layer_boundaries:
        layer_slice_client = client_flat[prev:boundary]
        layer_norms = []
        for u in all_flat:
            layer_norms.append(np.linalg.norm(u[prev:boundary]))
        layer_norms = np.array(layer_norms)
        median_layer_norm = np.median(layer_norms)
        client_layer_norm = np.linalg.norm(layer_slice_client)
        if median_layer_norm > 1e-12:
            layer_ratios.append(client_layer_norm / median_layer_norm)
        else:
            layer_ratios.append(1.0)
        prev = boundary

    # --- Magnitude statistics ---
    abs_vals = np.abs(client_flat)
    mag_mean = abs_vals.mean()
    mag_std = abs_vals.std()
    mag_max = abs_vals.max()

    features = np.array(
        [z_distance, cos_sim, norm_dev] + layer_ratios + [mag_mean, mag_std, mag_max],
        dtype=np.float32,
    )
    return features


def compute_layer_boundaries(param_shapes: List[Tuple[int, ...]]) -> List[int]:
    """Compute cumulative layer boundaries from parameter shapes.

    Args:
        param_shapes: List of shapes, one per parameter tensor
            (e.g., [(45, 64), (64,), (64, 32), (32,), (32, 2), (2,)])

    Returns:
        List of cumulative sizes for slicing the flattened vector.
    """
    boundaries = []
    cumulative = 0
    for shape in param_shapes:
        size = 1
        for dim in shape:
            size *= dim
        cumulative += size
        boundaries.append(cumulative)
    return boundaries


# ============================================================================
# GBDT COLD-START EXPERT
# ============================================================================

class GBDTColdStartExpert:
    """IsolationForest-based anomaly detector for cold-start clients.

    During warmup (first ``warmup_rounds`` rounds), collects feature vectors
    from ALL clients (not just accepted) so the model learns the true
    distribution including natural heterogeneity. After warmup, fits an
    IsolationForest on accumulated features and uses anomaly scores.

    The model is refitted periodically (every ``refit_interval`` rounds) to
    adapt to distribution shift. Post-warmup accumulation only includes
    accepted clients to avoid training on confirmed-anomalous updates.

    Calibration uses z-score normalization of IsolationForest raw scores
    against the training distribution, so natural heterogeneity (even
    relatively distant but honest clients) stays above the 0.7 threshold.
    Only extreme outliers (>2.5 sigma from training mean) get rejected.
    """

    def __init__(
        self,
        warmup_rounds: int = 3,
        refit_interval: int = 5,
        contamination: str = "auto",
        n_estimators: int = 100,
        seed: int = _RNG_SEED,
    ):
        self.warmup_rounds = warmup_rounds
        self.refit_interval = refit_interval
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.seed = seed

        self._model: Optional[IsolationForest] = None
        self._feature_buffer: List[np.ndarray] = []
        self._is_fitted = False
        self._last_fit_round = 0
        # Store training-set score statistics for calibration
        self._train_score_mean: float = 0.0
        self._train_score_std: float = 1.0
        self._train_score_min: float = 0.0  # minimum raw score in training set

    @property
    def is_ready(self) -> bool:
        return self._is_fitted

    def accumulate(self, features: np.ndarray):
        """Add a feature vector to the training buffer."""
        self._feature_buffer.append(features.copy())

    def fit(self, server_round: int):
        """Fit or refit the IsolationForest on accumulated features.

        After fitting, computes the score distribution on the training set
        to calibrate the z-score normalization. This ensures that clients
        whose features fall within the training distribution's range score
        above the 0.7 threshold, even if they are on the tail.
        """
        if len(self._feature_buffer) < 5:
            logger.debug("GBDT: not enough samples to fit (%d)", len(self._feature_buffer))
            return

        X = np.stack(self._feature_buffer)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=1,
        )
        self._model.fit(X)
        self._is_fitted = True
        self._last_fit_round = server_round

        # Compute training score distribution for calibration
        raw_scores = self._model.decision_function(X)
        self._train_score_mean = float(raw_scores.mean())
        self._train_score_std = float(max(raw_scores.std(), 1e-8))
        self._train_score_min = float(raw_scores.min())

        logger.debug(
            "GBDT fitted on %d samples at round %d "
            "(score mean=%.4f, std=%.4f, min=%.4f)",
            len(X), server_round, self._train_score_mean,
            self._train_score_std, self._train_score_min,
        )

    def maybe_refit(self, server_round: int):
        """Refit if enough rounds have passed since last fit."""
        if server_round - self._last_fit_round >= self.refit_interval:
            self.fit(server_round)

    def score(self, features: np.ndarray) -> float:
        """Score a single feature vector. Returns value in [0, 1].

        Higher = more trustworthy (normal). Lower = anomalous.

        Calibration is anchored to the training distribution: any raw score
        at or above the training set minimum maps to >= 0.75 (above the 0.7
        threshold). This ensures that clients whose behavior falls within the
        observed training distribution are never falsely rejected.

        Only clients whose raw score is significantly BELOW the training
        minimum (i.e., more anomalous than anything seen in training) get
        scores below 0.7. This is the correct behavior: the GBDT is trained
        on all clients during warmup, so the training minimum represents the
        most "different" honest client.

        Mapping:
        - raw >= train_mean: score ~0.90-0.95
        - raw = train_min: score ~0.75
        - raw < train_min (1 range below): score drops toward 0.5
        - raw << train_min (2+ ranges below): score drops toward 0.2
        """
        if not self._is_fitted or self._model is None:
            return 0.85  # neutral/benign score during warmup

        # IsolationForest.decision_function: positive = inlier, negative = outlier
        raw = self._model.decision_function(features.reshape(1, -1))[0]

        # Map raw score to [0, 1] anchored at training distribution boundaries
        # Range of training scores: [train_min, train_mean + 2*std]
        train_range = self._train_score_mean - self._train_score_min
        if train_range < 1e-12:
            train_range = self._train_score_std if self._train_score_std > 1e-12 else 1.0

        # Position relative to training min: 0 = at training min, 1 = at mean
        # Below 0 = worse than worst training sample
        pos = (raw - self._train_score_min) / train_range

        # Sigmoid centered so that pos=0 (training min) -> 0.75
        # pos=1 (training mean) -> 0.93
        # pos=-1 (one range below min) -> 0.38
        # score = 1 / (1 + exp(-k * (pos - shift)))
        # Want: pos=0 -> 0.75 => 0.75 = 1/(1+exp(k*shift)) => exp(k*shift) = 1/3 => k*shift = 1.099
        # Want: pos=-1 -> 0.40 => 0.40 = 1/(1+exp(k*(1+shift))) => k*(1+shift) = 0.405 + k*shift
        # Using k=1.5: shift = 1.099/1.5 = 0.733; pos=-1 -> 1/(1+exp(1.5*1.733)) = 1/(1+exp(2.6)) = 0.069 too low
        # Using k=0.8: shift = 1.099/0.8 = 1.374; pos=-1 -> 1/(1+exp(0.8*2.374)) = 1/(1+exp(1.9)) = 0.13 too low
        # Alternative: use a linear-then-sigmoid: above min is linear mapped to [0.75, 1.0],
        # below min uses a decay function
        if pos >= -0.5:
            # Within training distribution or up to half a range below min:
            # map [-0.5, 1.0] -> [0.75, 0.95]
            # pos=-0.5 -> 0.75 (threshold + margin), pos=0 -> 0.817, pos=1 -> 0.95
            normalized = (pos + 0.5) / 1.5  # maps [-0.5, 1.0] -> [0, 1]
            score = 0.75 + 0.20 * min(max(normalized, 0.0), 1.0)
        else:
            # Well below training minimum: smooth sigmoid decay from 0.75.
            # Continuous at boundary (pos=-0.5 -> 0.75), decaying for lower pos.
            # At pos=-0.5: 1.50/(1+exp(0)) = 0.75 (continuous)
            # At pos=-1.5: 1.50/(1+exp(1.0)) ~ 0.55
            # At pos=-2.5: 1.50/(1+exp(2.0)) ~ 0.20
            shifted = -(pos + 0.5)  # 0 at boundary, positive for anomalous
            # Clip to prevent overflow in exp() for extremely anomalous clients
            shifted_clipped = min(shifted, 20.0)
            score = 1.50 / (1.0 + np.exp(1.0 * shifted_clipped))

        return float(np.clip(score, 0.0, 1.0))


# ============================================================================
# LSTM TEMPORAL EXPERT
# ============================================================================

class LSTMAutoencoder(nn.Module):
    """Lightweight LSTM autoencoder for temporal anomaly detection.

    Encoder: LSTM that compresses a sequence of feature vectors into a
    fixed-size hidden state.
    Decoder: LSTM that reconstructs the input sequence from the hidden state.

    Anomaly score = reconstruction MSE.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode.

        Args:
            x: (batch, seq_len, input_dim) tensor

        Returns:
            Reconstructed sequence of same shape as x.
        """
        # Encode
        _, (h_n, c_n) = self.encoder(x)

        # Decode: repeat the encoded hidden state for each time step
        seq_len = x.size(1)
        # Use the last hidden state as input to decoder, repeated
        decoder_input = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder(decoder_input, (h_n, c_n))

        # Project back to input dimension
        reconstructed = self.output_layer(decoder_out)
        return reconstructed


class LSTMTemporalExpert:
    """LSTM autoencoder for detecting temporal behavioral drift.

    Maintains per-client history of geometric feature vectors. After a
    warmup period, trains the LSTM autoencoder on sequences from accepted
    clients. Scores clients by reconstruction error — high error means
    the temporal pattern is anomalous.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        max_seq_len: int = 10,
        warmup_rounds: int = 5,
        refit_interval: int = 5,
        train_epochs: int = 20,
        learning_rate: float = 0.001,
        min_sequences: int = 5,
        seed: int = _RNG_SEED,
    ):
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.warmup_rounds = warmup_rounds
        self.refit_interval = refit_interval
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.min_sequences = min_sequences
        self.seed = seed

        torch.manual_seed(seed)
        self._model = LSTMAutoencoder(input_dim, hidden_dim, num_layers)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self._criterion = nn.MSELoss()

        # Per-client feature history: client_id -> deque of feature vectors
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_seq_len)
        )

        # Training buffer: sequences from accepted clients
        self._train_sequences: List[np.ndarray] = []
        self._is_fitted = False
        self._last_fit_round = 0

        # Running statistics for score normalization
        self._mse_history: List[float] = []
        # Training MSE distribution (set during fit)
        self._train_mse_max: float = 1.0
        self._train_mse_mean: float = 0.0
        self._train_mse_range: float = 1.0

    @property
    def is_ready(self) -> bool:
        return self._is_fitted

    def record_features(self, client_id: str, features: np.ndarray):
        """Record a feature vector for a client (called each round)."""
        self._history[client_id].append(features.copy())

    def get_sequence(self, client_id: str) -> Optional[np.ndarray]:
        """Get the temporal feature sequence for a client.

        Returns:
            (seq_len, input_dim) array, or None if insufficient history.
        """
        hist = self._history.get(client_id)
        if hist is None or len(hist) < 2:
            return None
        return np.stack(list(hist))

    def accumulate_training_sequence(self, client_id: str):
        """Add a client's current sequence to the training buffer."""
        seq = self.get_sequence(client_id)
        if seq is not None and len(seq) >= 2:
            self._train_sequences.append(seq.copy())

    def fit(self, server_round: int):
        """Train the LSTM autoencoder on accumulated sequences."""
        if len(self._train_sequences) < self.min_sequences:
            logger.debug(
                "LSTM: not enough sequences to fit (%d < %d)",
                len(self._train_sequences), self.min_sequences,
            )
            return

        torch.manual_seed(self.seed + server_round)

        # Pad sequences to same length for batching
        max_len = max(len(s) for s in self._train_sequences)
        padded = []
        for seq in self._train_sequences:
            if len(seq) < max_len:
                padding = np.zeros((max_len - len(seq), self.input_dim), dtype=np.float32)
                padded.append(np.vstack([padding, seq]))
            else:
                padded.append(seq[-max_len:])

        X = torch.tensor(np.stack(padded), dtype=torch.float32)

        self._model.train()
        for epoch in range(self.train_epochs):
            self._optimizer.zero_grad()
            reconstructed = self._model(X)
            loss = self._criterion(reconstructed, X)
            loss.backward()
            self._optimizer.step()

        self._is_fitted = True
        self._last_fit_round = server_round
        self._mse_history.clear()  # reset normalization stats after refit

        # Compute training MSE distribution for calibration
        self._model.eval()
        with torch.no_grad():
            reconstructed_eval = self._model(X)
            per_sample_mse = torch.mean((reconstructed_eval - X) ** 2, dim=(1, 2))
            self._train_mse_max = float(per_sample_mse.max())
            self._train_mse_mean = float(per_sample_mse.mean())
            self._train_mse_range = float(self._train_mse_max - per_sample_mse.min())

        logger.debug(
            "LSTM fitted on %d sequences at round %d (final loss=%.6f, "
            "train MSE max=%.6f)",
            len(self._train_sequences), server_round, loss.item(),
            self._train_mse_max,
        )

    def maybe_refit(self, server_round: int):
        """Refit if enough rounds have passed since last fit."""
        if server_round - self._last_fit_round >= self.refit_interval:
            self.fit(server_round)

    def score(self, client_id: str) -> float:
        """Score a client based on temporal reconstruction error.

        Returns value in [0, 1]. Higher = more trustworthy.

        Neutral score (0.85) is returned when the LSTM cannot make a
        judgment (not fitted, insufficient history). This is above the
        0.7 threshold because "no evidence of anomaly" should not be
        treated as suspicious.
        """
        if not self._is_fitted:
            return 0.85  # neutral/benign during warmup

        seq = self.get_sequence(client_id)
        if seq is None or len(seq) < 2:
            return 0.85  # not enough history — no reason to suspect

        # Forward pass
        self._model.eval()
        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, dim)
            reconstructed = self._model(x)
            mse = self._criterion(reconstructed, x).item()

        # Track MSE for adaptive normalization
        self._mse_history.append(mse)

        # Calibration anchored to training MSE distribution:
        # If MSE <= training max: this client's temporal pattern is within
        #   the range seen in training -> high trust score (>= 0.75)
        # If MSE > training max: anomalous temporal pattern -> score drops
        #
        # This prevents false positives from natural heterogeneity because
        # the training data includes sequences from diverse honest clients.
        if self._train_mse_range > 1e-12:
            # How far above the training max MSE is this client's MSE?
            overshoot = (mse - self._train_mse_max) / self._train_mse_range
        else:
            overshoot = 0.0

        # Tolerance for floating point comparison: values within 1% of
        # the training range above the max are treated as in-range.
        # This prevents false rejections from float32/float64 rounding.
        OVERSHOOT_TOLERANCE = 0.01

        if overshoot <= OVERSHOOT_TOLERANCE:
            # Within training range (or negligibly above): safe.
            # Map [0, train_max * (1 + tol)] -> [0.80, 0.95]
            if self._train_mse_max > 1e-12:
                ratio = min(mse / self._train_mse_max, 1.0)  # clamp to 1.0
            else:
                ratio = 0.0
            score = 0.95 - 0.15 * ratio  # low MSE -> 0.95, high-but-in-range -> 0.80
        else:
            # Clearly above training max: anomalous. Smooth decay from 0.80.
            # Uses a modified sigmoid that equals 0.80 at overshoot=TOLERANCE
            # (continuity with the in-range path) and decays toward 0 for
            # large overshoot.
            # At shifted=0: 1.60/(1+exp(0)) = 0.80 (continuous)
            # At shifted=0.5: ~0.60, shifted=1.0: ~0.43, shifted=2.0: ~0.19
            shifted = overshoot - OVERSHOOT_TOLERANCE
            # Clip to prevent overflow in exp() for very large overshoot values
            shifted_clipped = min(shifted, 20.0)
            score = 1.60 / (1.0 + np.exp(1.0 * shifted_clipped))

        return float(np.clip(score, 0.0, 1.0))


# ============================================================================
# EMA REPUTATION EXPERT (TGE′ long-memory expert)
# ============================================================================

class EMAReputationExpert:
    """EMA reputation expert — the SECOND long-memory leg of the TGE′ bank (GWU-53).

    Added ALONGSIDE the LSTM (which is unchanged); it does not replace it. The
    expert tracks one scalar reputation R per logical client, an exponential
    moving average of that client's per-round cold-start score with an
    absence-decay term:

        R_init(join)     = r_init                        (0.85, on first join and
                                                          on every fresh logical id)
        R <- R * absence_decay ** gap                    (gap = rounds missed)
        R_t              = alpha * R + (1 - alpha) * s_cs(t)

    r_init is the ensemble's NEUTRAL "no evidence" value (0.85, the same value
    the LSTM/GBDT return before they can judge), NOT TrustScore's 0.5: a client
    with zero evidence must score neutral with respect to the 0.7 operational
    cutoff, so a fresh EMA leg never filters an honest client while its
    reputation matures. (TrustScore's 0.5 belonged to TrustScore's own threshold
    semantics and does not transfer — GWU-53)

    Convention (amendment v1.7 §2.1, = TrustScore's ``_ema_decay`` semantics in
    rmc/defenses.py): ``ema_alpha`` is the weight RETAINED on the prior
    reputation per update; ``(1 - ema_alpha)`` weights the new observation.
    ADOPTED (not tuned) at TrustScore's validated 0.9, so ``ema_alpha=0.9`` is
    heavy smoothing — only 10% new evidence per round. This is
    RMCDetectionPlugin._temporal_score with s_cs as the input.

    Input signal: ``s_cs(t)`` is the cold-start (IsolationForest) expert's
    per-round score for the client — NOT the geometric fallback (the geometric
    reputation is disqualified: TrustScore's all-rounds trust_score AUC is
    INVERTED at S0/S1, results/20260723/component_attribution). The model
    accumulates the reputation only from forest-fit onward (the cold-start
    window is gbdt-covered anyway), passing the fitted forest's score as s_cs;
    before the forest fits the EMA stays neutral (no update, score None). The
    geometric-EMA remains an offline sensitivity — the logged per-row Family-S
    signals (update_norm, cos_to_median, L2_to_median) suffice to re-derive
    geometric_fallback_score.

    Absence gap is tracked internally from the last round each logical id was
    updated (``server_round - last_seen - 1``), so no scenario_manager is
    required — it works identically in the plugin and standalone paths.

    Identity semantics (the RMC-relevant property): R is keyed by the logical
    client id the model passes in. A reconnect under a fresh logical id
    (``client_N_newM``) has no entry, so it re-initialises at ``r_init`` with no
    absence decay — an identity reset genuinely resets reputation. Inherent to
    the keying; there is no reset code path to get wrong.
    """

    def __init__(
        self,
        ema_alpha: float = 0.9,
        absence_decay: float = 0.9,
        r_init: float = 0.85,
        seed: int = _RNG_SEED,
    ):
        if not (0.0 < ema_alpha < 1.0):
            raise ValueError(
                f"ema_alpha ({ema_alpha}) must be in the open interval (0, 1): "
                f"it is the retention weight on prior reputation (0 keeps no "
                f"memory, 1 never updates) — both defeat the reputation expert"
            )
        if not (0.0 < absence_decay <= 1.0):
            raise ValueError(
                f"absence_decay ({absence_decay}) must be in (0, 1]: it decays "
                f"reputation by this factor per missed round"
            )
        self.ema_alpha = ema_alpha
        self.absence_decay = absence_decay
        self.r_init = r_init
        self.seed = seed

        self._reputation: Dict[str, float] = {}
        self._last_participated: Dict[str, int] = {}  # last round each logical id PARTICIPATED (not TGE-scored)
        self._n_updates = 0

    def has_update(self, client_id: str) -> bool:
        """True once at least one update has been applied for this client."""
        return client_id in self._reputation

    def reputation(self, client_id: str) -> Optional[float]:
        """Current reputation R for a client, or None if never updated."""
        return self._reputation.get(client_id)

    def update(self, client_id: str, s_cs: float, server_round: int) -> float:
        """Fold this round's cold-start score into the client's reputation.

        Applies absence decay for any rounds the (same) logical id was absent
        from scenario participation, then the EMA blend. A first-ever appearance
        (or a fresh logical id after a reset) starts from ``r_init`` with no
        absence decay. Returns the updated reputation.

        Called once per participating client per round via the cohort
        observation path (TGEnsembleModel.observe_ema), for EVERY participant —
        not just the ones a downstream filter kept — so a client filtered by an
        upstream Krum layer still folds in its evidence and its
        ``_last_participated`` advances, and the gap reflects scenario
        participation rather than TGE-survival (GWU-53).
        """
        prev = self._reputation.get(client_id, self.r_init)
        last = self._last_participated.get(client_id)
        if last is not None:
            gap = server_round - last - 1
            if gap > 0:
                prev *= self.absence_decay ** gap
        r = self.ema_alpha * prev + (1.0 - self.ema_alpha) * s_cs
        r = float(np.clip(r, 0.0, 1.0))
        self._reputation[client_id] = r
        self._last_participated[client_id] = server_round
        self._n_updates += 1
        return r

    def score(self, client_id: str) -> Optional[float]:
        """Reputation-based trust in [0, 1], or None until the client has had
        at least one update (the null-until-first-update contract the signal
        log logs verbatim)."""
        return self._reputation.get(client_id)


# ============================================================================
# TENURE-GATED DECISION RULE
# ============================================================================

class TenureGatedDecisionRule:
    """Routes scoring between GBDT (cold-start) and LSTM (warm-start).

    For clients with tenure < min_tenure: pure GBDT score.
    For clients with tenure >= ramp_rounds: pure LSTM score.
    In between: smooth linear blend.

    The configured ramp_rounds is authoritative (amendment v1.6 § 2): the
    default 8 is the provisional canonical value; the final value is selected
    at the H2 dev gate by the v1.6 § 3 pre-registered protocol. Invalid
    configurations raise instead of being silently corrected.
    """

    def __init__(self, min_tenure: int = 2, ramp_rounds: int = 8):
        if ramp_rounds <= min_tenure:
            raise ValueError(
                f"ramp_rounds ({ramp_rounds}) must be > min_tenure ({min_tenure}): "
                f"the gate needs at least one blend step between pure-GBDT and pure-LSTM"
            )
        self.min_tenure = min_tenure
        self.ramp_rounds = ramp_rounds

    def compute_score(
        self,
        gbdt_score: float,
        lstm_score: float,
        tenure: int,
    ) -> float:
        """Blend GBDT and LSTM scores based on client tenure.

        Args:
            gbdt_score: Score from GBDT cold-start expert [0, 1].
            lstm_score: Score from LSTM temporal expert [0, 1].
            tenure: Number of rounds this client has been seen.

        Returns:
            Blended trust score in [0, 1].
        """
        if tenure < self.min_tenure:
            # Pure GBDT — no temporal history available
            return gbdt_score

        if tenure >= self.ramp_rounds:
            # Pure LSTM — sufficient temporal history
            return lstm_score

        # Linear blend
        tenure_ratio = (tenure - self.min_tenure) / max(1, self.ramp_rounds - self.min_tenure)
        cold_weight = 1.0 - tenure_ratio
        warm_weight = tenure_ratio
        return cold_weight * gbdt_score + warm_weight * lstm_score


# ============================================================================
# GEOMETRIC FALLBACK (warmup scoring)
# ============================================================================

def geometric_fallback_score(features: np.ndarray) -> float:
    """Simple geometric score used during warmup before experts are fitted.

    Combines z-distance, cosine similarity, and norm deviation into a
    single trust score in [0, 1].

    Args:
        features: Feature vector from extract_geometric_features.
            features[0] = z_distance, features[1] = cos_sim, features[2] = norm_dev

    Returns:
        Trust score in [0, 1].
    """
    z_dist = features[0]    # higher z-score = more anomalous
    cos_sim = features[1]   # higher = more similar to median (good)
    norm_dev = features[2]  # higher = more deviation (bad)

    # Z-distance component: map to [0, 1] where low z = high trust
    z_component = float(np.clip(1.0 - z_dist / 3.0, 0.0, 1.0))

    # Cosine component: already roughly in [0, 1] for reasonable updates
    cos_component = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))

    # Norm deviation component
    norm_component = float(np.clip(1.0 - norm_dev, 0.0, 1.0))

    # Equal-weighted average
    return (z_component + cos_component + norm_component) / 3.0


# ============================================================================
# TG-ENSEMBLE MODEL (Main Class)
# ============================================================================

class TGEnsembleModel:
    """Tenure-Gated Ensemble model combining GBDT and LSTM experts.

    Orchestrates feature extraction, expert scoring, tenure gating,
    and training lifecycle. This is the core model used by both the
    standalone defense (rmc/defenses.py) and the Flower plugin
    (flowerfl/byzantine_defense.py).

    Lifecycle per round:
        1. extract_features() for each client
        2. score_client() for each client -> trust score
        3. after accept/reject decisions, call record_accepted() for accepted
        4. call on_round_end() to trigger refitting if needed
    """

    def __init__(
        self,
        num_features: int = 12,
        warmup_rounds: int = 3,
        threshold: float = 0.7,
        gbdt_refit_interval: int = 5,
        lstm_refit_interval: int = 5,
        lstm_hidden_dim: int = 32,
        lstm_num_layers: int = 1,
        lstm_max_seq_len: int = 10,
        lstm_train_epochs: int = 20,
        ramp_rounds: int = 8,
        long_memory_expert: str = "lstm",
        ema_alpha: float = 0.9,
        ema_absence_decay: float = 0.9,
        seed: int = _RNG_SEED,
    ):
        if long_memory_expert not in ("lstm", "ema", "bank"):
            raise ValueError(
                f"long_memory_expert ({long_memory_expert!r}) must be 'lstm' "
                f"(incumbent TGE), 'ema' (EMA leg only — component isolation), "
                f"or 'bank' (TGE′: min(LSTM, EMA), GWU-53)"
            )
        # Validate ema_alpha loudly even in "lstm" mode (mirrors the ramp
        # governance canary): a bad alpha must never silently reach a run.
        if not (0.0 < ema_alpha < 1.0):
            raise ValueError(
                f"ema_alpha ({ema_alpha}) must be in the open interval (0, 1)"
            )
        self.num_features = num_features
        self.warmup_rounds = warmup_rounds
        self.threshold = threshold
        self.seed = seed
        # Long-memory mode (GWU-53). "lstm" is the incumbent and is bit-identical
        # (self.ema stays None; no new code path runs). "ema"/"bank" add the EMA
        # reputation leg; "bank" is the deployed TGE′ combiner min(LSTM, EMA).
        self.long_memory_expert = long_memory_expert
        self.ema_alpha = ema_alpha
        self.ema_absence_decay = ema_absence_decay

        # Initialize experts
        # GBDT refits every 2 rounds to stay current with distribution shift
        self.gbdt = GBDTColdStartExpert(
            warmup_rounds=warmup_rounds,
            refit_interval=min(gbdt_refit_interval, 2),
            seed=seed,
        )
        # The LSTM long-memory expert is ALWAYS present and unchanged (incumbent
        # behavior). TGE′ (GWU-53) adds a SECOND long-memory leg — the EMA
        # reputation expert — alongside it; the tenure gate consumes a single
        # long-memory score derived from these legs by `_long_memory_score`
        # according to `long_memory_expert`. In "lstm" mode self.ema is None and
        # nothing new runs.
        self.lstm = LSTMTemporalExpert(
            input_dim=num_features,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            max_seq_len=lstm_max_seq_len,
            warmup_rounds=warmup_rounds + 2,  # LSTM needs more history
            refit_interval=lstm_refit_interval,
            train_epochs=lstm_train_epochs,
            seed=seed,
        )
        self.ema: Optional[EMAReputationExpert] = (
            EMAReputationExpert(
                ema_alpha=ema_alpha,
                absence_decay=ema_absence_decay,
                seed=seed,
            )
            if long_memory_expert in ("ema", "bank")
            else None
        )
        # Configured ramp is authoritative — the old silent max(ramp, 8) floor
        # is removed per amendment v1.6 § 2 (a dev-gate-selected ramp below 8
        # must actually deploy). The LSTM-data-sufficiency concern the floor
        # encoded now lives in the provisional default of 8 and in the v1.6
        # § 3 selection protocol that will lock the final value empirically.
        self.gate = TenureGatedDecisionRule(
            min_tenure=2,
            ramp_rounds=ramp_rounds,
        )

        # Track tenure: client_id -> first-seen round
        self._first_seen: Dict[str, int] = {}
        # Layer boundaries for feature extraction (discovered from first update)
        self._layer_boundaries: Optional[List[int]] = None
        # Current round features cache
        self._round_features: Dict[str, np.ndarray] = {}

    @property
    def long_memory_combiner(self) -> str:
        """Identity of the combiner the gate's long-memory leg uses, for
        provenance. 'min' only in bank mode; otherwise the single leg's name."""
        return "min" if self.long_memory_expert == "bank" else self.long_memory_expert

    def _long_memory_score(
        self, lstm_score: float, ema_score: Optional[float]
    ) -> float:
        """The combined long-memory score when BOTH legs are available, per mode.

        - "lstm": the LSTM score (incumbent — ema_score is ignored).
        - "ema": the EMA reputation.
        - "bank": min(LSTM, EMA) — the zero-parameter fail-safe combiner; an
          attacker must fool BOTH accumulators to raise the long-memory leg.

        Both raw leg scores are logged separately (tge_lstm_score,
        tge_ema_score) so the combiner is reconstructable offline.
        """
        if self.long_memory_expert == "lstm":
            return lstm_score
        if self.long_memory_expert == "ema":
            return ema_score
        return min(lstm_score, ema_score)

    def _resolve_long_memory(
        self, lstm_score: float, lstm_ready: bool, ema_score: Optional[float]
    ) -> Optional[float]:
        """The long-memory score the gate consumes given each leg's readiness,
        or None when no long-memory leg is engaged yet (=> GBDT-only).

        - "lstm": the LSTM once ready.
        - "ema": the EMA once it has had an update.
        - "bank": min(LSTM, EMA) once both ready; before the LSTM is ready it
          DEGRADES to the EMA leg alone; before the EMA's first update it is
          None (neutral, GBDT-only).
        """
        mode = self.long_memory_expert
        if mode == "lstm":
            return lstm_score if lstm_ready else None
        if mode == "ema":
            return ema_score  # None until the EMA's first update
        # bank
        if lstm_ready and ema_score is not None:
            return self._long_memory_score(lstm_score, ema_score)  # min
        if lstm_ready:
            return lstm_score  # transient: EMA not yet updated
        return ema_score  # degraded EMA-only window (None before first update)

    def _ensure_first_seen(self, client_id: str, server_round: int):
        """Track first-seen round for tenure calculation."""
        if client_id not in self._first_seen:
            self._first_seen[client_id] = server_round

    def get_tenure(self, client_id: str, server_round: int) -> int:
        """Get tenure (number of rounds since first seen, 1-indexed)."""
        first = self._first_seen.get(client_id, server_round)
        return server_round - first + 1

    def discover_layer_boundaries(self, param_shapes: List[Tuple[int, ...]]):
        """Discover layer boundaries from parameter shapes (called once)."""
        if self._layer_boundaries is None:
            self._layer_boundaries = compute_layer_boundaries(param_shapes)

    def extract_features(
        self,
        client_flat: np.ndarray,
        all_flat: List[np.ndarray],
    ) -> np.ndarray:
        """Extract geometric features for a single client.

        Args:
            client_flat: Flattened parameter vector for the client.
            all_flat: Flattened parameter vectors for ALL clients this round.

        Returns:
            Feature vector of shape (num_features,).
        """
        return extract_geometric_features(
            client_flat, all_flat, self._layer_boundaries
        )

    def score_client(
        self,
        client_id: str,
        features: np.ndarray,
        server_round: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Score a single client using the tenure-gated ensemble.

        During warmup, ALL clients are accepted (score = 1.0) so that the
        experts learn the true feature distribution including natural
        heterogeneity across clients. This prevents the self-reinforcing
        rejection loop where a geometrically-distant but honest client
        gets excluded from training data and perpetually flagged.

        Args:
            client_id: Unique identifier for the client.
            features: Geometric feature vector from extract_features().
            server_round: Current FL round.

        Returns:
            (trust_score, details_dict) where trust_score is in [0, 1].
        """
        self._ensure_first_seen(client_id, server_round)
        tenure = self.get_tenure(client_id, server_round)

        # Record features for LSTM temporal history (always, even during warmup)
        self.lstm.record_features(client_id, features)

        # Cache features for this round (used by record_accepted)
        self._round_features[client_id] = features

        # During warmup: accept all clients unconditionally so both experts
        # accumulate the full distribution. Return a permissive score (1.0)
        # so no clients are rejected before the experts have been trained. The
        # EMA leg has not accumulated yet (it starts at forest-fit; see below),
        # so its logged score is null here.
        if server_round <= self.warmup_rounds:
            geo = geometric_fallback_score(features)
            return 1.0, {
                "gbdt_score": None,
                "lstm_score": None,
                "ema_score": None,
                "fallback_score": round(geo, 4),
                "final_score": 1.0,
                "tenure": tenure,
                "phase": "warmup",
            }

        # Score with available experts
        gbdt_score = self.gbdt.score(features)
        lstm_score = self.lstm.score(client_id)

        # If GBDT is not ready, fall back to geometric scoring. The EMA leg does
        # not accumulate until the forest fits, so its score is null here too.
        if not self.gbdt.is_ready:
            fallback = geometric_fallback_score(features)
            return fallback, {
                "gbdt_score": None,
                "lstm_score": None,
                "ema_score": None,
                "fallback_score": round(fallback, 4),
                "final_score": round(fallback, 4),
                "tenure": tenure,
                "phase": "pre_gbdt",
            }

        # TGE′ EMA leg (ema/bank modes only — GWU-53). READ ONLY here: the
        # reputation is updated cohort-wide by observe_ema (driven by the
        # plugin's observe_cohort hook / the standalone's cohort loop) BEFORE
        # scoring, for EVERY participant — so a Krum-filtered client still folds
        # in its evidence, and a survivor is updated exactly once (never
        # double-counted, GWU-53). It is null until the EMA's
        # first post-forest-fit update. Kept wholly inside this guard so "lstm"
        # mode runs no new code and is bit-identical.
        ema_score: Optional[float] = None
        if self.ema is not None:
            ema_score = self.ema.score(client_id)

        # Resolve the long-memory leg the gate consumes (None => no long-memory
        # leg engaged yet -> GBDT-only). lstm mode: the LSTM once ready. bank
        # mode: min(LSTM, EMA) once both ready, degrading to the EMA leg alone
        # before the LSTM is ready (and to none before the EMA's first update).
        long_mem = self._resolve_long_memory(lstm_score, self.lstm.is_ready, ema_score)

        if long_mem is None:
            final_score = gbdt_score
            details = {
                "gbdt_score": round(gbdt_score, 4),
                "lstm_score": None,
                "ema_score": _round_or_none(ema_score),
                "final_score": round(final_score, 4),
                "tenure": tenure,
                "phase": "gbdt_only",
                "gate": "gbdt",
            }
        else:
            # A long-memory leg is engaged: tenure-gate it against the
            # cold-start score. Both raw leg scores are logged separately
            # (tge_lstm_score, tge_ema_score) so the combiner is reconstructable
            # offline. lstm_score is logged only when the LSTM is actually ready
            # (null during the bank degraded window). The `gate` label reports
            # the tenure regime (which side of the ramp), unchanged from the
            # incumbent, so lstm-mode rows are bit-identical.
            lstm_ready = self.lstm.is_ready
            final_score = self.gate.compute_score(gbdt_score, long_mem, tenure)
            details = {
                "gbdt_score": round(gbdt_score, 4),
                "lstm_score": round(lstm_score, 4) if lstm_ready else None,
                "ema_score": _round_or_none(ema_score),
                "final_score": round(final_score, 4),
                "tenure": tenure,
                "phase": "active",
                "gate": "gbdt" if tenure < self.gate.min_tenure
                        else ("lstm" if tenure >= self.gate.ramp_rounds else "blend"),
            }

        return final_score, details

    def record_scored(self, client_id: str, server_round: int):
        """Record that a client was scored — feeds features to GBDT.

        During warmup: only includes clients whose geometric fallback score
        is above a conservative threshold (0.5). This prevents attackers
        who are active from round 1 from contaminating the GBDT's initial
        training data. The threshold is deliberately low to avoid excluding
        legitimate-but-distant honest clients.

        Post-warmup: includes ALL clients' features (accepted and rejected)
        to prevent confirmation bias. The GBDT is already fitted and can
        distinguish outliers, so including all data helps it adapt to
        distribution shift without creating a self-reinforcing rejection loop.

        Call this for EVERY client after scoring, regardless of accept/reject.
        """
        features = self._round_features.get(client_id)
        if features is not None:
            if server_round <= self.warmup_rounds:
                # During warmup: pre-filter using geometric score to avoid
                # contaminating GBDT training data with obvious attackers
                geo_score = geometric_fallback_score(features)
                if geo_score >= 0.5:
                    self.gbdt.accumulate(features)
                else:
                    logger.debug(
                        "GBDT warmup filter: excluded %s (geo=%.3f)",
                        client_id, geo_score,
                    )
            else:
                # Post-warmup: include all features to prevent confirmation bias
                self.gbdt.accumulate(features)

    def record_accepted(self, client_id: str, server_round: int):
        """Record that a client was accepted — feeds LSTM training data.

        Only accepted clients' temporal sequences are used for LSTM training
        to keep the autoencoder focused on "normal" temporal patterns. The
        GBDT is fed by record_scored() for all clients.

        During warmup, an additional geometric check is applied: clients
        whose last recorded features had a low geometric score are excluded
        from LSTM training even though they are "accepted" (all warmup clients
        are accepted). This mirrors the GBDT warmup filter.
        """
        if server_round <= self.warmup_rounds:
            # During warmup: additional geometric gate for LSTM training
            features = self._round_features.get(client_id)
            if features is not None:
                geo_score = geometric_fallback_score(features)
                if geo_score < 0.5:
                    return  # exclude from LSTM training
        self.lstm.accumulate_training_sequence(client_id)

    def observe_ema(self, client_id: str, features: np.ndarray, server_round: int):
        """Fold one participant's cold-start evidence into its EMA reputation.

        The cohort-observation path (GWU-53): the plugin's
        observe_cohort hook / the standalone's cohort loop call this for EVERY
        participant this round — BEFORE scoring — so a client filtered by an
        upstream Krum layer still has its reputation evolve (once-per-
        participating-round EMA, cohort-wide, independent of upstream filtering),
        and each participant is updated exactly once (score_client only READS).

        ISOLATION (hard requirement): touches ONLY the EMA leg. It does not feed
        the IsolationForest training buffer (record_scored), the LSTM history
        (record_features), the tenure counters (_ensure_first_seen), or
        _last_details — so the signal-log scored-row population and every other
        expert are byte-identical to not running it. gbdt.score() is
        side-effect-free (no accumulation). No-op in lstm mode or before the
        forest fits (s_cs would be the neutral placeholder — the pre-fit window
        is gbdt-covered, EMA stays null).
        """
        if self.ema is None or not self.gbdt.is_ready:
            return
        s_cs = self.gbdt.score(features)
        self.ema.update(client_id, s_cs, server_round)

    def on_round_end(self, server_round: int):
        """End-of-round bookkeeping: fit/refit experts as needed."""
        # Clear round cache
        self._round_features.clear()

        # Initial fit after warmup completes
        if server_round == self.warmup_rounds:
            self.gbdt.fit(server_round)
            # LSTM fits later (needs more accumulated sequences)

        if server_round == self.warmup_rounds + 2:
            self.lstm.fit(server_round)

        # Periodic refitting
        if server_round > self.warmup_rounds:
            self.gbdt.maybe_refit(server_round)
        if server_round > self.warmup_rounds + 2:
            self.lstm.maybe_refit(server_round)
