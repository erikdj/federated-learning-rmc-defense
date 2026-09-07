"""
ClientApp Module - Flower Client Application for FlowerFL.

Uses NumPyClient pattern for compatibility with server strategies,
wrapped in the modern ClientApp framework.
"""

import math
import torch
from collections import OrderedDict
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

import flowerfl.task as task_module
from flowerfl import persistent_optimizer as po
from flowerfl.fingerprint_emission import (
    FINGERPRINT_METRIC_KEY,
    M_FP,
    compute_round_fingerprint_payload,
)
from flowerfl.seeding import seed_everything, derive_seed
from flowerfl.task import (
    Net,
    SzelagNet,
    train,
    train_brfss,
    train_label_flip,
    train_brfss_label_flip,
    add_parameter_noise,
    test,
    test_brfss,
    load_data,
    detect_input_shape,
    get_malicious_clients,
    is_brfss_dataset,
    create_model,
)


def _as_bool(value) -> bool:
    """Coerce a Flower run-config value to bool.

    Run-config values arrive as native bools from pyproject or as strings from
    CLI/extra_run_config overrides; accept the common truthy string spellings
    and treat everything else as False (the incumbent default).
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class FlowerClient(NumPyClient):
    """NumPy client for federated learning with malicious client support."""

    def __init__(self, trainloader, valloader, net, is_malicious=False,
                 attack_type="label_flip", partition_id=0, use_brfss=False,
                 lr: float = 0.01, local_epochs: int = 1,
                 weight_decay: float = 3e-3,
                 optimizer_state: str = "reset",      # NEW: "reset" | "persistent"
                 node_id: "int | None" = None,        # NEW: Flower-assigned id for keying
                 base_seed: int = 42,                 # experiment seed
                 max_steps: "int | None" = None,      # Stage-F §4: update-matching cap
                 arm_label: "str | None" = None,      # Stage-F: arm name (loud errors/manifest)
                 update_match: bool = False,          # Stage-F §4: update-match flag
                 weight_mode: str = "resampled",      # Stage-F §5: {resampled, original}
                 n_orig: "int | None" = None,         # Stage-F §5: pre-resample train count
                 prep_info: "dict | None" = None,     # Stage-F §5: resampling manifest fragment
                 semantic_target: bool = False,       # Stage-F §6: semantic attack-class policy
                 expected_arm: "dict | None" = None,  # Stage-F §5: declared-arm compliance check
                 fingerprint_pool=None,               # H3 §4.2: raw float64 pool (None = OFF)
                 m_fp: int = M_FP):                   # H3 §4.1: per-round bootstrap draw size
        self.trainloader = trainloader
        self.valloader = valloader
        self.net = net
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.partition_id = partition_id
        self.use_brfss = use_brfss
        self.lr = lr
        self.local_epochs = local_epochs
        self.weight_decay = weight_decay
        self.optimizer_state = optimizer_state
        self.node_id = node_id
        self.base_seed = base_seed
        self.max_steps = max_steps
        self.arm_label = arm_label
        self.update_match = update_match
        self.weight_mode = weight_mode
        self.n_orig = n_orig
        self.prep_info = prep_info
        self.semantic_target = semantic_target
        self.expected_arm = expected_arm
        # H3 fingerprint emission (EMISSION_CONTRACT § 4). None (the default) is
        # the incumbent: no pool is held, no draw is made, and no metrics key is
        # added, so a bare run stays byte-identical to every sealed H1/H2 run.
        self.fingerprint_pool = fingerprint_pool
        self.m_fp = int(m_fp)
        self._step_metrics = {}
        if optimizer_state == "persistent" and node_id is None:
            raise ValueError("optimizer_state='persistent' requires a node_id")

    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train model on local data.

        Attack behavior is determined by:
        1. Per-round config from ScenarioStrategy (attack_type in config dict)
        2. Fallback to static is_malicious flag (legacy non-scenario mode)
        """
        # seed all RNGs to a per-(client, round) value BEFORE any
        # stochastic op (dropout masks, DataLoader shuffle, attack noise), so
        # each client/round has a distinct-but-reproducible stream. server_round
        # arrives via FitIns.config (ScenarioStrategy injects it); it defaults to
        # 0 on non-scenario paths that don't thread it, which still yields
        # determinism (just round-invariant training draws).
        server_round = int(config.get("server_round", 0)) if config else 0
        seed_everything(derive_seed(self.base_seed, self.partition_id, server_round))

        # H3 (EMISSION_CONTRACT § 4.1): compute the fingerprint BEFORE training,
        # so a training failure cannot silently suppress the emission that gate
        # (e) requires for 100% of participating client-rounds. Placed AFTER
        # seed_everything on purpose: any accidental global-RNG consumption in
        # the fingerprint path would then shift the training draws, which is
        # exactly what the RNG-isolation regression test detects (§ 4.3).
        fingerprint_payload = self._round_fingerprint_payload(server_round)

        self.set_parameters(parameters)

        # Stage-F §4: fresh per-fit step capture. Every training path writes its
        # optimizer-step count here (via metrics_out) so the client can record the
        # actual step count in the resampling manifest and assert it hit the cap.
        self._step_metrics = {}

        # Per-round attack config from ScenarioStrategy takes priority
        config_attack = config.get("attack_type", None) if config else None

        # NOTE: persistent_optimizer routing is deliberately NOT applied to the
        # adversarial branches below. Per spec §9.1, the keying choice is
        # methodologically inert in the adversarial population — ALIE and other
        # attacks construct the malicious update from honest gradient statistics
        # (or replace the gradient entirely), not from local optimizer state.
        # Adversarial clients therefore always receive Flower-default per-round
        # Adam reset regardless of self.optimizer_state. See spec §9.1 for the
        # methodological rationale.
        if config_attack and config_attack != "":
            # Scenario-driven attack
            train_loss = self._execute_attack(parameters, config_attack, config)
        elif self.is_malicious:
            # Legacy static malicious mode
            if self.attack_type == "label_flip":
                if self.use_brfss:
                    train_loss = train_brfss_label_flip(self.net, self.trainloader, lr=self.lr, weight_decay=self.weight_decay, **self._stage_f_train_kwargs())
                else:
                    # legacy static branch equalized the same way as
                    # the scenario-driven label_flip dispatch below.
                    train_loss = train_label_flip(self.net, self.trainloader, epochs=self.local_epochs, lr=self.lr, weight_decay=self.weight_decay, **self._stage_f_train_kwargs())
            elif self.attack_type == "noise":
                if self.use_brfss:
                    train_loss = train_brfss(self.net, self.trainloader, epochs=self.local_epochs, lr=self.lr, weight_decay=self.weight_decay, **self._stage_f_train_kwargs())
                else:
                    train_loss = train(self.net, self.trainloader, epochs=self.local_epochs, lr=self.lr, weight_decay=self.weight_decay, **self._stage_f_train_kwargs())
                add_parameter_noise(self.net, noise_scale=0.1)
            else:
                if self.use_brfss:
                    train_loss = train_brfss(self.net, self.trainloader, epochs=self.local_epochs, lr=self.lr, weight_decay=self.weight_decay, **self._stage_f_train_kwargs())
                else:
                    train_loss = train(self.net, self.trainloader, epochs=self.local_epochs, lr=self.lr, weight_decay=self.weight_decay, **self._stage_f_train_kwargs())
        else:
            optimizer = None
            if self.optimizer_state == "persistent":
                optimizer = torch.optim.Adam(self.net.parameters(),
                                             lr=self.lr, weight_decay=self.weight_decay)
                po.load_state(self.node_id, optimizer)
            if self.use_brfss:
                train_loss = train_brfss(self.net, self.trainloader, epochs=self.local_epochs,
                                         lr=self.lr, weight_decay=self.weight_decay,
                                         optimizer=optimizer, **self._stage_f_train_kwargs())
            else:
                train_loss = train(self.net, self.trainloader, epochs=self.local_epochs,
                                   lr=self.lr, weight_decay=self.weight_decay,
                                   optimizer=optimizer, **self._stage_f_train_kwargs())
            if self.optimizer_state == "persistent":
                po.save_state(self.node_id, optimizer)

        metrics = {"train_loss": train_loss, "partition_id": float(self.partition_id)}
        # H3 § 4.4: transport the 180-dim vector under the key the FP plugin
        # reads. Absent entirely when the arm is off — no schema change at all.
        if fingerprint_payload is not None:
            metrics[FINGERPRINT_METRIC_KEY] = fingerprint_payload
        # Stage-F §5: report num_examples per weight-mode — ``original`` reports
        # the pre-resampling row count (n_orig) so every arm carries the SAME
        # FedAvg aggregation mass as off; ``resampled`` (default) keeps the
        # incumbent resampled-count behaviour byte-for-byte.
        reported_examples = self._reported_num_examples()
        # Stage-F §5: emit + self-check the durable per-client resampling manifest.
        self._emit_resampling_manifest(metrics, reported_examples)
        return (
            self.get_parameters(config={}),
            reported_examples,
            metrics,
        )

    def _round_fingerprint_payload(self, server_round: int) -> "str | None":
        """This round's encoded 180-dim fingerprint, or None when the arm is off.

        A pure function of ``(partition_id, base_seed, server_round)`` and the
        client's own fixed raw pool (EMISSION_CONTRACT § 4.3): it reads no model,
        no server state, no defense config and no enforcement outcome. That is
        what makes the fingerprint stream **arm-invariant by construction** — the
        same (scenario, seed) under ``tgefp`` and ``krumtgefp`` must produce
        byte-identical payloads for every client-round — and enforcement-
        independent, which the § 5.1 INTEGRITY ASSERTION requires.

        Failures are LOUD: ``compute_round_fingerprint_payload`` raises rather
        than returning nothing, and nothing is caught here, so a client that
        cannot fingerprint fails the unit instead of quietly breaking gate (e)'s
        100 %-emission requirement.
        """
        if self.fingerprint_pool is None:
            return None
        return compute_round_fingerprint_payload(
            self.fingerprint_pool,
            base_seed=self.base_seed,
            partition_id=self.partition_id,
            server_round=server_round,
            m_fp=self.m_fp,
        )

    def _stage_f_train_kwargs(self) -> dict:
        """Common Stage-F kwargs threaded into every train* call (§4).

        Carries the update-matching cap (``max_steps``; None on the canonical
        path ⇒ the original epochs loop runs byte-for-byte), the client/arm
        identity for the loud empty-loader error, and the step-count capture.
        """
        return {
            "max_steps": self.max_steps,
            "partition_id": self.partition_id,
            "arm": self.arm_label,
            "metrics_out": self._step_metrics,
        }

    def _reported_num_examples(self) -> int:
        """FedAvg aggregation mass this client reports (§5).

        ``weight-mode=original`` reports the pre-resampling row count so every arm
        carries the SAME mass as off (isolating balance from aggregation weight);
        ``resampled`` (default / off arm) keeps the incumbent
        ``len(trainloader.dataset)`` behaviour byte-for-byte.
        """
        if self.weight_mode == "original" and self.n_orig is not None:
            return int(self.n_orig)
        return len(self.trainloader.dataset)

    def _emit_resampling_manifest(self, metrics: dict, reported_examples: int) -> None:
        """Assemble, self-check, and emit the durable per-client manifest (§5).

        Runs the LOUD arm-compliance assertions (actual_steps == max_steps under
        update-match; reported num_examples == n_orig under original weight-mode;
        after-class-counts match the semantic policy) — a violation raises and
        FAILS the unit rather than writing a false record. Emitted both as a fit
        metric (``resampling_manifest`` JSON, NOT reused as aggregation weight) and
        a parseable ``[MANIFEST]`` stdout line for the run record. Inert for legacy
        callers that never requested ``prep_info``.
        """
        if self.prep_info is None:
            return
        import json
        from flowerfl.resampling_manifest import build_manifest_row, assert_arm_compliance

        row = build_manifest_row(
            partition_id=self.partition_id,
            arm=self.arm_label,
            weight_mode=self.weight_mode,
            update_match=self.update_match,
            num_examples=reported_examples,
            max_steps=self.max_steps,
            actual_steps=self._step_metrics.get("steps_taken"),
            semantic_policy=self.semantic_target,
            prep_info=self.prep_info,
        )
        # expected_arm makes the declared-vs-manifest branch fire in PRODUCTION
        # (not just tests): a unit whose recorded arm config
        # disagrees with what actually ran fails loudly instead of writing a
        # mislabelled row.
        assert_arm_compliance(row, expected=self.expected_arm)
        metrics["resampling_manifest"] = json.dumps(row)
        print(f"[MANIFEST] {json.dumps(row)}")

    def _honest_train(self):
        """Honest training dispatch based on dataset type.

        Uses per-client lr and local_epochs set via run_config (Phase 2 P2.0/P2.2).
        """
        if self.use_brfss:
            return train_brfss(self.net, self.trainloader, epochs=self.local_epochs, lr=self.lr, weight_decay=self.weight_decay, **self._stage_f_train_kwargs())
        else:
            return train(self.net, self.trainloader, epochs=self.local_epochs, lr=self.lr, weight_decay=self.weight_decay, **self._stage_f_train_kwargs())

    def _execute_attack(self, pre_train_params, attack_type, config):
        """Execute a scenario-driven attack based on config dict."""
        if attack_type == "gaussian_noise" or attack_type == "gaussian_noise_loud":
            sigma = float(config.get("attack_sigma", 1.0))
            train_loss = self._honest_train()
            add_parameter_noise(self.net, noise_scale=sigma)
            return train_loss
        elif attack_type == "norm_matched_noise":
            # Save pre-training weights
            pre_weights = [p.clone() for p in self.net.parameters()]
            train_loss = self._honest_train()
            # Compute honest update norm
            update_norm = 0.0
            for pre_w, post_w in zip(pre_weights, self.net.parameters()):
                update_norm += torch.norm(post_w.data - pre_w).item() ** 2
            update_norm = update_norm ** 0.5
            # Replace with noise of same norm
            with torch.no_grad():
                noise_parts = []
                total_noise_norm = 0.0
                for pre_w, post_w in zip(pre_weights, self.net.parameters()):
                    noise = torch.randn_like(post_w)
                    noise_parts.append(noise)
                    total_noise_norm += torch.norm(noise).item() ** 2
                total_noise_norm = total_noise_norm ** 0.5
                scale = update_norm / max(total_noise_norm, 1e-8)
                for pre_w, post_w, noise in zip(pre_weights, self.net.parameters(), noise_parts):
                    post_w.copy_(pre_w + noise * scale)
            return train_loss
        elif attack_type == "alie":
            # Real ALIE (Baruch et al., 2019) — client trains honestly.
            # The server replaces this client's update with the cross-client
            # statistical attack in ScenarioStrategy._maybe_apply_real_alie.
            return self._honest_train()
        elif attack_type == "label_flip":
            if self.use_brfss:
                return train_brfss_label_flip(self.net, self.trainloader, **self._stage_f_train_kwargs())
            else:
                # pass epochs=local_epochs exactly as _honest_train
                # does for train — label_flip attackers previously ran a
                # single natural pass (5x under-trained vs honest clients).
                return train_label_flip(self.net, self.trainloader, epochs=self.local_epochs, **self._stage_f_train_kwargs())
        else:
            # Unknown attack type — train honestly
            return self._honest_train()

    def evaluate(self, parameters, config):
        """Evaluate model on local validation data."""
        self.set_parameters(parameters)
        if self.use_brfss:
            loss, accuracy, precision, recall, f1 = test_brfss(self.net, self.valloader)
        else:
            loss, accuracy, precision, recall, f1 = test(self.net, self.valloader)
        return (
            float(loss),
            len(self.valloader.dataset),
            {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )


def client_fn(context: Context):
    """Factory function to create a client instance."""
    # Extract configuration from context
    run_config = context.run_config
    dataset = run_config.get("dataset", "cic")
    batch_size = int(run_config.get("batch-size", 32))
    malicious_fraction = float(run_config.get("malicious-fraction", 0.0))
    attack_type = run_config.get("attack-type", "label_flip")
    max_samples = int(run_config.get("max-samples", 0))  # 0 = use default

    # Training hparams (v2 P2.2 grid search): defaults match pyproject.toml
    lr = float(run_config.get("learning-rate", 0.01))
    local_epochs = int(run_config.get("local-epochs", 1))
    weight_decay = float(run_config.get("weight-decay", 3e-3))
    base_seed = int(run_config.get("seed", 42))  # per-(client, round) RNG base

    # Get partition ID from node config
    partition_id = int(context.node_config.get("partition-id", 0))

    # SMOTE study knob : flag-gated per-client training-split oversampling.
    # Default OFF; when enabled, config is validated LOUDLY here at parse time
    # (before any data load / training) so an unknown variant or invalid target
    # fails fast rather than silently downgrading. Applied uniformly to ALL
    # clients incl. malicious (attackers control their own pipeline; scenario
    # semantics unchanged). Seed is per-(base_seed, client): load_data runs once
    # per client per run, before the server round is known.
    smote_enabled = _as_bool(run_config.get("smote-enabled", False))
    smote_variant = str(run_config.get("smote-variant", "smote"))
    smote_target = run_config.get("smote-target", "balanced")
    if smote_enabled:
        from flowerfl.smote_resampler import validate_smote_variant, normalize_smote_target
        validate_smote_variant(smote_variant)
        smote_target = normalize_smote_target(smote_target)

    # Stage-F knobs (DESIGN_STAGE_F §4/§5/§6). All default to the incumbent so a
    # bare run is byte-identical: update-match OFF (epochs loop), weight-mode
    # resampled (report post-resample count), semantic-target OFF (legacy min/max).
    update_match = _as_bool(run_config.get("update-match", False))
    from flowerfl.resampling_manifest import validate_weight_mode
    weight_mode = validate_weight_mode(str(run_config.get("weight-mode", "resampled")))
    smote_semantic_target = _as_bool(run_config.get("smote-semantic-target", False))
    # m1 leakage fix (opt-in, default OFF): when set, load_data fits the Z-score
    # on the seed-42 training rows only instead of all rows (train+val+test). Off
    # by default so the canonical pipeline is byte-identical to every sealed run.
    normalize_train_only = _as_bool(run_config.get("normalize-train-only", False))
    # Arm identity for the loud empty-loader error and the manifest.
    arm_label = f"{smote_variant}@{smote_target}" if smote_enabled else "off"
    # Declared arm config for the §5 compliance assertion.
    # variant/target are None on the off arm — matching the manifest fragment
    # load_data records (which only sets them when SMOTE is enabled).
    expected_arm = {
        "variant": smote_variant if smote_enabled else None,
        "target_fraction": smote_target if smote_enabled else None,
        "weight_mode": weight_mode,
        "update_match": update_match,
    }

    # H3 fingerprint emission (EMISSION_CONTRACT § 4). Default OFF: with the flag
    # absent or false NO pool is built (zero extra memory, zero extra parquet
    # work) and no metrics key is emitted, so the client is byte-identical to
    # every sealed H1/H2 run.
    fingerprint_enabled = _as_bool(run_config.get("fingerprint-enabled", False))

    # Check if running in scenario mode (attack controlled per-round by server)
    scenario_mode = run_config.get("scenario", "") != ""

    if scenario_mode:
        # In scenario mode, malicious status is set per-round via FitIns.config
        is_malicious = False
        print(f"[Client {partition_id}] Scenario mode (attack controlled per-round)")
    else:
        # Legacy mode: static malicious assignment at creation time
        malicious_clients = set(get_malicious_clients(dataset, malicious_fraction))
        is_malicious = partition_id in malicious_clients
        if is_malicious:
            print(f"[Client {partition_id}] MALICIOUS ({attack_type})")
        else:
            print(f"[Client {partition_id}] Benign")

    # Override max samples if specified (to match standalone simulation)
    if max_samples > 0:
        task_module.MAX_SAMPLES_PER_CLIENT = max_samples

    # Load data for this partition. Stage-F: request prep_info so the client has
    # n_orig (the update-matching budget input) and the resampling manifest
    # fragment; this changes no loaders, so evaluation/data are unaffected.
    loaded = load_data(
        partition_id=partition_id,
        dataset_name=dataset,
        batch_size=batch_size,
        smote_enabled=smote_enabled,
        smote_variant=smote_variant,
        smote_target=smote_target,
        smote_seed=derive_seed(base_seed, partition_id),
        smote_semantic_target=smote_semantic_target,
        normalize_train_only=normalize_train_only,
        return_prep_info=True,
        return_fingerprint_pool=fingerprint_enabled,
    )
    if fingerprint_enabled:
        train_loader, val_loader, _, prep_info, fingerprint_pool = loaded
        # LOUD, not silent: the pool is None only when the partition fell back to
        # synthetic data (missing/unreadable parquet). Continuing would produce a
        # run with the FP arm nominally on and ZERO fingerprints emitted, which
        # is precisely the gate-(e) failure this arm exists to make impossible.
        if fingerprint_pool is None:
            raise RuntimeError(
                f"[Client {partition_id}] fingerprint-enabled but no fingerprint "
                f"pool could be built for dataset {dataset!r} — the partition fell "
                "back to synthetic data. Refusing to run an FP arm that would emit "
                "no fingerprints (gate (e) requires 100% of client-rounds)."
            )
    else:
        train_loader, val_loader, _, prep_info = loaded
        fingerprint_pool = None
    n_orig = prep_info["n_orig"]
    # Stage-F §4 budget: K = ceil(n_orig / B) * E — exactly the off-arm's per-
    # client optimizer-step count. None when update-match is off (canonical loop).
    max_steps = math.ceil(n_orig / batch_size) * local_epochs if update_match else None

    # Initialize model (SzelagNet for BRFSS, Net for everything else)
    input_shape = detect_input_shape(dataset)
    use_brfss = is_brfss_dataset(dataset)
    net = create_model(dataset, input_shape)

    if use_brfss:
        print(f"[Client {partition_id}] Using SzelagNet (17->8->16->8->1) with BCEWithLogitsLoss")

    # NEW: read optimizer-state mode from run_config (default "reset" for back-compat).
    optimizer_state = str(run_config.get("optimizer-state", "reset"))
    if optimizer_state not in ("reset", "persistent"):
        raise ValueError(f"optimizer-state must be 'reset' or 'persistent', got {optimizer_state!r}")

    # Return NumPyClient instance
    return FlowerClient(
        train_loader,
        val_loader,
        net,
        is_malicious=is_malicious,
        attack_type=attack_type,
        partition_id=partition_id,
        use_brfss=use_brfss,
        lr=lr,
        local_epochs=local_epochs,
        weight_decay=weight_decay,
        optimizer_state=optimizer_state,                # NEW
        node_id=int(context.node_id) if context.node_id is not None else None,  # NEW
        base_seed=base_seed,                            #
        max_steps=max_steps,                            # Stage-F §4
        arm_label=arm_label,                            # Stage-F
        update_match=update_match,                      # Stage-F §4
        weight_mode=weight_mode,                        # Stage-F §5
        n_orig=n_orig,                                  # Stage-F §5
        prep_info=prep_info,                            # Stage-F §5
        semantic_target=smote_semantic_target,          # Stage-F §6
        expected_arm=expected_arm,                      # Stage-F §5 (P2-4)
        fingerprint_pool=fingerprint_pool,              # H3 §4.2 (None = arm off)
    ).to_client()


# Create the ClientApp with the client factory function
app = ClientApp(client_fn=client_fn)
