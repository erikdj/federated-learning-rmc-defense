"""Steps-driven update matching for Stage-F (DESIGN_STAGE_F §4).

Stage F holds the per-client optimizer-update count fixed across every arm and
every training path, so only minibatch class composition varies. The primitive
is a single generic loop that runs EXACTLY ``max_steps`` optimizer steps by
re-entering the loader whenever it is exhausted: an under-sampled loader cycles
as many times as needed to reach the cap, an over-sampled loader stops mid-pass.

This lives in its own module (rather than inlined four times in flowerfl/task.py)
so the exact-cap semantics — break at ``steps_taken == max_steps`` (never
overshoot), reach the cap by CYCLING not by adding epochs, and a LOUD empty-
loader ``RuntimeError`` — are defined and tested once. Each ``train*`` function
supplies its own per-batch closure (honest vs. label-flipped, CE vs. BCE) so the
minibatch body stays byte-identical to its epochs-bounded original.

The budget itself is ``K = ceil(n_orig / batch_size) * local_epochs`` — exactly
the off-arm's per-client update count over the pre-resampling row count — and is
computed at the client (flowerfl/client_app.py); this module only executes it.
"""
from __future__ import annotations

from typing import Callable


def run_matched_steps(
    loader,
    max_steps: int,
    *,
    step_fn: Callable,
    partition_id: "int | None" = None,
    arm: "str | None" = None,
) -> tuple[float, int]:
    """Run exactly ``max_steps`` optimizer steps over ``loader``, cycling it.

    ``step_fn(batch)`` performs one optimizer step (zero_grad / forward / loss /
    backward / step) and returns that step's scalar loss. The loader is re-entered
    each time it is exhausted; the loop breaks the instant ``steps_taken ==
    max_steps`` (exact — never overshoots). Returns ``(total_loss, steps_taken)``.

    LOUD empty-loader behaviour (§4): a client whose loader is empty while
    ``max_steps > 0`` can never reach the cap, so it raises ``RuntimeError``
    naming the client and arm rather than silently returning under-updated.
    """
    if max_steps > 0 and len(loader) == 0:
        raise RuntimeError(
            f"empty post-resampling loader for client {partition_id} "
            f"arm {arm} with max_steps={max_steps}: cannot reach the "
            f"update cap — never silently return"
        )

    total_loss = 0.0
    steps_taken = 0
    while steps_taken < max_steps:
        for batch in loader:  # re-enters the loader each time it is exhausted
            total_loss += step_fn(batch)
            steps_taken += 1
            if steps_taken == max_steps:
                break
    return total_loss, steps_taken
