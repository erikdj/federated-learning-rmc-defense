---
exp_id: {{EXP_ID}}
slug: {{SLUG}}
hypothesis: <FILL: one sentence stating the expected result>
methodology_version: {{METHODOLOGY_VERSION}}
params:
  defense: <FILL: a defense token accepted by the runner>
  scenario: rmc/scenarios/S4_full_mix.json
  seed: 42
  mode: persistent_optimizer
  max_per_client: 2000000
predictions:
  primary_metric: <FILL: metric name and expected range or direction>
---

# {{EXP_ID}} — {{SLUG}}

**Date:** {{DATE}}
**Methodology:** {{METHODOLOGY_VERSION}}

## Hypothesis

<FILL: state the falsifiable hypothesis and the population it covers>

## Design

<FILL: identify the scenario, defense, seed allocation, and controlled variables>

## Success criteria

<FILL: state the decision rule before launch>

## Risks and limitations

<FILL: record known threats to validity and interpretation boundaries>

## Reproduction references

- Experiment definitions and result boundaries: `docs/reproduction/experiments.md`
- Harness setup and recovery: `docs/harness/README.md`
- Software methodology record: `docs/METHODOLOGY_LOG.md`
