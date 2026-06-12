# Stage 10.2 — Seed Support Validation

Stage 10.2 fixes a failure seen in Google Earth: the first accepted anchors can be wrong even though LightGlue/RANSAC accepts them.  Instead of manually deleting bad frames, this patch makes the planner reject small unsupported seed islands automatically.

## New idea

A first-pass accepted match is only allowed to become a seed anchor if it belongs to a locally supported temporal/spatial component.

The planner now runs:

1. normal accepted-anchor quality filter,
2. dominant connected-component filter,
3. **seed support validation** using a stricter support graph,
4. isolated teleport filter,
5. expected-path interpolation and local-rerun window creation.

This means a small early false cluster, for example frames 540/1170/1260, can be rejected automatically if the main supported path is elsewhere.

## New planner options

```text
--use-seed-support-validation / --no-seed-support-validation
--seed-support-radius-m
--seed-support-max-speed-mps
--seed-support-order-window
--seed-min-support
--seed-support-min-component-size
```

Recommended starting values for DJI_0011 every-45:

```text
--seed-support-radius-m 140
--seed-support-max-speed-mps 30
--seed-support-order-window 8
--seed-min-support 1
--seed-support-min-component-size 4
```

## Outputs

`stage10_seed_path.csv` and `stage10_path_rerun_plan.csv` now include:

```text
seed_support_component_id
seed_support_component_size
seed_support_count
seed_support_status
```

Use these columns to understand why a seed was kept or rejected.
