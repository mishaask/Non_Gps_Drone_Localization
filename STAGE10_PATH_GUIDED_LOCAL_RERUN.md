# Stage 10 — Path-guided local rerun

Stage 10 uses the first global run only to build a physically plausible seed path.
It then reruns bad/uncertain frames around the expected path location, not across
the whole map and not across broad guessed regions.

Core idea:

1. Global first pass with low top-k.
2. Keep accepted first-pass matches only if they do not imply impossible jumps.
3. Interpolate/hold a seed path through the trusted anchors.
4. Build small local reference indexes around the expected path.
5. Rerun non-seed frames with higher top-k inside those local indexes.
6. Merge only accepted local reruns that stay close to the expected path.
7. Export accepted visual matches and the full filtered/interpolated path.

For now, this pipeline intentionally disables YOLO masking by default in the
recommended commands, because the current errors are mostly repeated campus
structures rather than cars.
