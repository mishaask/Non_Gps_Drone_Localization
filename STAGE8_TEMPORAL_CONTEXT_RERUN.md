# Stage 8 — Temporal Context Rerun Planner

Stage 8 corrects the Stage 7 rerun strategy.

Instead of rerunning a frame against the region it weakly guessed, Stage 8:

1. Uses only high-confidence accepted frames as anchors.
2. Clusters anchors into stable GPS regions.
3. Finds weak / rejected / sudden-jump frames.
4. Assigns each bad frame to candidate regions from nearby stable anchors and/or the globally most common regions.
5. Writes per-region frame lists and local reference indexes for reruns.
6. Merges only accepted context-approved rerun results that improve confidence.

This is still debug-friendly and intentionally conservative.
