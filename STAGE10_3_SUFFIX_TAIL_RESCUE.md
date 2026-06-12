# Stage 10.3 — suffix / landing-tail rescue

Stage 10.2 fixed unsupported false prefix seed anchors, but it could over-filter the end of the flight.
The landing/end tail often has only one or two good accepted anchors, so a symmetric support filter may reject it
because there are no future anchors to support it.

Stage 10.3 adds an asymmetric suffix rescue:

- unsupported prefix islands remain rejected;
- the main supported seed path is kept;
- a sparse suffix after the main path can be rescued if it is strong and physically plausible from the last trusted seed.

New planner options:

```bat
--use-seed-tail-rescue
--no-seed-tail-rescue
--seed-tail-rescue-min-points 1
--seed-tail-rescue-min-confidence 8.0
--seed-tail-rescue-max-distance-m 420
--seed-tail-rescue-max-speed-mps 35
--seed-tail-rescue-max-gap-s 120
```

Recommended DJI_0011 command is included in the ChatGPT response.
