# Failure cases

These compare saved final predictions with project labels. Proposed mechanisms should
not be confused with independent verification of every underlying video frame.

## 1. Sitting classified as standing

`granny.mp4`, **2.500-5.667 seconds**: reference SITTING_ON_BED, prediction STANDING.
[Prediction](examples/granny.json), [labels](../evaluation/labels/granny.json).
Front-facing bent legs can appear straight in 2D, defeating a knee-angle rule. Sparse
review leaves stable errors uncorrected. At the separately examined 10.667-second frame,
mean knee angle was 153.5 degrees versus the 145-degree sitting cutoff; Gemini corrected
that later moment. Persistent thigh foreshortening and an upright-over-bed review trigger
address parts of this problem, but review is still budget-limited.

## 2. Short walking classified as standing

`granny.mp4`, **17.625-19.125 seconds**: reference WALKING, prediction STANDING.
[Prediction](examples/granny.json), [labels](../evaluation/labels/granny.json).
Recorded hip/ankle motion fell below thresholds; filtering can discard small movements.
Gemini's separated images also described standing. Accumulating small motion reduced
local accuracy and was reverted. This remains unresolved; consecutive-frame inspection
is needed before changing walking thresholds.

## 3. Early activity lost to UNKNOWN

`chair_sitting.mp4`, early approximately **0-3 seconds**.
[Prediction](examples/chair_sitting.json), [labels](../evaluation/labels/chair_sitting.json).
The clip has 2.875 seconds UNKNOWN (28.75%) and 59.00% overall accuracy. Labelled
standing/walking is lost to uncertainty. UNKNOWN avoids unsupported classifications
but still reduces useful coverage and is not correct activity recognition.
Improve initial pose/tracking continuity while preserving uncertainty when evidence is
missing. Current labels contain no bed-exit event for this clip.

## Additional event limitation

UV_camera has an unmatched predicted return and missed labelled return under the
one-second matching tolerance. Inspect event timing before assuming two physical events.
Overall exit recall is only 20%, despite 80% activity accuracy.
