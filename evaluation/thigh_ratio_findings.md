# Frontal sitting cue

Added a supplementary sitting cue in `elderly_monitor/vision/pose_observer.py`.
Both confident legs must have vertical thigh/shin ratio below 0.65, sufficiently
vertical shins, and knee angles below 170 degrees. Shoulder width must exceed
0.45 torso lengths and hips must project inside the bed/mattress region.
The geometry must persist for 0.5 seconds with continuous identity/visibility;
walking evidence blocks the cue and hip motion must stay below 0.2 torso
lengths/second. These are provisional camera-dependent thresholds, not learned
anatomical constants. Whole-bed overlap does not prove physical mattress support.

## Saved-pose comparison

Run `python -m evaluation.thigh_ratio_replay` from the project root.
Outputs go to `output/evaluation/thigh_ratio`; original final outputs and human
labels are preserved. Both arms use identical pre-Gemini observations and saved
temporal settings, reclassifying poses chronologically with the cue off/on.
No new detector inference or Gemini calls are made. This isolates the local cue;
new review-window selection and full Gemini-assisted accuracy remain unmeasured.

Across ten development clips (160 seconds), local pose + temporal accuracy went
from 61.07% to 65.16%. Granny improved from 38.55% to 60.36%; all other nine clips
were unchanged. At granny 10.667 seconds the local classification now reads
SITTING_ON_BED, reason persistent_foreshortened_thighs. Granny's event matching
did not improve. The earlier final Gemini-assisted score is a different experiment
and must not be compared directly with these local-only scores.

106 unit tests passed, including short-thigh standing, persistent sitting,
identity/visibility interruptions and moving poses. Broader validation on unseen
front-facing sitting/standing footage is still needed.
