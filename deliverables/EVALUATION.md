# Saved-run evaluation snapshot

Recomputed from saved final prediction JSON files and the current reviewed labels. This is not a fresh run of the latest source code. See [submission notes](README.md) for code/evidence differences.

10 clips; 160.0 seconds. Duration-weighted activity accuracy: **80.00%**. Unweighted mean clip accuracy: **77.83%**.

Accuracy is matching labelled time divided by total time. UNKNOWN predictions count as disagreement against known labels. Event matching uses a 1-second tolerance. Labels marked reviewed are supplied by the project; this export does not independently verify them.

> **Video versus final results:** Annotated MP4 files show first-pass candidate states. They do not include later dense-context reclassification, accepted Gemini corrections or final temporal smoothing. Reported accuracy, timelines and events use final reviewed JSON. A second rendering pass from that timeline is still needed for a final-reviewed demonstration video.

| Clip | Accuracy | Predicted UNKNOWN |
|---|---:|---:|
| chair_sitting | 59.00% | 28.75% |
| granny | 68.55% | 2.22% |
| japan_cctv | 70.78% | 27.92% |
| night_time | 74.00% | 3.33% |
| night_view | 90.92% | 2.50% |
| sleep_sit | 95.94% | 2.29% |
| sleeping_turn_aruond | 100.00% | 0.00% |
| standing_bed | 87.50% | 1.25% |
| UV_camera | 54.92% | 22.50% |
| walking | 76.72% | 14.37% |

## Bed events

| Event | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| bed_exit | 1 | 3 | 4 | 25.0% | 20.0% |
| return_to_bed | 1 | 3 | 2 | 25.0% | 33.3% |

Unmatched predictions can reflect incorrect events or timing outside tolerance. Activity accuracy does not imply reliable event detection.

## Duration errors

Absolute error is summed per clip, so overestimates and underestimates cannot cancel.

| State | Labelled seconds | Predicted seconds | Sum of absolute errors (s) |
|---|---:|---:|---:|
| LYING_IN_BED | 54.564 | 49.125 | 5.439 |
| SITTING_ON_BED | 52.249 | 47.667 | 9.408 |
| SITTING_OUTSIDE_BED | 6.182 | 6.250 | 1.338 |
| STANDING | 15.896 | 18.417 | 10.413 |
| WALKING | 30.079 | 24.958 | 11.373 |
| OUT_OF_BED | 0.000 | 0.500 | 0.500 |
| UNKNOWN | 1.030 | 13.083 | 12.803 |

Full per-clip confusion matrices, duration errors and event metrics: [evaluation_metrics.json](evaluation_metrics.json).

Saved Gemini request statuses: `{'reviewed': 42}`. Successful review is not a correctness guarantee.

These short development clips have been used for tuning, include generated footage, and are not an independent real-world validation set. Long-duration alert behaviour is unit-tested but not demonstrated by these short videos.
