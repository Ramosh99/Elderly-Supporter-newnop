# Assignment deliverables

Prepared against section 11 of `ASE_AIML_Assignment.pdf`. A CLI is sufficient; no frontend
is required. This is a working prototype with documented event-detection limitations.

| Requirement | Deliverable |
|---|---|
| Source and run instructions | [Project README](../README.md), `../elderly_monitor/` |
| Architecture diagram | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Activity timelines | `examples/*.json`: `timeline` |
| Activity durations | `examples/*.json`: `activity_duration_sec`, numeric seconds |
| Bed-exit/return events | `examples/*.json`: `events` and event counts |
| Evaluation and duration errors | [EVALUATION.md](EVALUATION.md), [metrics](evaluation_metrics.json) |
| Three failure examples | [FAILURE_CASES.md](FAILURE_CASES.md) |

## Evidence provenance

The latest batch report was written September 30, 2026 at 12:47 local time. This snapshot
recomputes metrics from saved per-video predictions and current reviewed labels.
[SHA256 hashes](evidence_manifest.json) identify the inputs. Export makes no API calls
and changes neither source predictions nor labels. The saved run has no source-commit
identifier; timestamps alone cannot establish exact correspondence to this working tree.

Source includes the frontal-thigh cue and revised transition guard/review trigger.
Their isolated replays are separate experiments: granny's 69.92% guard replay is **not**
the saved full-run 68.55%. Do not combine gains or substitute replay scores for full runs.
109 unit tests validate code behaviour, not unseen-video accuracy or clinical safety.
The ten short development clips include generated footage and were used for tuning.
Labels are marked reviewed; packaging does not independently certify their correctness.

## Demonstration

1. Explain the architecture; show `examples/sleeping_turn_aruond.json` for stable lying.
2. Show `examples/standing_bed.json` for a matched exit and its confirmation delay.
3. Show granny's final sitting at 10.7 seconds alongside the candidate video; explain
   why the first-pass overlay can still say standing.
4. Present failure cases, duration errors and event precision/recall, not only accuracy.
5. Explain review budgets, correction gates, UNKNOWN and alert rules.

Candidate videos remain in `../output/evaluation/final/` locally. Share permitted media
separately; examples here omit raw keypoints and local absolute video paths.

## Remaining limitations

Short walks, projected bed overlap, event timing and false events remain imperfect.
A final-reviewed overlay would improve presentation; current overlays show candidates.
Held-out real recordings and longer sequences are needed to validate generalisation
and prolonged alerts. Further model/threshold tuning is deferred for this wrap-up.

Submit source, requirements, example config, labels and this folder. Exclude `.env`,
API keys, `.venv`, caches and private media. Check media sharing rights. The assignment
requests a repository link; no email, publication or upload was performed here.
