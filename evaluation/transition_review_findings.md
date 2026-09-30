# Review coverage and transition corrections

Stable STANDING/WALKING observations projecting inside an uncalibrated whole-bed
region now trigger `ambiguous_upright_over_bed` review. This closes a trigger blind
spot, not a guarantee of full coverage: review/frame/request budgets still apply.
Calibrated floor support and stable lying do not trigger this extra review.
Newly selected windows need a fresh pipeline run to obtain Gemini assessments.

The fragmentation guard now permits up to state_hold_sec additional UNKNOWN time
when all changed samples have identified, visible targets, carry bracketed Gemini
posture evidence, survive smoothing as the proposed state, and span at least
posture_hold_sec. It does not force short residual transitions into a known state.
Single-sample corrections and unidentified corrections cannot use this exception.

`python -m evaluation.transition_review_replay` compares strict and revised guards
on identical saved pre-Gemini observations and responses. Outputs are separate in
output/evaluation/transition_review. Granny accuracy improves from 68.55% to 69.92%;
the other nine scored clips are unchanged. This isolates the guard and does not
measure the changed review selection or a fresh Gemini run. Labels are unchanged.

A motion experiment accumulated sub-threshold displacement rather than dropping
each small movement. It reduced granny local pose + tracking accuracy from 60.36%
to 58.69%, so it was reverted. Before/experimental comparisons are preserved under
output/evaluation/thigh_ratio. Short-walk detection remains unresolved; no walking
thresholds were lowered. Sparse Gemini images also labelled this interval standing.

109 tests pass, including review budgets, stable upright review, bounded transition
uncertainty, and rejection of isolated or unidentified corrections.
