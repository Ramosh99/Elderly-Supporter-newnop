"""Explain the recording-level monitoring decision."""


def decide_alert(has_unknown, longest_sitting, longest_away, events, sitting_monitor_sec, alert_after_sec):
    reasons = []
    decision = "NORMAL"
    if has_unknown:
        reasons.append("uncertain_or_missing_observations")
    if longest_sitting >= sitting_monitor_sec:
        reasons.append("prolonged_sitting_on_bed")
    if events and events[-1].event == "bed_exit":
        reasons.append("bed_exit_without_confirmed_return")
    if reasons:
        decision = "MONITOR"
    if longest_away >= alert_after_sec:
        decision = "ALERT"
        reasons.append("prolonged_confirmed_absence_from_bed")
    return decision, reasons
