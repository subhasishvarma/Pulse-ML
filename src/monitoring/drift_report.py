"""
Runs Evidently's DataDriftPreset + DataQualityPreset comparing the reference
(early) period against the current (late) period, and saves an HTML report.
"""
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

from src.monitoring.generate_reference import get_reference_and_current

REPORT_DIR = Path("reports/drift_reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def generate_drift_report(version: str = "v1") -> Path:
    reference_df, current_df = get_reference_and_current()

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    out_path = REPORT_DIR / f"drift_report_{version}.html"
    report.save_html(str(out_path))
    print(f"Drift report saved to {out_path}")
    return out_path


if __name__ == "__main__":
    generate_drift_report()
