from __future__ import annotations

from baymax.models.sft_artifact_report import (
    SFTArtifactReport,
    SFTBehaviorReportRow,
    SFTDuplicateByBehaviorRow,
    SFTDuplicateReport,
    SFTSourceReportRow,
)
from scripts.report_sft_artifact import format_artifact_report


def test_format_artifact_report_includes_behavior_source_and_duplicates() -> None:
    report = SFTArtifactReport(
        row_count=2,
        behavior=[
            SFTBehaviorReportRow(
                behavior_class="direct_tool_call",
                rows=1,
                row_percent=0.5,
                trainable_tokens=10,
                token_percent=0.25,
            ),
            SFTBehaviorReportRow(
                behavior_class="refuse_request",
                rows=1,
                row_percent=0.5,
                trainable_tokens=30,
                token_percent=0.75,
            ),
        ],
        source=[
            SFTSourceReportRow(source="xlam", rows=1, row_percent=0.5),
            SFTSourceReportRow(source="baymax", rows=1, row_percent=0.5),
        ],
        duplicates=SFTDuplicateReport(exact_duplicate_rows=0, unique_rows=2),
        duplicates_by_behavior=[
            SFTDuplicateByBehaviorRow(
                behavior_class="direct_tool_call",
                exact_duplicate_rows=0,
                unique_rows=1,
            ),
            SFTDuplicateByBehaviorRow(
                behavior_class="refuse_request",
                exact_duplicate_rows=0,
                unique_rows=1,
            ),
        ],
    )

    formatted = format_artifact_report(report)

    assert "direct_tool_call" in formatted
    assert "refuse_request" in formatted
    assert "xlam" in formatted
    assert "baymax" in formatted
    assert "exact duplicate rows: 0" in formatted
    assert "duplicates by class" in formatted
