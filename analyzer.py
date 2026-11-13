"""OpenAI-backed analysis workflow for time-and-motion study reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeVar

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_STUDY_PATH = Path(__file__).resolve().parent / "data" / "time_motion_study.md"
DEFAULT_MODEL = "gpt-5.4-mini"


class OutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceRef(OutputModel):
    quote: str = Field(..., min_length=1)
    section: str | None = None
    day_index: int | None = Field(default=None, ge=1)


class Activity(OutputModel):
    activity_id: str = Field(..., min_length=1)
    day_index: int | None = Field(default=None, ge=1)
    worker: str
    role: str = Field(..., min_length=1)
    task: str
    location: str | None = None
    start_raw: str | None = None
    end_raw: str | None = None
    start_minute_of_shift: int | None = Field(default=None, ge=0)
    end_minute_of_shift: int | None = Field(default=None, ge=0)
    duration_minutes: float | None = Field(default=None, ge=0)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    source_section: str | None = None
    confidence: float = Field(default=0.7, ge=0, le=1)

    @model_validator(mode="after")
    def _validate_activity_timing(self) -> "Activity":
        if self.start_minute_of_shift is not None and self.end_minute_of_shift is not None:
            if self.end_minute_of_shift < self.start_minute_of_shift:
                # Preserve raw times but clear invalid normalized values.
                self.start_minute_of_shift = None
                self.end_minute_of_shift = None
                return self

            if self.duration_minutes is not None:
                expected_duration = self.end_minute_of_shift - self.start_minute_of_shift
                # If parsed range and duration diverge materially, trust explicit
                # time bounds and normalize duration to keep the pipeline stable.
                if abs(self.duration_minutes - expected_duration) > 10:
                    self.duration_minutes = float(expected_duration)
        return self


class WorkerClassification(OutputModel):
    worker: str
    role: str = Field(..., min_length=1)
    station: str | None = None
    activity_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0, le=1)


class TimingAnalysis(OutputModel):
    timing_check_id: str = Field(..., min_length=1)
    activity_ids: list[str] = Field(default_factory=list)
    worker: str | None = None
    task: str
    status: Literal["consistent", "inconsistent", "uncertain"]
    stated_duration_minutes: float | None = Field(default=None, ge=0)
    calculated_duration_minutes: float | None = Field(default=None, ge=0)
    details: str
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_timing_status(self) -> "TimingAnalysis":
        if self.stated_duration_minutes is None or self.calculated_duration_minutes is None:
            return self

        delta = abs(self.stated_duration_minutes - self.calculated_duration_minutes)
        if self.status == "consistent" and delta > 5:
            self.status = "inconsistent"
        if self.status == "inconsistent" and delta < 1:
            self.status = "uncertain"
        return self


class EfficiencyFlag(OutputModel):
    flag_id: str = Field(..., min_length=1)
    issue_type: str = Field(..., min_length=1)
    severity: Literal["low", "medium", "high"]
    description: str
    workers: list[str] = Field(default_factory=list)
    activity_ids: list[str] = Field(default_factory=list)
    time_window_raw: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    likely_cause: str | None = None
    recommendation: str | None = None


class AnalysisResult(OutputModel):
    activities: list[Activity] = Field(default_factory=list)
    worker_classifications: list[WorkerClassification] = Field(default_factory=list)
    timing_checks: list[TimingAnalysis] = Field(default_factory=list)
    efficiency_flags: list[EfficiencyFlag] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)


class DeepReviewPatch(OutputModel):
    activities: list[Activity] = Field(default_factory=list)
    worker_classifications: list[WorkerClassification] = Field(default_factory=list)
    timing_checks: list[TimingAnalysis] = Field(default_factory=list)
    efficiency_flags: list[EfficiencyFlag] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    validation_notes: list[str] = Field(default_factory=list)


class AnalysisState(TypedDict, total=False):
    study_path: str
    source_text: str
    analysis: dict[str, Any]


ANALYSIS_INSTRUCTIONS = """You are an industrial engineering analyst. Analyze the provided time-and-motion study and return the complete structured analysis.

Schema and data rules:
- Return all fields required by the schema.
- Create stable IDs for each activity, timing check, and efficiency flag.
- Create concise role labels from the observed duties. Do not use a fixed role taxonomy; invent a precise label when the evidence supports a role not seen elsewhere.
- Create concise issue_type labels from the observed inefficiency. Reuse the exact same label for the same issue concept within the result, but do not use a fixed issue taxonomy.
- Preserve original time phrases in start_raw/end_raw/time_window_raw and also provide normalized minute fields when possible.
- For evidence, provide structured evidence objects with quote, section, and day_index when available.
- Link timing checks and efficiency flags to activities with activity_ids where applicable.

Analysis requirements:
- Extract discrete activities: what, who, where, and for how long.
- Classify workers from observed duties.
- Normalize timings and flag inconsistencies between stated and calculated durations.
- Detect inefficiencies: idle/waiting, unnecessary movement, bottlenecks, duplicated activity, handoff/process issues.
- Include cross-day comparisons and practical recommendations.

Use only source evidence. Do not invent workers, tasks, times, causes, or outcomes. Distinguish observed facts from likely causes."""

REVIEW_INSTRUCTIONS = """Review the draft analysis against the original study text and return a corrected complete schema-compliant result.

Validation priorities:
- Activities, worker roles, and links (activity_ids) are evidence-backed.
- Normalized timing fields are correct when derivable and preserved as uncertainty when not.
- Timing inconsistencies are explicitly flagged, not silently reconciled.
- Efficiency flags and likely causes are grounded and not overstated.
- Required fields and string labels match the schema exactly.

Include concise validation_notes for material corrections or explicit confirmation of review checks."""

DEEP_REVIEW_INSTRUCTIONS = """Review the draft analysis against the original study text and return only targeted patch updates.

You are performing focused deep review for one inefficiency issue cluster at a time.
The draft may include a __deep_review_focus object with:
- issue_type
- flag_ids
- cluster_size

Deep-review requirements:
- Re-check all efficiency flags in this cluster against source evidence.
- Tighten issue descriptions, likely causes, and recommendations for this cluster.
- Verify linked activity_ids and workers are grounded in text.
- Return only changed records in the patch fields. Leave unrelated records out.

Validation priorities:
- Activities, worker roles, and links (activity_ids) are evidence-backed.
- Normalized timing fields are correct when derivable and preserved as uncertainty when not.
- Timing inconsistencies are explicitly flagged, not silently reconciled.
- Efficiency flags and likely causes are grounded and not overstated.
- Required fields and string labels match the schema exactly.

Include concise validation_notes for material corrections or explicit confirmation of review checks."""


ModelT = TypeVar("ModelT", bound=BaseModel)


def _enforce_strict_additional_properties(schema: Any) -> Any:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            # OpenAI strict json_schema requires object nodes to explicitly
            # disallow unspecified keys.
            schema["additionalProperties"] = False
            properties = schema.get("properties")
            if isinstance(properties, dict):
                # OpenAI strict json_schema also requires every property key
                # to be listed in the required array. Optional fields are
                # represented as nullable types in the property schema.
                schema["required"] = list(properties.keys())
        for value in schema.values():
            _enforce_strict_additional_properties(value)
    elif isinstance(schema, list):
        for item in schema:
            _enforce_strict_additional_properties(item)
    return schema


def _strict_schema_for(model_type: type[ModelT]) -> dict[str, Any]:
    return _enforce_strict_additional_properties(model_type.model_json_schema())


def _progress_enabled() -> bool:
    raw_value = os.getenv("ANALYZER_PROGRESS", "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def _log_progress(message: str) -> None:
    if _progress_enabled():
        print(f"[analyzer] {message}", flush=True)


def _generate_analysis(source_text: str, review_of: dict[str, Any] | None = None) -> AnalysisResult:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set it in the environment before running the analyzer.")

    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    instructions = ANALYSIS_INSTRUCTIONS
    user_content = f"SOURCE STUDY:\n{source_text}"
    if review_of is not None:
        instructions = REVIEW_INSTRUCTIONS
        user_content = (
            f"SOURCE STUDY:\n{source_text}\n\n"
            "DRAFT ANALYSIS TO REVIEW:\n"
            f"{json.dumps(review_of, ensure_ascii=False)}"
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "analysis_result",
                    "strict": True,
                    "schema": _strict_schema_for(AnalysisResult),
                },
            }
        )

    response_text = response.choices[0].message.content if response.choices else None
    if not response_text:
        raise RuntimeError("OpenAI returned no structured analysis.")
    return AnalysisResult.model_validate_json(response_text)


def _cluster_efficiency_flags(analysis: AnalysisResult) -> list[dict[str, Any]]:
    clusters: dict[str, list[EfficiencyFlag]] = {}
    for flag in analysis.efficiency_flags:
        clusters.setdefault(flag.issue_type, []).append(flag)

    cluster_items: list[dict[str, Any]] = []
    for issue_type, flags in clusters.items():
        cluster_items.append(
            {
                "issue_type": issue_type,
                "flag_ids": [flag.flag_id for flag in flags],
                "cluster_size": len(flags),
            }
        )
    return cluster_items


def _build_deep_review_payload(review_of: dict[str, Any], cluster: dict[str, Any]) -> dict[str, Any]:
    analysis = AnalysisResult.model_validate(review_of)
    target_flag_ids = set(cluster.get("flag_ids", []))

    selected_flags = [flag for flag in analysis.efficiency_flags if flag.flag_id in target_flag_ids]
    related_activity_ids = {activity_id for flag in selected_flags for activity_id in flag.activity_ids}
    related_workers = {worker for flag in selected_flags for worker in flag.workers}

    selected_activities = [
        activity
        for activity in analysis.activities
        if activity.activity_id in related_activity_ids or activity.worker in related_workers
    ]
    selected_activity_ids = {activity.activity_id for activity in selected_activities}

    selected_worker_classifications = [
        classification
        for classification in analysis.worker_classifications
        if classification.worker in related_workers
        or any(activity_id in selected_activity_ids for activity_id in classification.activity_ids)
    ]

    selected_timing_checks = [
        check
        for check in analysis.timing_checks
        if check.worker in related_workers
        or any(activity_id in selected_activity_ids for activity_id in check.activity_ids)
    ]

    payload = {
        "activities": [activity.model_dump(mode="json") for activity in selected_activities],
        "worker_classifications": [
            classification.model_dump(mode="json") for classification in selected_worker_classifications
        ],
        "timing_checks": [check.model_dump(mode="json") for check in selected_timing_checks],
        "efficiency_flags": [flag.model_dump(mode="json") for flag in selected_flags],
        "recommendations": analysis.recommendations,
        "validation_notes": analysis.validation_notes[-10:],
        "__deep_review_focus": cluster,
    }
    return payload


def _merge_deep_review_patch(review_of: dict[str, Any], patch: DeepReviewPatch, cluster: dict[str, Any]) -> dict[str, Any]:
    analysis = AnalysisResult.model_validate(review_of)
    target_flag_ids = set(cluster.get("flag_ids", []))

    activities_by_id = {activity.activity_id: activity for activity in analysis.activities}
    for activity in patch.activities:
        activities_by_id[activity.activity_id] = activity

    worker_key = lambda c: (c.worker, c.role)
    classifications_by_key = {worker_key(c): c for c in analysis.worker_classifications}
    for classification in patch.worker_classifications:
        classifications_by_key[worker_key(classification)] = classification

    timing_by_id = {check.timing_check_id: check for check in analysis.timing_checks}
    for check in patch.timing_checks:
        timing_by_id[check.timing_check_id] = check

    flags_by_id = {flag.flag_id: flag for flag in analysis.efficiency_flags}
    for flag in patch.efficiency_flags:
        if not target_flag_ids or flag.flag_id in target_flag_ids:
            flags_by_id[flag.flag_id] = flag

    merged_recommendations = list(analysis.recommendations)
    for recommendation in patch.recommendations:
        if recommendation not in merged_recommendations:
            merged_recommendations.append(recommendation)

    merged_validation_notes = list(analysis.validation_notes)
    for note in patch.validation_notes:
        if note not in merged_validation_notes:
            merged_validation_notes.append(note)

    merged = AnalysisResult(
        activities=list(activities_by_id.values()),
        worker_classifications=list(classifications_by_key.values()),
        timing_checks=list(timing_by_id.values()),
        efficiency_flags=list(flags_by_id.values()),
        recommendations=merged_recommendations,
        validation_notes=merged_validation_notes,
    )
    return merged.model_dump(mode="json")


def _generate_deep_review_for_cluster(
    source_text: str,
    review_of: dict[str, Any],
    cluster: dict[str, Any],
) -> DeepReviewPatch:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set it in the environment before running the analyzer.")

    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    review_payload = _build_deep_review_payload(review_of=review_of, cluster=cluster)
    user_content = (
        f"SOURCE STUDY:\n{source_text}\n\n"
        "DRAFT ANALYSIS TO REVIEW:\n"
        f"{json.dumps(review_payload, ensure_ascii=False)}"
    )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DEEP_REVIEW_INSTRUCTIONS},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "deep_review_patch",
                "strict": True,
                "schema": _strict_schema_for(DeepReviewPatch),
            },
        },
    )

    response_text = response.choices[0].message.content if response.choices else None
    if not response_text:
        raise RuntimeError("OpenAI returned no structured analysis.")
    return DeepReviewPatch.model_validate_json(response_text)


def _read_study(state: AnalysisState) -> dict[str, Any]:
    study_path = Path(state["study_path"])
    return {"source_text": study_path.read_text(encoding="utf-8")}


def _analyze(state: AnalysisState) -> dict[str, Any]:
    _log_progress("Running initial analysis pass.")
    analysis = _generate_analysis(state["source_text"])
    return {"analysis": analysis.model_dump(mode="json")}


def _needs_review(state: AnalysisState) -> str:
    analysis = AnalysisResult.model_validate(state["analysis"])
    has_timing_issue = any(check.status != "consistent" for check in analysis.timing_checks)
    has_efficiency_flags = bool(analysis.efficiency_flags)
    if has_efficiency_flags:
        return "deep_review"
    if has_timing_issue:
        return "review"
    return "done"


def _review(state: AnalysisState) -> dict[str, Any]:
    _log_progress("Running general review pass.")
    analysis = _generate_analysis(state["source_text"], review_of=state["analysis"])
    return {"analysis": analysis.model_dump(mode="json")}


def _deep_review(state: AnalysisState) -> dict[str, Any]:
    current = AnalysisResult.model_validate(state["analysis"])
    clusters = _cluster_efficiency_flags(current)
    _log_progress(f"Running deep review for {len(clusters)} issue cluster(s).")

    analysis_dict = current.model_dump(mode="json")
    for index, cluster in enumerate(clusters, start=1):
        issue_type = str(cluster.get("issue_type", "unknown"))
        cluster_size = cluster.get("cluster_size", "?")
        _log_progress(
            f"Deep review cluster {index}/{len(clusters)}: {issue_type} ({cluster_size} flag(s))."
        )
        patch = _generate_deep_review_for_cluster(
            source_text=state["source_text"],
            review_of=analysis_dict,
            cluster=cluster,
        )
        analysis_dict = _merge_deep_review_patch(
            review_of=analysis_dict,
            patch=patch,
            cluster=cluster,
        )
    _log_progress("Deep review complete.")

    return {"analysis": analysis_dict}


def build_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("read_study", _read_study)
    graph.add_node("analyze", _analyze)
    graph.add_node("review", _review)
    graph.add_node("deep_review", _deep_review)
    graph.set_entry_point("read_study")
    graph.add_edge("read_study", "analyze")
    graph.add_conditional_edges(
        "analyze",
        _needs_review,
        {"deep_review": "deep_review", "review": "review", "done": END},
    )
    graph.add_edge("review", END)
    graph.add_edge("deep_review", END)
    return graph.compile()


def analyze_study(study_path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    path = Path(study_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Study document not found: {path}")

    _log_progress(f"Starting analysis for {path.name}.")
    result = build_graph().invoke({"study_path": str(path)})
    _log_progress("Analysis workflow complete.")
    return AnalysisResult.model_validate(result["analysis"]).model_dump(mode="json")
