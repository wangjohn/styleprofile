"""The shapes of the JSON reports styleprofile writes, as TypedDicts.

There are three kinds of report (``profile.KINDS``): a reference profile
(``ReferenceReport``, from ``styleprofile build``), a score report (``ScoreReport``, drafts
scored against a reference) and an evaluation report (``EvaluationReport``, from
``styleprofile evaluate``). The code that writes and reads them is typed with these, so
pyright catches a misspelled or missing key, and ``load_report`` refuses a saved report that
lacks a required key, or holds the wrong kind of container or a null where one is needed
(``find_problem``), instead of failing later with a ``KeyError`` or ``TypeError``.

A key is ``NotRequired`` only when a report can genuinely lack it: a reference drawn from one
document has no ``calibration``, one built without contrast drafts has no ``contrast``, and
settings made by the lower-level functions in ``styleprofile.profile`` record only what they
were given. Everything else is always written.

Adding a key to a report means adding it here, in the TypedDict for the part it belongs to,
and nowhere else; each part has one home:

- reference calibration (``calibration.by_length``, say): ``Calibration``, with a
  TypedDict of its own for the entries (a key only some entries have, such as ``delta``, is
  ``NotRequired`` there);
- a new top-level key of a reference: ``ReferenceReport``, and of a score report
  (``documents``): ``ScoreReport``; keys both share (``document_count``) go in
  ``ReportBase``;
- a score report's figures over all its chunks (``reference.verdict``, say):
  ``ReferenceScore``; each chunk's own (``chunks[i].reference.calibration``): ``ChunkScore``;
- how an AUC's interval was found (``bootstrap.method``): ``Bootstrap``, shared by the
  contrast calibration and every AUC of an evaluation (``AucResult``);
- a recorded ``api.Settings`` field: ``InputSettings``;
- what a score report copies from its reference for rendering: ``Baseline``. When it copies
  a part in another shape than the reference's (``BaselineCalibration``, a calibration
  without each length's per-metric ``reliability``), that copy gets its own TypedDict beside
  ``Baseline`` rather than loosening the reference's.

Changes that land in parallel each add their own keys; whichever merges second adds its keys
to the schema already there. ``tests/test_schema.py`` checks real reports against these, both
ways: a key the code writes but the schema lacks fails, and so does a key the schema declares
that none of its reports write (add a report that writes it there).

This module deliberately does not use ``from __future__ import annotations``: with string
annotations, Python (3.11 to 3.14 at least) counts ``NotRequired`` keys as required in
``__required_keys__``, which ``find_problem`` and the tests rely on.
"""

import functools
import json
import types
from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    NamedTuple,
    NotRequired,
    TypedDict,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

# A chunk's metrics, per area (``surface.Metrics``): None where a metric does not apply.
ChunkMetrics = dict[str, dict[str, float | None]]
# One number per metric, per area: z-scores, contrast effects, held-out reliability.
Grouped = dict[str, dict[str, float]]


# Settings


class SyntaxUsed(TypedDict):
    """The spaCy parser that ran, or ``syntax_used: None`` when none did."""

    model: str
    model_version: str
    spacy_version: str


class InputSettings(TypedDict, total=False):
    """``api.Settings`` as a report records it (``Settings.to_report``), plus the inputs.

    ``build``, ``Profile.score`` and ``evaluate`` record every field; the lower-level
    functions record only what they are passed, so each field may be absent."""

    window_words: int
    min_words: int
    text_field: str | None
    syntax: Literal["auto"] | bool
    input_format: str
    group_field: str | None
    pool: Literal["auto"] | bool
    # Whether short texts were joined into windows (``pool`` is what was asked).
    pool_used: bool
    # "auto", "heading", "heading:1" to "heading:6", "rule" or "none" (``split.SplitOn``).
    split_on: str
    # How texts were split into documents (``split_on`` is what was asked): "heading",
    # "rule" and "stand-in" (consecutive windows of a text with neither), sorted; empty when
    # nothing was split.
    split_used: list[str]
    # Paths as given, and ``Text`` or ``Chunk`` inputs by name.
    inputs: list[str]


class ReportSettings(InputSettings):
    """A reference or score report's settings."""

    top_k: int
    syntax_used: SyntaxUsed | None
    # A reference's contrast inputs; None when built without them.
    contrast: NotRequired[list[str] | None]
    # How a reference's contrast set was split, as ``split_used`` (empty without one).
    contrast_split_used: NotRequired[list[str]]
    generic_contrast: NotRequired[bool]


class EvaluationSettings(InputSettings):
    """An evaluation report's settings. ``top_k`` does not apply to an evaluation."""

    retrain: bool
    syntax_used: SyntaxUsed | None
    contrast: NotRequired[list[str]]
    # Each edited set's inputs, by label.
    edited: NotRequired[dict[str, list[str]]]


# Parts every reference and score report has


class MetricStats(TypedDict):
    """One metric over a report's chunks. ``sd`` needs two chunks, and ``cv`` a nonzero
    mean; with no chunk measuring the metric, all but ``n`` are None."""

    n: int
    mean: float | None
    sd: float | None
    cv: float | None
    min: float | None
    max: float | None


# Area -> metric -> statistics.
Summary = dict[str, dict[str, MetricStats]]


class ChunkRow(TypedDict):
    id: str
    source: str
    metrics: ChunkMetrics


class ReportBase(TypedDict):
    """What reference and score reports share: their own chunks, measured."""

    version: int
    settings: ReportSettings
    chunk_count: int
    # Distinct documents the chunks came from (windows of one file are one document).
    document_count: int
    word_count: int
    summary: Summary
    # Distribution name -> entry -> share, with the long tail pooled as ``<other>``.
    distributions: dict[str, dict[str, float]]
    warnings: list[str]


# Reference profile


class GroupRange(TypedDict):
    """A held-out Delta range: typical, the centre a mean over many chunks is read against
    (an upper confidence bound on the mean, ``weighting.upper_mean``), 95th percentile, and
    the share of their variation the chunks of one run share (``weighting.run_similarity``,
    from the windows' held-out values for this range)."""

    median: float
    mean: float
    p95: float
    similarity: float


class DeltaRange(GroupRange):
    """The reference's own Delta on held-out chunks, overall and per area."""

    max: float
    by_group: dict[str, GroupRange]


class LengthDelta(GroupRange):
    """A calibrated length's Delta range on held-out pieces, overall and per area. Every
    95th percentile is an upper confidence bound; the 99th is a plain percentile."""

    p99: float
    by_group: dict[str, GroupRange]


class LengthLikenessContrast(TypedDict):
    median: float


class LengthLikeness(TypedDict):
    """A calibrated length's likeness range: held-out reference pieces, and contrast pieces."""

    reference: GroupRange
    contrast: LengthLikenessContrast


class BaselineLength(TypedDict):
    """One length of ``calibration.by_length`` as a score report's baseline copies it:
    everything but the per-metric ``reliability``, which only scoring reads."""

    # The pieces cut at this length, the documents they came from, and their median length
    # in prose words.
    pieces: int
    documents: int
    words: float
    # How many independent pieces they are worth; absent with too few pieces or documents.
    effective: NotRequired[float]
    # Only for a length with enough independent pieces (``calibration.enough``).
    delta: NotRequired[LengthDelta]
    # Only for a length with a Delta range, in a reference with a contrast set; the
    # likeness range needs enough contrast pieces too.
    contrast_pieces: NotRequired[int]
    likeness: NotRequired[LengthLikeness]


class LengthCalibration(BaselineLength):
    """One length of ``calibration.by_length``: the reference's held-out range for texts of
    about that many words (``calibration.calibrate_length``)."""

    # Held-out rms per metric at this length; only with ``delta``.
    reliability: NotRequired[Grouped]


class DriftTail(TypedDict):
    """The upper tail of a paragraph null (``drift.fit_tail``): ``share`` of ``count`` values
    exceed ``threshold``, by ``scale`` on average (an exponential tail)."""

    count: int
    threshold: float
    share: float
    scale: float


class DriftLevels(TypedDict):
    """The spread of held-out reference documents' levels: the median, and the
    ``drift.LEVEL_SHARE`` quantile."""

    median: float
    high: float


class DriftCalibration(TypedDict):
    """The null of the paragraph statistic, from held-out reference documents read in parts
    (``profile._calibrate_drift``), which sets each scored document's thresholds."""

    # The score spans are judged by, the span length and the chance of any false flag in a
    # document of the writer's that the thresholds aim at.
    by: Literal["delta", "likeness"]
    span_words: int
    alpha: float
    # What the null rests on.
    documents: int
    paragraphs: int
    # The spread of the documents' levels (each document's median statistic), the tail of a
    # paragraph's excess over its document's level (paragraphs on two spans), and the tail
    # of the statistic of paragraphs on one span. None when the reference is too small to
    # estimate one; without ``levels`` and ``within`` no paragraph drifts, and without
    # ``one`` no paragraph on one span does.
    levels: DriftLevels | None
    within: DriftTail | None
    one: DriftTail | None


class _CalibrationBase(TypedDict):
    # How many documents the chunks came from.
    sources: int
    delta: DeltaRange
    # The median length of the reference's chunks, in prose words: the longest anchor of
    # the length calibration.
    chunk_words: float


class Calibration(_CalibrationBase):
    """How Delta behaves on the reference's own held-out writing, at its chunks' length and
    (``by_length``) for shorter texts."""

    # Length in words ("75", "150", "300") -> its calibration. Empty when the chunks are too
    # short to cut into pieces.
    by_length: dict[str, LengthCalibration]
    # The paragraph null (``drift``); only for a reference whose shorter lengths were
    # calibrated (not for ``evaluate``). A score's baseline leaves it out.
    drift: NotRequired[DriftCalibration]


class BaselineCalibration(_CalibrationBase):
    """A reference's ``calibration`` as a score report's baseline copies it (``_without_rms``)."""

    by_length: dict[str, BaselineLength]


class LikenessRange(TypedDict):
    """Held-out likeness of the reference's own chunks."""

    median: float
    # An upper confidence bound on the mean (``weighting.upper_mean``).
    mean: float
    p95: float
    # The share of their variation a run's chunks share (``weighting.run_similarity``).
    similarity: float
    max: float


class ContrastRange(TypedDict):
    """Held-out likeness of the contrast chunks."""

    median: float
    min: float


class Bootstrap(TypedDict):
    """How an AUC's confidence interval was found (``weighting.auc_confidence``)."""

    # "exact" when every resample would give the same AUC, so none were drawn; None, with no
    # interval, when a side has fewer than 2 documents.
    method: Literal["exact", "bootstrap"] | None
    resamples: int
    unit: str
    reference_documents: int
    contrast_documents: int


class AucConfidence(TypedDict):
    """An AUC's 95% document-bootstrap interval, and how it was found."""

    # None with fewer than 2 documents on a side.
    auc_ci: list[float] | None
    bootstrap: Bootstrap


class AucResult(AucConfidence):
    """An AUC of contrast likeness over the reference's, with its interval."""

    auc: float | None


class LengthBaseline(TypedDict):
    """How well chunk word count alone separates the contrast set."""

    auc: float
    direction: str
    reference_median_words: float
    contrast_median_words: float


class ContrastCalibration(AucResult):
    """Where likeness falls for held-out reference and contrast chunks, and how well it
    separates them."""

    reference: LikenessRange
    contrast: ContrastRange
    length_baseline: LengthBaseline | None
    cross_validated: bool


class LearnedContrast(TypedDict):
    """What ``weighting.summarize_contrast`` learns from the contrast set."""

    effects: Grouped
    calibration: ContrastCalibration


class Contrast(LearnedContrast):
    """A reference's contrast set (``--contrast``): its name, size and what was learned."""

    label: str
    chunk_count: int
    sources: int


class ReferenceReport(ReportBase):
    """A reference profile, as ``styleprofile build`` saves it."""

    kind: Literal["reference"]
    # Only with ``keep_chunks``: scoring needs only the summary.
    chunks: NotRequired[list[ChunkRow]]
    # Held-out reliability and calibration need chunks from at least two documents.
    reliability: NotRequired[Grouped]
    calibration: NotRequired[Calibration]
    contrast: NotRequired[Contrast]


# Score report


class Deviation(TypedDict):
    metric: str
    z: float
    value: float | None
    reference_mean: float | None


class Unseen(TypedDict):
    """A metric the reference never varies on, on which the chunk differs."""

    metric: str
    value: float | None
    reference_value: float | None


class LikenessSignal(TypedDict):
    metric: str
    z: float
    # Its share of the chunk's likeness.
    contribution: float


class LikenessAtLength(GroupRange):
    """The likeness range at one chunk's length, and the contrast set's typical score."""

    target: float


class ChunkCalibration(TypedDict):
    """What judging one chunk read from the reference, for its length
    (``calibration.AtLength.row``). The ranges are None or empty without calibration."""

    # False when the chunk is too short to judge; ``reason`` then says why, and the ranges
    # are the nearest calibrated length's, for indicative numbers only.
    judged: bool
    reason: str | None
    delta: GroupRange | None
    delta_by_group: dict[str, GroupRange]
    likeness: LikenessAtLength | None
    # How many times more each metric swings at this length than in a window (above 1 only).
    length_scale: Grouped
    # Without calibration, how many times wider the fixed steps are read at this length.
    stretch: float


class ChunkScore(TypedDict):
    """One chunk scored against the reference."""

    delta: float | None
    delta_by_group: dict[str, float]
    divergence: dict[str, float | None]
    largest_deviations: list[Deviation]
    unseen_in_reference: list[Unseen]
    z: Grouped
    # Only against a reference with a contrast set.
    likeness: NotRequired[float]
    likeness_signals: NotRequired[list[LikenessSignal]]
    calibration: ChunkCalibration


class ScoredChunk(ChunkRow):
    reference: ChunkScore


class BaselineStats(TypedDict):
    mean: float | None
    sd: float | None


class BaselineContrast(TypedDict):
    label: str
    calibration: ContrastCalibration


class Baseline(TypedDict):
    """What rendering a score needs from its reference, copied into the score report so it
    can be shown without the reference file. Unlike the reference's own keys,
    ``calibration`` and ``contrast`` are always present, and None when it has none."""

    chunk_count: int
    summary: dict[str, dict[str, BaselineStats]]
    calibration: BaselineCalibration | None
    contrast: BaselineContrast | None


class VerdictDelta(TypedDict):
    """Mean Delta over the judged chunks against the pooled range of their lengths."""

    value: float | None
    # The ranges' mean median and 95th percentile, and the bound of the mean; None when a
    # chunk's length has no range (the reference has no calibration).
    typical: float | None
    p95: float | None
    ceiling: float | None
    # 0-3 (``core.DISTANCES``); None when too short to judge or not comparable.
    level: int | None


class VerdictArea(TypedDict):
    """One area's mean Delta against the pooled range of the chunks' lengths."""

    value: float
    typical: float | None
    ceiling: float | None
    # ``value`` over ``ceiling``; None without a range.
    relative: float | None
    level: int | None


class VerdictLikeness(TypedDict):
    """Mean likeness against the pooled likeness range of the chunks' lengths."""

    value: float
    typical: float | None
    p95: float | None
    # The contrast set's typical likeness at these lengths.
    target: float | None
    ceiling: float | None
    level: int | None
    verdict: str


class ScoreVerdict(TypedDict):
    """The headline verdict of a score report (``calibration.verdict``): over the chunks
    long enough to judge, or, when none is, over all of them as an indication only."""

    judged: bool
    # Prose words and chunks in the report, and how many chunks the verdict judges.
    words: int
    chunks: int
    chunks_judged: int
    # Why there is no verdict; None when there is one.
    reason: str | None
    # The setting ``reason`` names (``window_words``), for a front end to name its own way.
    setting: str | None
    delta: VerdictDelta
    # The verdict words, or "too short to judge".
    verdict: str
    by_group: dict[str, VerdictArea]
    # None without a contrast set (or with no chunk scored for likeness).
    likeness: VerdictLikeness | None
    # How many judged chunks are flagged on their own (``calibration.chunk_flagged``): the
    # verdict judges their mean, which a few very different chunks move little.
    flagged: int


class ReferenceScore(TypedDict):
    """A score report's figures over all of its chunks, and its reference's baseline."""

    # Where the reference was loaded from or saved, if anywhere.
    path: str | None
    chunk_count: int
    # None when no metric could be compared.
    delta_mean: float | None
    # None without a contrast set.
    likeness_mean: float | None
    delta_by_group_mean: dict[str, float | None]
    divergence_mean: dict[str, float | None]
    pooled_divergence: dict[str, float | None]
    verdict: ScoreVerdict
    baseline: Baseline


class DocumentDifference(TypedDict):
    """A metric (``group.name``) and its mean z over a document's judged chunks, uncapped."""

    metric: str
    z: float


class DocumentSignal(DocumentDifference):
    """A metric that adds to a document's likeness, and its mean share of it."""

    contribution: float


class DocumentEntry(TypedDict):
    """One scored document (a file, a JSONL record or a ``Text``), judged on its own chunks
    by ``calibration.verdict``, as the headline is over all of them (plan PR 7)."""

    # How reports list it (``posts/a.md``, ``comments.jsonl:17``), and its saved source.
    name: str
    path: str
    chunks: int
    words: int
    # As ``ScoreVerdict``: whether any chunk is long enough to judge, how many are, and why
    # there is no verdict (None when there is one).
    judged: bool
    chunks_judged: int
    reason: str | None
    # Means over the judged chunks (all of them when none is); None when not comparable.
    delta: float | None
    # The verdict words: a distance, "too short to judge" or "not comparable".
    verdict: str
    # None without a contrast set.
    likeness: float | None
    likeness_verdict: str | None
    # As ``ScoreVerdict.flagged``: its judged chunks flagged on their own.
    flagged: int
    differences: list[DocumentDifference]
    # Only against a reference with a contrast set.
    signals: NotRequired[list[DocumentSignal]]


class FailLevels(TypedDict):
    """The levels ``score --fail-above``, ``--fail-likeness`` and ``--fail-flagged`` asked
    for; None if not."""

    above: Literal["somewhat", "clearly", "very"] | None
    likeness: Literal["few", "leans", "like"] | None
    # The fewest chunks flagged on their own that fail a document.
    flagged: int | None


class FailedDocument(TypedDict):
    """A document that reached a fail level or could not be compared.

    ``reason`` names an inability to compare; verdicts are None in that case.
    """

    reason: NotRequired[Literal["could not be compared with the reference"]]
    name: str
    path: str
    delta: str | None
    likeness: str | None
    # How many of its judged chunks are flagged on their own (``DocumentEntry.flagged``),
    # whichever check it failed, and how many were judged.
    flagged: int
    chunks_judged: int


class PassageTrait(TypedDict):
    """One way a span differs from the writer (``drift.traits``)."""

    metric: str
    # In standard deviations of the writer's own text at the span's length.
    z: float
    value: float | None
    # The writer's mean.
    reference: float | None


class SpanScore(TypedDict):
    """One span of at least ``span_words`` words, judged at its length (``drift.judge_span``)."""

    # The paragraphs it covers, as a [start, end) range of indices into ``paragraphs``.
    paragraphs: list[int]
    words: int
    # The score it is judged by: likeness with a likeness range at its length, else Delta.
    by: Literal["delta", "likeness"]
    # That score over its 95% bound; None when the span is not judged.
    relative: float | None
    delta: float | None
    # The Delta verdict's bound, with its floor.
    ceiling: float | None
    likeness: float | None
    likeness_ceiling: float | None
    # The Delta level (0-3) against its length's range; None when not judged.
    level: int | None


class ParagraphScore(TypedDict):
    """One paragraph and how it reads, from the lower of its two spans (``drift.judge``)."""

    # The first and last line of its prose in the document as read.
    lines: list[int]
    words: int
    # The start of its prose, about 60 characters.
    excerpt: str
    # How many judged spans its statistic rests on (1 or 2); the figures below are None
    # when none.
    spans: int
    by: Literal["delta", "likeness"] | None
    # Its statistic: the lower span's score over its 95% bound; and the threshold it drifts
    # above (None when the reference has no null for it).
    relative: float | None
    threshold: float | None
    delta: float | None
    level: int | None
    likeness: float | None
    likeness_level: int | None
    # The paragraph's own traits, measured on it alone.
    traits: list[PassageTrait]
    # Its own score over the bound at its length, read alone (not a verdict: below 75 words
    # the bound is only widened). One above 1 reads unlike the writer by itself, so no
    # neighbour explains it away.
    alone: float | None
    # Whether it drifts (a paragraph's own flag, distinct from a chunk's ``flagged``), and
    # when its statistic is above its span's 95% bound but it does not, why.
    drifts: bool
    note: str | None


class DriftThresholds(TypedDict):
    """A document's thresholds for paragraphs resting on two spans and on one (None: a
    paragraph on one span never drifts against this reference)."""

    two: float
    one: float | None


class DocumentPassages(TypedDict):
    """Where one scored document drifts (``profile._passages``)."""

    # As ``documents`` names it (``DocumentEntry.name``), and its saved source.
    name: str
    source: str
    # Read from HTML: its lines are those of the Markdown conversion.
    converted: bool
    # Scored as pooled windows of records, so not read in parts (``reason`` says so).
    pooled: bool
    words: int
    span_words: int
    # False when no span could be judged; ``reason`` says why, and ``paragraphs`` is empty.
    judged: bool
    reason: str | None
    # The score its spans are judged by; None when not judged.
    by: Literal["delta", "likeness"] | None
    # Whether the reference has a null for that score (``calibration.drift``), so
    # paragraphs can be flagged at all, and the thresholds it sets for this document.
    sensitive: bool
    # The document's own level (``drift.level``): its median paragraph statistic.
    own_level: float | None
    thresholds: DriftThresholds | None
    spans: list[SpanScore]
    paragraphs: list[ParagraphScore]


class ScoreReport(ReportBase):
    """Drafts scored against a reference, as ``styleprofile score`` saves them."""

    kind: Literal["score"]
    chunks: list[ScoredChunk]
    # Each document's own verdict, in input order.
    documents: list[DocumentEntry]
    # Each document read in overlapping spans (``drift``); None when the score did not
    # (by default, a score of several documents).
    passages: list[DocumentPassages] | None
    reference: ReferenceScore
    # Only when ``score`` was given --fail-above, --fail-likeness or --fail-flagged.
    fail: NotRequired[FailLevels]
    failed: NotRequired[list[FailedDocument]]


# Evaluation report


class DraftVerdict(TypedDict):
    draft: str
    chunks: int
    likeness: float
    level: int
    verdict: str


class EditStatistics(TypedDict):
    """How much an edited set changed its drafts."""

    ngram13_changed_median: float | None
    ngram13_changed_mean: float | None
    word_ratio_median: float | None


class SetSummary(AucResult):
    """One set of drafts (the originals or an edited set) scored without leaking."""

    drafts: int
    chunks: int
    likeness_median_chunks: float | None
    flagged: int
    # Verdict words -> how many drafts got them.
    verdicts: dict[str, int]
    by_draft: list[DraftVerdict]
    # Edited sets only: drafts without an edited copy, edited drafts too short to score,
    # and edited drafts skipped because their original was.
    missing: NotRequired[list[str]]
    too_short: NotRequired[list[str]]
    skipped: NotRequired[list[str]]
    partial: NotRequired[bool]
    # Partial edited sets only: the originals' AUC over the drafts the set covers.
    original_auc_same_drafts: NotRequired[float | None]
    edits: NotRequired[EditStatistics]


class Survival(TypedDict):
    z: float | None
    original_z: float | None
    # The edited set's gap from the reference over the originals'; None when too small.
    remaining: float | None


class Signal(TypedDict):
    """One of the strongest contrast metrics, and how much of it each edited set keeps."""

    metric: str
    effect: float
    reference_z: float
    original_z: float | None
    edited: dict[str, Survival]


class RetrainedSet(AucResult):
    # The set's AUC under the original weights.
    before: float | None


class Retrain(AucResult):
    by_set: dict[str, RetrainedSet]


class EvaluationReference(TypedDict):
    chunks: int
    documents: int
    likeness: LikenessRange


class EvaluationContrast(TypedDict):
    chunks: int
    drafts: int
    cross_validated: bool


class EvaluationReport(TypedDict):
    """The rewording stress test, as ``styleprofile evaluate`` saves it."""

    kind: Literal["evaluation"]
    version: int
    settings: EvaluationSettings
    label: str
    reference: EvaluationReference
    contrast: EvaluationContrast
    # Set label ("original" first) -> its results.
    sets: dict[str, SetSummary]
    signals: list[Signal]
    # Only with ``retrain``.
    retrain: Retrain | None
    warnings: list[str]


Report = ReferenceReport | ScoreReport | EvaluationReport


# Checking a loaded report


class Problem(NamedTuple):
    """Why a report cannot be read: the dotted path of the part (``reference.baseline``,
    ``chunks[3].reference``) and what is wrong with it, ``"missing"`` or what was found
    where something else was needed (``"null, not an object"``)."""

    path: str
    reason: str


def find_problem(value: object, schema: type) -> Problem | None:
    """The first part of ``value`` that ``schema`` (a TypedDict) cannot describe, or None.

    A required key must be present; a TypedDict must be an object, a ``list`` a list and a
    ``dict`` an object; null is only allowed where the type says ``| None``; and a TypedDict's
    own numbers, strings and true-or-false values must be of their type. Nested TypedDicts
    are followed through lists, dict values and ``X | None``. The entries of maps of plain
    values, such as the per-metric numbers a large reference has thousands of, are not
    checked. A key the schema does not know is allowed."""
    return _checker(schema).check(value, "")


class _Checker:
    """Checks values of one type. Built once per type (``_checker``), so checking a report
    with tens of thousands of chunks costs little more than walking it."""

    def __init__(self, hint: Any) -> None:
        self.optional = _optional(hint)
        hint = _strip_none(hint)
        self.expected = _describe(hint)
        # A plain value (not a TypedDict, list or dict) whose entries a map need not visit.
        self.plain = not (is_typeddict(hint) or get_origin(hint) in (list, dict))
        self.fields: list[tuple[str, _Checker]] = []
        self.required: list[str] = []
        self.container: type | None = None
        self.item: _Checker | None = None
        self.types: frozenset[type] | None = None
        self.values: tuple[object, ...] = ()
        if is_typeddict(hint):
            self.container = dict
            self.required = sorted(hint.__required_keys__)
            self.fields = [(key, _checker(item)) for key, item in get_type_hints(hint).items()]
        elif get_origin(hint) in (list, dict):
            self.container = get_origin(hint)
            item = _checker(get_args(hint)[-1])
            self.item = None if item.plain else item
        else:
            self.types, self.values = _plain_types(hint)

    def check(self, value: object, path: str) -> Problem | None:
        if value is None:
            return None if self.optional else Problem(path, f"null, not {self.expected}")
        if self.container is None:
            if self.types is None or type(value) in self.types or value in self.values:
                return None
            return Problem(path, f"{_found(value)}, not {self.expected}")
        if not isinstance(value, self.container):
            return Problem(path, f"{_found(value)}, not {self.expected}")
        if self.fields:
            return self._check_fields(value, path)  # type: ignore[arg-type]
        item = self.item
        if item is None:
            return None
        entries = (
            ((f"{path}[{index}]", entry) for index, entry in enumerate(value))  # type: ignore[arg-type]
            if self.container is list
            else ((f"{path}.{key}", entry) for key, entry in value.items())  # type: ignore[union-attr]
        )
        for where, entry in entries:
            found = item.check(entry, where)
            if found:
                return found
        return None

    def _check_fields(self, value: dict[str, object], path: str) -> Problem | None:
        for key in self.required:
            if key not in value:
                return Problem(f"{path}.{key}" if path else key, "missing")
        for key, field in self.fields:
            if key in value:
                entry = value[key]
                # A plain value is checked in place; the path is built only for a problem.
                if field.container is None and entry is not None:
                    if field.types is None or type(entry) in field.types:
                        continue
                    if entry in field.values:
                        continue
                found = field.check(entry, f"{path}.{key}" if path else key)
                if found:
                    return found
        return None


@functools.cache
def _checker(hint: Any) -> _Checker:
    return _Checker(hint)


def _plain_types(hint: Any) -> tuple[frozenset[type] | None, tuple[object, ...]]:
    """The exact types a plain value of ``hint`` may have (a number may be an int), and the
    values a ``Literal`` allows; ``(None, ())`` for a hint that is not checked."""
    arms = get_args(hint) if get_origin(hint) in (Union, types.UnionType) else (hint,)
    allowed: set[type] = set()
    values: list[object] = []
    for arm in arms:
        if arm is type(None):
            continue
        if get_origin(arm) is Literal:
            values += get_args(arm)
        elif arm is float:
            allowed |= {int, float}
        elif arm in (int, str, bool):
            allowed.add(arm)
        else:
            return None, ()
    return frozenset(allowed), tuple(values)


def _found(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        return "an object"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, str):
        return "a string"
    return "a number"


def _optional(hint: Any) -> bool:
    return hint is type(None) or (
        get_origin(hint) in (Union, types.UnionType) and type(None) in get_args(hint)
    )


def _strip_none(hint: Any) -> Any:
    """``X`` for ``X | None``; any other hint as it is."""
    if get_origin(hint) in (Union, types.UnionType):
        arms = [arm for arm in get_args(hint) if arm is not type(None)]
        if len(arms) == 1:
            return arms[0]
    return hint


def _describe(hint: Any) -> str:
    if is_typeddict(hint) or get_origin(hint) is dict:
        return "an object"
    if get_origin(hint) is list:
        return "a list"
    if hint is str:
        return "a string"
    if hint is bool:
        return "true or false"
    if hint in (int, float):
        return "a number"
    if get_origin(hint) is Literal:
        return "one of " + ", ".join(json.dumps(choice) for choice in get_args(hint))
    return "a value"
