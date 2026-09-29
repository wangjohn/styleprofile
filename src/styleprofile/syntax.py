"""Parser-based metrics: clause structure, part-of-speech mix, and how sentences open.

spaCy is an optional dependency: install the `syntax` extra (`styleprofile[syntax]`), then
its English model with `styleprofile setup` (see ``styleprofile.spacy_model``). Tags come from an
automatic parser, so passive and nominalization counts in particular are approximate.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from importlib.util import find_spec
from typing import Any

from styleprofile.core import StyleProfileError
from styleprofile.metrics import grouped
from styleprofile.surface import Metrics

DEFAULT_MODEL = "en_core_web_sm"
# The model version the package is tested with, which `styleprofile setup` installs. Keep it
# equal to the spacy-model dependency group in pyproject.toml (a test checks).
MODEL_VERSION = "3.8.0"
_CLAUSE_DEPS = frozenset({"advcl", "ccomp", "relcl", "acl", "xcomp", "csubj"})
_PASSIVE_DEPS = frozenset({"nsubjpass", "auxpass", "csubjpass"})
# A nominalization is a noun that names the action, state or quality of a *different*
# English verb or adjective, formed with one of these suffixes: decide -> decision, move ->
# motion, fuse -> fusion, argue -> argument, dark -> darkness, able -> ability, distant ->
# distance, silent -> silence. Judged from spelling alone:
#
# - the word keeps at least _MIN_NOMINALIZATION_STEM letters before the suffix, which rules
#   out fence, city, pity and dance, whose ending is part of a one-syllable root;
# - it is not in _NOT_NOMINALIZATIONS, the frequent words the suffix test gets wrong, by
#   one rule: no different verb or adjective whose action, state or quality the noun
#   names. That covers words whose ending is part of the root (nation, station, moment,
#   chance), words with no English base (science, quality, community, university,
#   tradition), and nouns whose only related verb is the same word (question, comment,
#   document, influence, experience, sentence);
# - US and British spellings count alike (_US_SPELLINGS): defense as defence.
#
# The list names frequent words only, so the metric stays approximate. It is also somewhat
# topic-sensitive: a text about institutions or generations has more nominalizations than
# one about fences, whoever writes it (see docs/method.md).
_NOMINALIZATION = re.compile(r"^(?P<stem>.+?)(?:tion|sion|ment|ness|ity|ance|ence)$")
_MIN_NOMINALIZATION_STEM = 2
_US_SPELLINGS = {"defense": "defence", "offense": "offence", "pretense": "pretence"}
_NOT_NOMINALIZATIONS = frozenset(
    {
        # -tion, -sion: the ending is part of the root, or there is no English base.
        "nation", "notion", "lotion", "potion", "ration", "auction", "station", "section",
        "fiction", "portion", "position", "tradition", "question", "mention", "mansion",
        "mission", "session", "pension", "passion", "version", "dimension", "occasion",
        "television", "lesion",
        # -ment
        "moment", "cement", "lament", "document", "element", "segment", "garment",
        "comment", "apartment", "department", "experiment", "instrument", "basement",
        "compliment", "supplement", "sediment", "parliament", "monument", "ornament",
        "fragment", "pigment", "testament", "tournament",
        # -ness
        "business", "witness", "harness", "wilderness",
        # -ity
        "deity", "entity", "charity", "quality", "identity", "community", "university",
        "opportunity", "authority",
        # -ance, -ence
        "chance", "glance", "stance", "advance", "balance", "finance", "instance",
        "romance", "substance", "circumstance", "ambulance", "nuisance", "renaissance",
        "sentence", "science", "audience", "sequence", "essence", "influence", "licence",
        "conscience", "experience",
    }
)  # fmt: skip
_MIN_SENTENCE_WORDS = 3
# Texts per spaCy batch. On the medium benchmark corpus (500-word windows), speed hardly
# depends on it (11.8s to 12.9s for every size from 2 to 32), but memory grows with it: a
# process parsing peaks at 200 MB with 2, 280 MB with 4, 650 MB with 16 (the old value) and
# 1.7 GB with 64.
BATCH_SIZE = 4
# The pipeline components the metrics read: tokens, tags (``tag_`` and, through the attribute
# ruler, ``pos_``) and the dependency parse, which also sets sentence boundaries. Everything
# else is left out of the load entirely (``exclude``), not just switched off.
EXCLUDED = ("ner", "lemmatizer", "senter")


class SyntaxUnavailableError(StyleProfileError):
    """spaCy or its English model is not installed. ``model_missing`` is True when spaCy is
    installed but the model is not, which `styleprofile setup` fixes."""

    def __init__(self, message: str, *, model_missing: bool = False) -> None:
        super().__init__(message, code="syntax_unavailable")
        self.model_missing = model_missing


class Parser:
    """The spaCy parser: its model's name and versions, and the pipeline itself (``nlp``),
    which a lazily made parser (``load_parser(lazy=True)``) loads only when something first
    parses in this process, since worker processes load their own."""

    def __init__(
        self,
        nlp: Any,
        model: str,
        model_version: str,
        spacy_version: str,
        *,
        loader: Callable[[], Any] | None = None,
    ) -> None:
        self._nlp = nlp
        self._loader = loader
        self.model = model
        self.model_version = model_version
        self.spacy_version = spacy_version

    @property
    def nlp(self) -> Any:
        if self._nlp is None:
            assert self._loader is not None
            self._nlp = self._loader()
        return self._nlp

    @property
    def loaded(self) -> bool:
        return self._nlp is not None

    def docs(self, texts: Iterable[str]) -> Iterator[Any]:
        """Each text parsed once, as a spaCy ``Doc``."""
        yield from self.nlp.pipe(texts, batch_size=BATCH_SIZE)

    def parse(self, texts: Iterable[str]) -> Iterator[tuple[Metrics, Counter[str]]]:
        for doc in self.docs(texts):
            yield syntax_metrics(doc), pos_trigrams(doc)

    def used(self) -> dict[str, str]:
        """The model and versions, as a report records them (``syntax_used``)."""
        return {
            "model": self.model,
            "model_version": self.model_version,
            "spacy_version": self.spacy_version,
        }


def _load(model: str) -> Any:
    try:
        spacy = import_module("spacy")
    except ImportError as error:
        raise SyntaxUnavailableError(
            f"syntax metrics need spaCy and {model!r}; install the `syntax` extra "
            "(pip install 'styleprofile[syntax]'), then run `styleprofile setup`, or profile "
            "without syntax metrics"
        ) from error
    try:
        nlp = spacy.load(model, exclude=list(EXCLUDED))
    except OSError as error:
        raise SyntaxUnavailableError(
            f"syntax metrics need spaCy's English model {model!r}, which is not installed; "
            + (
                "run `styleprofile setup` to install it"
                if model == DEFAULT_MODEL
                else f"install it with `python -m spacy download {model}`"
            )
            + ", or profile without syntax metrics",
            model_missing=True,
        ) from error
    nlp.max_length = 5_000_000
    return nlp


def _installed_meta(model: str) -> tuple[str, str] | None:
    """The model's version (from its ``meta.json``, as the loaded model reports it) and
    spaCy's, without importing either; None when they cannot be read that way."""
    try:
        if find_spec("spacy") is None or find_spec(model) is None:
            return None
        meta = json.loads((files(model) / "meta.json").read_text(encoding="utf-8"))
        return str(meta.get("version", "")), version("spacy")
    except (ImportError, OSError, ValueError, PackageNotFoundError, ModuleNotFoundError):
        return None


def load_parser(model: str = DEFAULT_MODEL, *, lazy: bool = False) -> Parser:
    """spaCy's ``model``, loaded now, or with ``lazy`` only when first used in this process
    (its name and versions are read from its package; if they cannot be, it loads now)."""
    if lazy:
        found = _installed_meta(model)
        if found is not None:
            return Parser(None, model, *found, loader=lambda: _load(model))
    nlp = _load(model)
    spacy = import_module("spacy")
    return Parser(nlp, model, str(nlp.meta.get("version", "")), str(spacy.__version__))


def _depth(root: Any) -> int:
    """Height of a parse tree, iteratively so long run-on sentences cannot hit recursion limits."""
    deepest = 0
    stack = [(root, 0)]
    while stack:
        token, depth = stack.pop()
        deepest = max(deepest, depth)
        stack.extend((child, depth + 1) for child in token.children)
    return deepest


def _pct(part: int, whole: int) -> float | None:
    return 100 * part / whole if whole else None


def _sentences(doc: Any) -> list[Any]:
    """The sentences of a ``Doc``, or of a ``Span`` clipped to its edges: ``Span.sents``
    yields whole sentences, running past the span when the parser puts a sentence boundary
    somewhere other than where the span starts or ends."""
    if not hasattr(doc, "start"):
        return list(doc.sents)
    whole = doc.doc
    clipped = [
        whole[max(sentence.start, doc.start) : min(sentence.end, doc.end)] for sentence in doc.sents
    ]
    return [sentence for sentence in clipped if len(sentence)]


def syntax_metrics(doc: Any) -> Metrics:
    """Parser-based metrics for one parsed chunk, named and grouped by the registry.

    ``doc`` is a spaCy ``Doc`` or a ``Span`` of one, so a part of a parsed text (a shorter
    piece cut from a window, say) is measured without parsing it again.
    """
    words = [token for token in doc if token.is_alpha]
    per_1k = 1000 / len(words) if words else None

    def rate(count: int) -> float | None:
        return count * per_1k if per_1k is not None else None

    sentences = [
        sentence
        for sentence in _sentences(doc)
        if sum(token.is_alpha for token in sentence) >= _MIN_SENTENCE_WORDS
    ]
    openers = [next(token for token in sentence if token.is_alpha) for sentence in sentences]
    pos = Counter(token.pos_ for token in words)
    clauses = sum(token.dep_ in _CLAUSE_DEPS for sentence in sentences for token in sentence)
    passive = sum(any(token.dep_ in _PASSIVE_DEPS for token in sentence) for sentence in sentences)

    def opener_pct(*tags: str) -> float | None:
        return _pct(sum(token.pos_ in tags for token in openers), len(openers))

    values: dict[str, float | None] = {
        "parse_depth_mean": (
            statistics.fmean(_depth(sentence.root) for sentence in sentences) if sentences else None
        ),
        "clauses_per_sentence": clauses / len(sentences) if sentences else None,
        "passive_sentences_pct": _pct(passive, len(sentences)),
        "noun_verb_ratio": (pos["NOUN"] + pos["PROPN"]) / pos["VERB"] if pos["VERB"] else None,
        "adjectives_per_1k": rate(pos["ADJ"]),
        "adverbs_per_1k": rate(pos["ADV"]),
        "modals_per_1k": rate(sum(token.tag_ == "MD" for token in words)),
        "nominalizations_per_1k": rate(
            sum(token.pos_ == "NOUN" and is_nominalization(token.lower_) for token in words)
        ),
        "opens_pronoun_pct": opener_pct("PRON"),
        "opens_determiner_pct": opener_pct("DET"),
        "opens_adverb_or_preposition_pct": opener_pct("ADV", "ADP"),
        "opens_subordinator_pct": opener_pct("SCONJ"),
        "opens_conjunction_pct": opener_pct("CCONJ"),
        "opens_noun_pct": opener_pct("NOUN", "PROPN"),
        "opens_verb_pct": opener_pct("VERB", "AUX"),
    }
    return grouped(values, syntax=True)


def _singular(word: str) -> str:
    """``word`` without a plural ending, for the endings nominalizations take."""
    if word.endswith("ities"):
        return word[:-3] + "y"
    if word.endswith("nesses"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def is_nominalization(word: str) -> bool:
    """Whether a noun reads as a nominalization, judged from its spelling alone: a
    nominalizing suffix after a long enough stem, and not a known exception. Plurals
    (decisions, abilities, weaknesses) count as their singular, and US spellings as the
    British ones (defense as defence)."""
    singular = _singular(word.lower())
    singular = _US_SPELLINGS.get(singular, singular)
    match = _NOMINALIZATION.match(singular)
    return (
        match is not None
        and len(match["stem"]) >= _MIN_NOMINALIZATION_STEM
        and singular not in _NOT_NOMINALIZATIONS
    )


def pos_trigrams(doc: Any) -> Counter[str]:
    """The grammatical transition matrix: coarse tag trigrams within each sentence."""
    counts: Counter[str] = Counter()
    for sentence in doc.sents:
        tags = [token.pos_ for token in sentence if not token.is_space]
        counts.update(" ".join(tags[index : index + 3]) for index in range(len(tags) - 2))
    return counts
