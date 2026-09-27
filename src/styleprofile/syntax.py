"""Parser-based metrics: clause structure, part-of-speech mix, and how sentences open.

spaCy is an optional dependency: install the `syntax` extra (`styleprofile[syntax]`). Tags
come from an automatic parser, so passive and nominalization counts in particular are
approximate.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from styleprofile.core import StyleProfileError
from styleprofile.metrics import grouped
from styleprofile.surface import Metrics

DEFAULT_MODEL = "en_core_web_sm"
_CLAUSE_DEPS = frozenset({"advcl", "ccomp", "relcl", "acl", "xcomp", "csubj"})
_PASSIVE_DEPS = frozenset({"nsubjpass", "auxpass", "csubjpass"})
_NOMINALIZATION = re.compile(r"(?:tion|sion|ment|ness|ity|ance|ence)s?$")
_MIN_SENTENCE_WORDS = 3


class SyntaxUnavailableError(StyleProfileError):
    """spaCy or its English model is not installed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="syntax_unavailable")


@dataclass(frozen=True)
class Parser:
    nlp: Any
    model: str
    model_version: str
    spacy_version: str

    def parse(self, texts: Iterable[str]) -> Iterator[tuple[Metrics, Counter[str]]]:
        for doc in self.nlp.pipe(texts, batch_size=16):
            yield syntax_metrics(doc), pos_trigrams(doc)


def load_parser(model: str = DEFAULT_MODEL) -> Parser:
    try:
        spacy = import_module("spacy")
        nlp = spacy.load(model, disable=["ner", "lemmatizer"])
    except (ImportError, OSError) as error:
        raise SyntaxUnavailableError(
            f"syntax metrics need spaCy and {model!r}; install the `syntax` extra "
            "(pip install 'styleprofile[syntax]') or profile without syntax metrics"
        ) from error
    nlp.max_length = 5_000_000
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


def syntax_metrics(doc: Any) -> Metrics:
    """Parser-based metrics for one parsed chunk, named and grouped by the registry."""
    words = [token for token in doc if token.is_alpha]
    per_1k = 1000 / len(words) if words else None

    def rate(count: int) -> float | None:
        return count * per_1k if per_1k is not None else None

    sentences = [
        sentence
        for sentence in doc.sents
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
            sum(
                token.pos_ == "NOUN" and bool(_NOMINALIZATION.search(token.lower_))
                for token in words
            )
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


def pos_trigrams(doc: Any) -> Counter[str]:
    """The grammatical transition matrix: coarse tag trigrams within each sentence."""
    counts: Counter[str] = Counter()
    for sentence in doc.sents:
        tags = [token.pos_ for token in sentence if not token.is_space]
        counts.update(" ".join(tags[index : index + 3]) for index in range(len(tags) - 2))
    return counts
