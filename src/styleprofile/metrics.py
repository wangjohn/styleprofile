"""The metric registry: every metric's group, label, unit, denominator and meaning, in one place.

``surface.py`` and ``syntax.py`` compute the values (they share tokens, sentences and parses),
and key them by the names defined here; everything else (display labels and order, which
groups are scored, each metric's resolution floor) reads this registry. The lexicons and
thresholds those metrics count with live here too, so a metric's definition is not split
between its label and its word list.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

Values = Mapping[str, float | None]
Grouped = dict[str, dict[str, float | None]]

# Thresholds.
SHORT_SENTENCE_WORDS = 8
LONG_SENTENCE_WORDS = 30
LONG_WORD_CHARS = 7

# Lexicons. Phrase patterns are written with straight apostrophes.
HEDGE = re.compile(
    r"\b(?:i think|i suspect|i guess|i'm not sure|maybe|perhaps|probably|likely|seems?|"
    r"arguably|somewhat|roughly|generally|usually|tends? to|kind of|sort of|"
    r"in my experience)\b",
    re.I,
)
BOOSTER = re.compile(
    r"\b(?:clearly|definitely|obviously|certainly|really|very|extremely|incredibly|"
    r"absolutely|totally|always|never|undoubtedly)\b",
    re.I,
)
# Words and constructions that are markedly more frequent in LLM prose than in human blogs.
# Several are ordinary words, so read this as a drift signal, not a detector.
LLM_MARKER = re.compile(
    r"\b(?:delve[sd]?|delving|crucial|pivotal|tapestry|testament|realm|landscape|"
    r"multifaceted|holistic|paramount|seamless(?:ly)?|robust|leverag(?:e|es|ed|ing)|"
    r"foster(?:s|ed|ing)?|underscor(?:e|es|ed|ing)|intricate|vibrant|streamlin(?:e|es|ed|ing)|"
    r"navigat(?:e|es|ing) the|moreover|furthermore|additionally|"
    r"it(?:'s| is) worth noting|in today's|in conclusion|ever-evolving|game-changer)\b",
    re.I,
)
NOT_JUST_BUT = re.compile(r"\bnot (?:just|only|merely|simply)\b[^.?!]{0,80}?\bbut\b", re.I)
TRANSITION_OPENER = re.compile(
    r"^(?:however|moreover|furthermore|additionally|overall|ultimately|importantly|notably|"
    r"in fact|that said|in short|in conclusion|as a result|consequently|thus|therefore)\b",
    re.I,
)
AND_BUT_SO_OPENER = re.compile(r"^(?:and|but|so)\b", re.I)

CONTRACTION_SUFFIX = re.compile(r"(?:n't|'re|'ve|'ll|'d|'m)$")
# "'s" is usually a possessive, so only these pronoun and adverb forms count as contractions.
S_CONTRACTIONS = frozenset(
    {"it's", "that's", "there's", "here's", "what's", "who's", "he's", "she's", "let's",
     "where's", "how's", "when's", "why's"}
)  # fmt: skip
NEGATIONS = frozenset({"not", "no", "never", "nothing", "nobody", "none", "neither", "nor"})
FIRST_SINGULAR = frozenset({"i", "i'm", "i've", "i'd", "i'll", "me", "my", "mine", "myself"})
FIRST_PLURAL = frozenset({"we", "we're", "we've", "we'd", "we'll", "us", "our", "ours"})
SECOND = frozenset({"you", "you're", "you've", "you'd", "you'll", "your", "yours", "yourself"})

FUNCTION_WORDS: tuple[str, ...] = (
    "the", "a", "an", "and", "but", "or", "so", "because", "if", "when", "while", "that",
    "which", "who", "this", "these", "those", "it", "its", "of", "to", "in", "on", "for",
    "with", "at", "by", "from", "as", "into", "about", "than", "then", "there", "here",
    "just", "really", "very", "much", "more", "most", "also", "even", "still", "only",
    "actually", "though", "although", "however", "instead", "rather", "where", "how", "what",
    "why", "not", "no", "all", "some", "any", "every", "both", "each", "other", "is", "are",
    "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "can", "could",
    "will", "would", "should", "might", "may", "i", "you", "we", "they", "he", "she", "my",
    "your", "our", "their",
)  # fmt: skip


class Unit(StrEnum):
    """How a metric's value reads; the value is the suffix shown after it."""

    PCT = "%"
    PER_1K = "/1k"
    WORDS = "words"
    PLAIN = ""


class Count(StrEnum):
    """A per-chunk count a rate or share is taken over; values name the ``size`` metrics.

    Parser metrics that are shares of (3+ word) sentences use the sentence count as an
    approximation, since the reference's typical parser sentence count is not recorded.
    """

    WORDS = "words"
    SENTENCES = "sentences"
    PARAGRAPHS = "paragraphs"
    APOSTROPHES = "apostrophes"


@dataclass(frozen=True)
class Metric:
    """One metric's definition.

    ``share_of`` is the count a rate or percentage is taken over, which sets the smallest
    change one occurrence can make (see ``resolution``); it is None for metrics without a
    count resolution (lengths, ratios, diversity). ``syntax`` metrics need the parser.
    """

    group: str
    name: str
    label: str
    unit: Unit
    share_of: Count | None
    syntax: bool = False
    about: str = ""

    def resolution(self, counts: Mapping[str, float]) -> float:
        """Half of one occurrence per chunk, in this metric's units: the smallest meaningful
        spread.

        ``counts`` are the reference's typical words, sentences, paragraphs and apostrophes
        per chunk. A rate per 1,000 words moves by 1000 / words per occurrence; a percentage
        by 100 / (whatever it is a share of). Other metrics have no count resolution.
        """
        if self.share_of is None:
            return 0.0
        units = counts.get(self.share_of, 0.0)
        scale = 1000.0 if self.unit is Unit.PER_1K else 100.0
        return scale / 2 / units if units else 0.0


class Group(NamedTuple):
    """A stylistic area. Delta weighs every ``scored`` area equally; ``size`` is context."""

    name: str
    title: str
    scored: bool


GROUPS: tuple[Group, ...] = (
    Group("size", "Size", scored=False),
    Group("sentence_shape", "Sentence shape", scored=True),
    Group("rhythm", "Rhythm", scored=True),
    Group("vocabulary", "Vocabulary", scored=True),
    Group("punctuation", "Punctuation", scored=True),
    Group("voice", "Voice and stance", scored=True),
    Group("markdown", "Formatting", scored=True),
    Group("syntax", "Syntax", scored=True),
    Group("sentence_openers", "How sentences open", scored=True),
    Group("function_words", "Function words", scored=True),
)
GROUP_BY_NAME: dict[str, Group] = {group.name: group for group in GROUPS}
UNSCORED_GROUPS = frozenset(group.name for group in GROUPS if not group.scored)

PCT, PER_1K, WORDS, PLAIN = Unit.PCT, Unit.PER_1K, Unit.WORDS, Unit.PLAIN
_WORDS, _SENTENCES = Count.WORDS, Count.SENTENCES


def _size(name: str, label: str, about: str) -> Metric:
    return Metric("size", name, label, PLAIN, None, about=about)


def _rate(group: str, name: str, label: str, about: str) -> Metric:
    """A count per 1,000 prose words."""
    return Metric(group, name, label, PER_1K, _WORDS, about=about)


def _opener(name: str, label: str, about: str) -> Metric:
    """A share of (3+ word) sentences whose first word has a given part of speech."""
    return Metric("sentence_openers", name, label, PCT, _SENTENCES, syntax=True, about=about)


METRICS: tuple[Metric, ...] = (
    _size("words", "Words", "Prose words, after code, tables and markup are removed."),
    _size("sentences", "Sentences", "Sentences in the prose."),
    _size("paragraphs", "Paragraphs", "Prose paragraphs; list items are not paragraphs."),
    _size("apostrophes", "Apostrophes", "Straight and curly apostrophes."),
    Metric(
        "sentence_shape",
        "sentence_words_mean",
        "Average sentence length",
        WORDS,
        None,
        about="Mean words per sentence.",
    ),
    Metric(
        "sentence_shape",
        "sentence_words_median",
        "Median sentence length",
        WORDS,
        None,
        about="Median words per sentence.",
    ),
    Metric(
        "sentence_shape",
        "sentence_words_sd",
        "Sentence length spread",
        WORDS,
        None,
        about="Standard deviation of words per sentence.",
    ),
    Metric(
        "sentence_shape",
        "short_sentences_pct",
        f"Short sentences ({SHORT_SENTENCE_WORDS} words or fewer)",
        PCT,
        _SENTENCES,
        about=f"Share of sentences of {SHORT_SENTENCE_WORDS} words or fewer.",
    ),
    Metric(
        "sentence_shape",
        "long_sentences_pct",
        f"Long sentences ({LONG_SENTENCE_WORDS}+ words)",
        PCT,
        _SENTENCES,
        about=f"Share of sentences of {LONG_SENTENCE_WORDS} words or more.",
    ),
    Metric(
        "sentence_shape",
        "paragraph_sentences_mean",
        "Sentences per paragraph",
        PLAIN,
        None,
        about="Mean sentences per paragraph.",
    ),
    Metric(
        "sentence_shape",
        "paragraph_words_mean",
        "Words per paragraph",
        WORDS,
        None,
        about="Mean words per paragraph.",
    ),
    Metric(
        "sentence_shape",
        "one_sentence_paragraphs_pct",
        "One-sentence paragraphs",
        PCT,
        Count.PARAGRAPHS,
        about="Share of paragraphs that are a single sentence.",
    ),
    Metric(
        "rhythm",
        "sentence_length_cv",
        "Sentence length variation (0 = uniform)",
        PLAIN,
        None,
        about="Sentence length spread over its mean (coefficient of variation).",
    ),
    Metric(
        "rhythm",
        "sentence_length_change_mean",
        "Length change from one sentence to the next",
        WORDS,
        None,
        about="Mean absolute difference in length between consecutive sentences.",
    ),
    Metric(
        "vocabulary",
        "word_chars_mean",
        "Average word length (letters)",
        PLAIN,
        None,
        about="Mean characters per word.",
    ),
    Metric(
        "vocabulary",
        "long_words_pct",
        f"Long words ({LONG_WORD_CHARS}+ letters)",
        PCT,
        _WORDS,
        about=f"Share of words of {LONG_WORD_CHARS} characters or more.",
    ),
    Metric(
        "vocabulary",
        "mattr_100",
        "Vocabulary variety (MATTR, 0-1)",
        PLAIN,
        None,
        about="Moving-average type/token ratio over 100-word windows.",
    ),
    Metric(
        "vocabulary",
        "mtld",
        "Lexical diversity (MTLD)",
        PLAIN,
        None,
        about="Measure of textual lexical diversity; higher repeats words less.",
    ),
    Metric(
        "vocabulary",
        "hapax_share_100",
        "Words used only once, per 100",
        PCT,
        None,
        about="Share of word types used once, averaged over 100-word blocks.",
    ),
    _rate("punctuation", "commas_per_1k", "Commas", "Commas."),
    _rate("punctuation", "semicolons_per_1k", "Semicolons", "Semicolons."),
    _rate("punctuation", "colons_per_1k", "Colons", "Colons."),
    _rate(
        "punctuation",
        "em_dashes_per_1k",
        "Em dashes",
        "Em dashes, including -- and spaced hyphens between words.",
    ),
    _rate("punctuation", "parentheses_per_1k", "Parentheses", "Opening parentheses."),
    _rate("punctuation", "questions_per_1k", "Question marks", "Question marks."),
    _rate("punctuation", "exclamations_per_1k", "Exclamation marks", "Exclamation marks."),
    _rate("punctuation", "ellipses_per_1k", "Ellipses", "Ellipses (... or the single glyph)."),
    _rate(
        "punctuation",
        "quotations_per_1k",
        "Quotations",
        "Quotations: opening curly double quotes, or pairs of straight ones.",
    ),
    Metric(
        "punctuation",
        "curly_apostrophe_pct",
        "Curly apostrophes (vs straight)",
        PCT,
        Count.APOSTROPHES,
        about="Share of apostrophes that are curly.",
    ),
    _rate(
        "voice",
        "contractions_per_1k",
        "Contractions",
        "Contractions such as don't, we're and it's (not possessives).",
    ),
    _rate("voice", "first_singular_per_1k", "I / me / my", "First-person singular."),
    _rate("voice", "first_plural_per_1k", "We / us / our", "First-person plural."),
    _rate("voice", "second_person_per_1k", "You / your", "Second person."),
    _rate("voice", "negations_per_1k", "Negations", "Negations: not, no, never, -n't."),
    _rate(
        "voice",
        "hedges_per_1k",
        "Hedges (maybe, I think)",
        "Hedges such as maybe, probably and I think.",
    ),
    _rate(
        "voice",
        "boosters_per_1k",
        "Intensifiers (really, clearly)",
        "Intensifiers such as really, clearly and always.",
    ),
    Metric(
        "voice",
        "and_but_so_openers_pct",
        "Sentences starting And / But / So",
        PCT,
        _SENTENCES,
        about="Share of sentences that open with And, But or So.",
    ),
    Metric(
        "voice",
        "transition_openers_pct",
        "Sentences starting However / Moreover",
        PCT,
        _SENTENCES,
        about="Share of sentences that open with a transition such as However or Moreover.",
    ),
    _rate(
        "voice",
        "llm_markers_per_1k",
        "LLM marker words (delve, crucial)",
        "Words and phrases markedly more common in LLM prose (a drift signal, not a detector).",
    ),
    _rate(
        "voice",
        "not_just_but_per_1k",
        '"Not just X, but Y"',
        'The "not just X, but Y" construction.',
    ),
    _rate("markdown", "headings_per_1k", "Headings", "Markdown headings."),
    _rate("markdown", "list_items_per_1k", "List items", "Markdown list items."),
    _rate("markdown", "links_per_1k", "Links", "Markdown links."),
    _rate("markdown", "bold_per_1k", "Bold phrases", "Bold phrases."),
    _rate("markdown", "code_spans_per_1k", "Inline code", "Inline code spans."),
    Metric(
        "syntax",
        "parse_depth_mean",
        "Parse tree depth (clause nesting)",
        PLAIN,
        None,
        syntax=True,
        about="Mean dependency-tree height per sentence.",
    ),
    Metric(
        "syntax",
        "clauses_per_sentence",
        "Subordinate clauses per sentence",
        PLAIN,
        None,
        syntax=True,
        about="Adverbial, relative, complement and other subordinate clauses per sentence.",
    ),
    Metric(
        "syntax",
        "passive_sentences_pct",
        "Passive sentences",
        PCT,
        _SENTENCES,
        syntax=True,
        about="Share of (3+ word) sentences with a passive construction.",
    ),
    Metric(
        "syntax",
        "noun_verb_ratio",
        "Nouns per verb",
        PLAIN,
        None,
        syntax=True,
        about="Nouns and proper nouns per verb.",
    ),
    Metric(
        "syntax",
        "adjectives_per_1k",
        "Adjectives",
        PER_1K,
        _WORDS,
        syntax=True,
        about="Adjectives.",
    ),
    Metric("syntax", "adverbs_per_1k", "Adverbs", PER_1K, _WORDS, syntax=True, about="Adverbs."),
    Metric(
        "syntax",
        "modals_per_1k",
        "Modal verbs (can, should)",
        PER_1K,
        _WORDS,
        syntax=True,
        about="Modal verbs such as can, should and might.",
    ),
    Metric(
        "syntax",
        "nominalizations_per_1k",
        "Nominalizations (-tion, -ment)",
        PER_1K,
        _WORDS,
        syntax=True,
        about="Nouns ending in -tion, -ment, -ness, -ity and similar.",
    ),
    Metric(
        "sentence_openers",
        "opens_pronoun_pct",
        "Pronoun (I, It, You)",
        PCT,
        _SENTENCES,
        syntax=True,
        about="Share of (3+ word) sentences that open with a pronoun.",
    ),
    _opener(
        "opens_determiner_pct",
        "Determiner (The, A, This)",
        "Share of (3+ word) sentences that open with a determiner.",
    ),
    _opener(
        "opens_adverb_or_preposition_pct",
        "Adverb or preposition (Still, In)",
        "Share of (3+ word) sentences that open with an adverb or preposition.",
    ),
    _opener(
        "opens_subordinator_pct",
        "Subordinator (If, When, Because)",
        "Share of (3+ word) sentences that open with a subordinating conjunction.",
    ),
    _opener(
        "opens_conjunction_pct",
        "Conjunction (And, But)",
        "Share of (3+ word) sentences that open with a coordinating conjunction.",
    ),
    _opener("opens_noun_pct", "Noun", "Share of (3+ word) sentences that open with a noun."),
    _opener("opens_verb_pct", "Verb", "Share of (3+ word) sentences that open with a verb."),
    *(
        _rate("function_words", f"fw_{word}_per_1k", f'"{word}"', f'The word "{word}".')
        for word in FUNCTION_WORDS
    ),
)
METRIC_BY_NAME: dict[str, Metric] = {metric.name: metric for metric in METRICS}

# The default view: a short, stable set that covers each level without the long tail, in
# this order and under these headings.
KEY_VIEW: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Sentences",
        (
            ("sentence_shape", "sentence_words_mean"),
            ("sentence_shape", "short_sentences_pct"),
            ("sentence_shape", "long_sentences_pct"),
            ("rhythm", "sentence_length_cv"),
            ("sentence_shape", "paragraph_words_mean"),
            ("sentence_shape", "one_sentence_paragraphs_pct"),
        ),
    ),
    ("Vocabulary", (("vocabulary", "long_words_pct"), ("vocabulary", "mattr_100"))),
    (
        "Punctuation",
        (
            ("punctuation", "commas_per_1k"),
            ("punctuation", "em_dashes_per_1k"),
            ("punctuation", "parentheses_per_1k"),
            ("punctuation", "colons_per_1k"),
        ),
    ),
    (
        "Voice",
        (
            ("voice", "contractions_per_1k"),
            ("voice", "first_singular_per_1k"),
            ("voice", "second_person_per_1k"),
            ("voice", "hedges_per_1k"),
            ("voice", "transition_openers_pct"),
            ("voice", "llm_markers_per_1k"),
        ),
    ),
    (
        "Syntax",
        (
            ("syntax", "clauses_per_sentence"),
            ("syntax", "passive_sentences_pct"),
            ("syntax", "nominalizations_per_1k"),
            ("sentence_openers", "opens_pronoun_pct"),
        ),
    ),
)

# Pooled pattern distributions compared with Jensen-Shannon divergence.
DISTRIBUTION_LABELS: dict[str, str] = {
    "masked_bigram": "Function-word transitions",
    "pos_trigram": "Grammar patterns (POS trigrams)",
    "char_trigram": "Character trigrams",
}


def title(group: str) -> str:
    """A group's display title; an unregistered group (from another version) gets its name."""
    known = GROUP_BY_NAME.get(group)
    return known.title if known else group.replace("_", " ").capitalize()


def label(name: str) -> tuple[str, Unit]:
    """A metric's display label and unit; an unregistered metric gets its name, plain."""
    known = METRIC_BY_NAME.get(name)
    return (known.label, known.unit) if known else (name.replace("_", " ").capitalize(), PLAIN)


def resolution(name: str, counts: Mapping[str, float]) -> float:
    """The named metric's resolution floor (``Metric.resolution``), or 0 if unregistered."""
    known = METRIC_BY_NAME.get(name)
    return known.resolution(counts) if known else 0.0


def grouped(values: Values, *, syntax: bool) -> Grouped:
    """Nest one computation's values (surface or syntax) by group, in registry order.

    Raises ValueError when the computed names and the registered ones differ, so a metric
    cannot be computed without a definition or defined without a computation.
    """
    expected = [metric for metric in METRICS if metric.syntax == syntax]
    missing = [metric.name for metric in expected if metric.name not in values]
    extra = sorted(set(values) - {metric.name for metric in expected})
    if missing or extra:
        raise ValueError(f"metrics out of sync with the registry: {missing=} {extra=}")
    nested: Grouped = {}
    for metric in expected:
        nested.setdefault(metric.group, {})[metric.name] = values[metric.name]
    return nested


class Description(NamedTuple):
    group: str
    label: str
    unit: str
    about: str


def describe(*, syntax: bool = True) -> list[Description]:
    """One row per metric (group title, label, unit, meaning) in display order, for listing
    what styleprofile measures. ``syntax=False`` leaves out the parser-based metrics."""
    return [
        Description(title(metric.group), metric.label, metric.unit.value, metric.about)
        for metric in METRICS
        if syntax or not metric.syntax
    ]
