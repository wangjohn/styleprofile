"""String-first library calls and explicit cache ownership."""

from pathlib import Path

import pytest

import styleprofile as sp
from styleprofile import api

ROOT = Path(__file__).resolve().parent.parent


def essays() -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted((ROOT / "examples/writer").glob("*.md"))]


def test_strings_follow_the_named_text_pipeline(tmp_path: Path) -> None:
    texts = essays()
    contrast = [
        p.read_text(encoding="utf-8") for p in sorted((ROOT / "examples/llm-drafts").glob("*.md"))
    ]
    profile = sp.build_texts(iter(texts), contrast=iter(contrast), syntax=False)
    expected = sp.build(
        [sp.Text(t) for t in texts], contrast=[sp.Text(t) for t in contrast], syntax=False
    )
    assert profile.report == expected.report
    draft = (ROOT / "examples/draft.md").read_text(encoding="utf-8")
    assert profile.score_text(draft).report == profile.score(sp.Text(draft)).report
    assert (
        profile.score_text(iter([draft, texts[0]])).report
        == profile.score([sp.Text(draft), sp.Text(texts[0])]).report
    )
    assert [d.name for d in profile.score_text([draft, texts[0]]).documents] == ["text1", "text2"]
    path = tmp_path / "profile.json"
    profile.save(path)
    assert sp.load(path).report == sp.Profile.load(path).report


def test_build_keywords_and_score_settings_deprecation() -> None:
    profile = sp.build_texts(essays(), syntax=False, window_words=200, min_words=3)
    assert profile.settings.window_words == 200
    draft = essays()[0]
    inherited = profile.score_text(draft, syntax=False)
    assert inherited.report["settings"]["window_words"] == 200
    assert inherited.report["settings"]["min_words"] == 3
    with pytest.warns(DeprecationWarning, match="replaces all inherited settings"):
        replaced = profile.score(sp.Text(draft), sp.Settings(syntax=False))
    assert replaced.report["settings"]["window_words"] == sp.Settings().window_words
    named = sp.build([sp.Text(t) for t in essays()], sp.Settings(window_words=200), syntax=False)
    assert named.settings.window_words == 200 and named.settings.syntax is False
    with pytest.raises(TypeError, match="unknown setting"):
        sp.build_texts(essays(), typo=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="expected raw text strings"):
        profile.score_text([Path("draft.md")])  # type: ignore[list-item]


def test_library_cache_is_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_home = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    texts = essays()
    profile = sp.build_texts(texts, syntax=False)
    profile.score_text(texts[0])
    contrast = [
        sp.Text(p.read_text(encoding="utf-8"), p.name)
        for p in sorted((ROOT / "examples/llm-drafts").glob("*.md"))
    ]
    sp.evaluate(
        [sp.Text(t) for t in texts], contrast, {"same": contrast}, sp.Settings(syntax=False)
    )
    assert not cache_home.exists()
    sp.build_texts(texts, syntax=False, cache=True)
    assert cache_home.exists()


def test_missing_path_keeps_path_semantics(tmp_path: Path) -> None:
    missing = str(tmp_path / "my essay.md")
    with pytest.raises(sp.StyleProfileError, match=r"build_texts.*Profile.score_text"):
        sp.build(missing, syntax=False)
    with pytest.raises(sp.StyleProfileError, match="not found"):
        sp.build(str(tmp_path / "nope.md"), syntax=False)
    assert api.DEFAULTS.to_report() == sp.Settings().to_report()
