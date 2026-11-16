from styleprofile.metrics import UNSCORED_GROUPS
from styleprofile.surface import surface_metrics, symbol_bullets


def test_symbol_bullets_count_lines_led_by_emoji_arrows_or_keycaps():
    text = (
        "Three things I learned:\n\n"
        "\u2705 Ship early\n"
        "\U0001f539 Talk to users\n"
        "\u21b3 even the angry ones\n"
        "1\ufe0f\u20e3 Keep going\n"
        "- a dash item\n"
        "2. a numbered item\n"
        "A sentence with an emoji \U0001f680 inside.\n"
    )
    assert symbol_bullets(text) == 4


def test_symbol_bullets_skip_code():
    assert symbol_bullets("```\n\u2705 not prose\n```\n") == 0


def test_social_rates_are_reported_but_not_scored():
    text = "Big news \U0001f680\n\n\u2705 One\n\u2705 Two\n\nThanks all."
    social = surface_metrics(text)["social"]
    words = 6
    assert social["symbol_bullets_per_1k"] == 2 * 1000 / words
    assert social["emoji_per_1k"] == 3 * 1000 / words
    assert "social" in UNSCORED_GROUPS
    assert surface_metrics("")["social"]["emoji_per_1k"] is None
