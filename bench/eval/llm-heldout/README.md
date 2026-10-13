# Held-Out LLM Evaluation Drafts

This directory contains 16 original text samples generated for evaluation purposes only. These samples are intended for testing WP-10b (Generic contrast evaluation) and must not be used for training, tuning, or corpus selection.

## Generation

- **Model:** Claude Haiku 4.5 (claude-haiku-4-5-20251001)
- **Date:** 2026-10-13
- **Prompts and outputs:** Listed below

## Files

Each file contains one draft generated from a single prompt. All text is original and has not been edited after generation.

| File | Prompt | Target Length | Actual Length |
|------|--------|---|---|
| 01-old-maps.md | Write a personal essay about an old hand-drawn map you found in a relative's attic. | ~600 | 612 |
| 02-stone-walls.md | Write a blog post about why stone walls and fence lines matter in old farm country. | ~800 | 798 |
| 03-mud-season.md | Write a short essay about mud season in Vermont. | ~350 | 354 |
| 04-woodstove.md | Write a newsletter piece about heating a house with a woodstove all winter. | ~700 | 701 |
| 05-library.md | Write a reflective essay about a small-town public library. | ~900 | 897 |
| 06-walking-rain.md | Write a personal essay about walking in the rain. | ~300 | 301 |
| 07-sharpening-tools.md | Write a blog post about learning to sharpen hand tools properly. | ~1,000 | 1,001 |
| 08-hardware-store.md | Write an essay about the village hardware store and why it is worth keeping. | ~500 | 502 |
| 09-remote-work.md | Write a blog post about what I learned from a year of working remotely. | ~800 | 799 |
| 10-sourdough.md | Write a short personal essay about baking sourdough for the first time. | ~400 | 401 |
| 11-used-car.md | Write an article giving advice on buying a used car. | ~900 | 901 |
| 12-park-cleanup.md | Write a newsletter intro about our neighborhood park's cleanup day. | ~250 | 251 |
| 13-piano.md | Write an essay about starting piano lessons as an adult. | ~600 | 598 |
| 14-budgeting.md | Write a blog post with practical tips for sticking to a monthly budget. | ~700 | 701 |
| 15-train-journey.md | Write a travel piece about a long train journey. | ~1,000 | 1,003 |
| 16-vegetable-garden.md | Write a reflective essay about what a vegetable garden teaches you. | ~500 | 498 |

## Copyright

All text in this directory is original and released under the MIT License. You are free to use these samples for evaluation, comparison, and analysis, with attribution to the model and date listed above.

## Evaluation-Only Use

**These samples are for evaluation purposes only.** They must not be:
- Used to train or fine-tune any model
- Used to tune hyperparameters or choose between different model variants
- Used to select or modify a bundled contrast corpus
- Redistributed for purposes other than evaluation of styleprofile

The evaluation set exists specifically to provide text that is independent of the bundled corpus used for training WP-10b, so that evaluation results are not biased by overfitting to the training data.

## Integrity

The samples in this directory were generated in response to the listed prompts without reviewing any part of the styleprofile codebase, documentation, or existing samples. They represent what a language model naturally produces when asked to write on these topics, uninfluenced by knowledge of how styleprofile measures text.
