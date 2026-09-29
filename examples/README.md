# Sample corpus

A small corpus for trying styleprofile end to end. Run `make demo` from the repository root:
it scores `draft.md`, which slips into the LLM register in two paragraphs, first against the
seven essays, then against a larger synthetic corpus remixed from them, which finds the two
paragraphs under "Where it drifts" (experimental, shown with `--by-paragraph`).

All text here is original, written for this repository, and released under the repository's
MIT license. No third-party or public-domain text is included.

- `writer/`: seven short personal essays (about 600 to 700 words each) by one invented
  narrator, a retired country surveyor in northern New England. The voice is kept consistent
  on purpose: first person, plain words, short declaratives mixed with longer sentences,
  semicolons and parentheses, no em dashes, no headings beyond the title, no lists.
- `llm-drafts/`: five drafts on the same topics as five of the essays, written deliberately in
  a generic modern LLM register: section headings, bulleted and numbered lists, bold
  lead-ins, em dashes, "Conclusion" sections and abstract, upbeat vocabulary. They are
  roughly length-matched to the essays so the contrast does not simply learn length.
- `draft.md`: a new essay on a different topic (the village hardware store), mostly in the
  narrator's voice but with two paragraphs that slip into the LLM register, so the comparison
  has something to find: line 9 ("But the store is more than a place to buy things — it's a
  vital part of the community's fabric...") and line 15 ("Ultimately, the future of the
  village hardware store depends on all of us..."). Neither is in `llm-drafts/`.

## What `make demo` does

1. `styleprofile build` makes `profiles/demo-essays.json` (ignored by version control) from
   `writer/`, contrasted with `llm-drafts/`, and `styleprofile score --by-paragraph` scores
   `draft.md` against it. The whole draft reads "close", and "Where it drifts" says the reference is
   too small to set paragraph thresholds: seven essays (about 4,500 words, 55 paragraphs)
   are far fewer than the 10 documents and about 200 paragraphs they need.
2. `python bench/gen.py --corpus demo --out profiles` writes `profiles/demo/`: 40 documents
   of about 700 words in the narrator's voice, made by recombining the seven essays'
   paragraphs and sentences, and 10 contrast documents made the same way from `llm-drafts/`
   (seeded, so always the same text).
3. `styleprofile build` makes `profiles/demo-writer.json` from them, and
   `styleprofile score --by-paragraph` scores `draft.md` against it: "Where it drifts"
   points to lines 9 and 15, and `--by-paragraph` lists every paragraph.

The generated corpus stands in for a real writer's larger archive, **for illustration only**.
It is optimistic: its documents share paragraphs and sentences with each other, so they vary
less than a real writer's topics do, and its thresholds are tighter than yours will be.
Treat the demo's numbers as an illustration of the output, not as calibrated scores.
