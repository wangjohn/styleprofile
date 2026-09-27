# Sample corpus

A small corpus for trying styleprofile end to end. Run `make demo` from the repository root:
it builds a reference from `writer/`, contrasted with `llm-drafts/`, into
`profiles/demo-writer.json` (git-ignored), then scores `draft.md` against it.

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
  has something to find.

The corpus is far smaller than the README recommends (15 or more documents and 20,000 or
more words), so treat the demo's numbers as an illustration of the output, not as calibrated
scores.
