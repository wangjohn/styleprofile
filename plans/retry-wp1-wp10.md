# Plan: a second attempt at WP-1 and WP-10

The [onboarding plan](onboarding.md) shipped twelve of its fourteen work packages in 0.2.0.
WP-1 (a z-score cap) and WP-10 (bundled generic contrast drafts) failed their acceptance
checks and were deferred ([#30](https://github.com/wangjohn/styleprofile/pull/30),
[#37](https://github.com/wangjohn/styleprofile/pull/37)). This plan replaces them with three
work packages built on what those attempts showed. The original plan's
[Rules for every work package](onboarding.md#rules-for-every-work-package) apply to all
three, unchanged.

| WP | What | Model | Starts |
|---|---|---|---|
| **WP-1b** | Typography never changes a verdict (replaces the global cap) | Sonnet 5.5 | now |
| **WP-10a** | A held-out set of LLM drafts, written by a different model, for testing WP-10b | Haiku 4.5 | now |
| **WP-10b** | Generic contrast, second attempt: a varied corpus, a fair anchor, a test written down in advance | Sonnet 5.5 | after WP-10a merges |

Launch prompt, with `WP-N` filled in:

> Implement work package WP-N of `plans/retry-wp1-wp10.md` in wangjohn/styleprofile. Read it
> from `main`; if it isn't there yet, read it with
> `git fetch origin claude/style-profile-repo-review-i441nw && git show origin/claude/style-profile-repo-review-i441nw:plans/retry-wp1-wp10.md`.
> Follow the rules it points to and implement WP-N only. When it's done and validated, open
> a draft PR against `main`, then keep driving it until CI is green.

## What the first attempts showed

**WP-1 (#30).**
- The motivating case is fixed already: WP-5 leaves out paragraph metrics for text with no
  paragraph breaks, so the writer's essays with their blank lines removed read "close".
- The cap at |z| = 5 worked on that case, but it broke three empirical checks that were
  already in place:
  - **Short LLM text was caught less often.** Delta flagged 19 of 20 150-word LLM pieces
    before the cap and 5 or 6 of 20 after it; the checks require 80%.
  - **A two-document reference became too narrow.** Its range at 300 words shrank, so
    23% of the writer's own held-out pieces read "not close"; the check allows 10%.
- An extreme z is often the signal itself (em dashes in a writer who never uses them), so
  capping every metric throws signal away.

**WP-10 (#37).**
- The 24 drafts were all 447–492 words long, so **length alone** told them apart from the
  writer (AUC 1.0 on the samples).
- They shared framing and transitions. The "LLM drafts typically score" anchor is computed
  leaving out one draft at a time, so near-twins of the left-out draft were still in
  training and the anchor came out high.
- LLM text from outside the set scored 4–7, below halfway to that anchor: **0 of 5** sample
  LLM drafts and **0 of 40** synthetic ones reached "leans LLM".
- The writer side held up: 38 of 40 synthetic held-out documents and 6 of 7 held-out essays
  read "like the reference".

**New evidence on 0.2.0.** Typographic quotes flip the writer's own essays. Converting their
straight quotes and apostrophes to curly ones, as Word, Google Docs and most blog editors do,
moves all seven from "close" to "somewhat different" (5) or "clearly different" (2), Delta
1.22–1.46. The cause is one metric, `curly_apostrophe_pct`, which the reference never varies
on (0%) and the converted text has at 100%. Straightening the quotes in the sample LLM drafts
changes none of their scores, so the metric adds false alarms and no signal.

---

## WP-1b: Typography never changes a verdict (M)

**Replaces WP-1.** Don't add a global or per-metric z cap, and don't change calibration
math; #30 showed both cost real detection. Make the metrics blind to how text was typed or
converted, and prove it with a set of transforms.

**Reproduce.**
```bash
uv run styleprofile build examples/writer --contrast examples/llm-drafts -o profiles/w.json
mkdir -p /tmp/smart && for f in examples/writer/*.md; do
  python3 -I -c 'import re,sys
t=open(sys.argv[1],encoding="utf-8").read()
t=re.sub(r"(\w)\x27(\w)", "\\1\u2019\\2", t)
t=re.sub(r"(^|\s)\x27", "\\1\u2018", t); t=t.replace("\x27","\u2019")
t=re.sub(r"(^|\s)\"", "\\1\u201c", t); t=t.replace("\"","\u201d")
sys.stdout.write(t)' "$f" > /tmp/smart/$(basename "$f"); done
uv run styleprofile score -q /tmp/smart profiles/w.json   # 0.2.0: somewhat or clearly different, Delta 1.22–1.46
```

**Do.**
1. **Write the invariance suite first**, as `tests/test_invariance.py`, plus a script in
   `bench/` that prints a table of the same checks for the PR. Apply each transform to the
   texts listed below and compare against the untransformed text.
   - **Transforms:**
     1. straight to typographic quotes and apostrophes (as in Reproduce);
     2. typographic to straight;
     3. CRLF line endings;
     4. hard wrap at 72 columns within paragraphs;
     5. `—` to `--`, and `--` to `—`;
     6. `...` to `…`, and back;
     7. two spaces after sentence ends, and a non-breaking space after them;
     8. one newline instead of a blank line between paragraphs;
     9. blank lines removed (WP-5's case, as a regression check);
     10. Markdown rendered to simple HTML (`<p>`, `<em>`, `<h1>`) and read back with
         `--input-format html`;
     11. trailing spaces and tabs.
   - **Texts and bars:**
     - **Writer side:** every essay in `examples/writer`, and the 40 held-out documents
       (every fifth, sorted) of the synthetic `medium` corpus (`bench/gen.py --corpus
       medium`), each against a profile built without it. Its Delta verdict and its
       LLM-likeness verdict must not change, and |ΔDelta| must be at most 0.25.
     - **LLM side:** every draft in `examples/llm-drafts`, scored against the
       `examples/writer` profile. Its verdicts must not drop a band, so typography
       normalization can't erase real signal.
   - Run the suite with and without spaCy. Record which transforms fail on `main` before
     any fix.
2. **Make metrics typography-neutral:**
   - count quotations, apostrophes, dashes and ellipses the same whatever glyph they use
     (for example, the quotations metric counts curly opening quotes but straight *pairs*;
     make both count pairs);
   - read typographic quotes in every lexicon and phrase match (most already do; check all
     of them).
3. **Stop scoring metrics that only record encoding.** `curly_apostrophe_pct` is one; the
   suite will show any others. Remove them from Delta, area Deltas and LLM-likeness. Either
   list them under `--all` marked "not scored", or remove them; choose, and say why. Check on
   `examples/` and the `medium` corpus that removing them doesn't lower the contrast AUC by
   more than 0.01, and report the numbers.
4. If a transform still fails after steps 2 and 3, don't reach for a cap. Stop, and report
   which metric and transform in the PR.
5. Document it in `docs/method.md` under a new section, "What styleprofile ignores", listing
   the transforms the suite guarantees. Add a CHANGELOG entry under Fixed.
6. This changes measurement, so follow rule 6: report version 9.0, unless another package
   has already bumped it for this release.

**Done when.**
- The invariance suite passes.
- **Every existing test passes unchanged.** In particular, these three must pass with their
  limits as they are:
  - `tests/test_calibration.py::test_texts_shorter_than_a_window_get_a_wider_range_without_shorter_lengths`
  - `tests/test_calibration.py::test_llm_drafts_cut_to_150_words_are_still_flagged` (both
    cutters)
- The bench gate passes.
- The PR shows the Reproduce output before and after, and the suite's table on `main` and
  on the branch.

---

## WP-10a: A held-out set of LLM drafts (S)

**Why.** WP-10b needs LLM text it has never seen to test against. The sample LLM drafts are
5 drafts from one author, and the synthetic corpora are remixed from them. The test drafts
must come from a **different model** than the one writing the bundled corpus, and be written
**before** it.

**Do.** Run this in a session using **Haiku 4.5**.
1. **Don't look at the project first.** Before writing, don't open `examples/`, `bench/`,
   `src/` or any of styleprofile's docs; read only this section. The drafts should be what
   an assistant writes when someone asks, uninfluenced by what the tool measures.
2. **Write 16 drafts, one per file,** as `bench/eval/llm-heldout/NN-slug.md`. Answer each
   prompt below as a fresh request, in your natural default style for that request. Don't
   try to sound human or to sound like an AI; don't edit the drafts afterwards. Aim for the
   requested length.
   - **Topics close to the sample essays (a rural New England narrator):**
     1. "Write a personal essay about an old hand-drawn map you found in a relative's attic." (~600 words)
     2. "Write a blog post about why stone walls and fence lines matter in old farm country." (~800 words)
     3. "Write a short essay about mud season in Vermont." (~350 words)
     4. "Write a newsletter piece about heating a house with a woodstove all winter." (~700 words)
     5. "Write a reflective essay about a small-town public library." (~900 words)
     6. "Write a personal essay about walking in the rain." (~300 words)
     7. "Write a blog post about learning to sharpen hand tools properly." (~1,000 words)
     8. "Write an essay about the village hardware store and why it is worth keeping." (~500 words)
   - **General topics:**
     9. "Write a blog post about what I learned from a year of working remotely." (~800 words)
     10. "Write a short personal essay about baking sourdough for the first time." (~400 words)
     11. "Write an article giving advice on buying a used car." (~900 words)
     12. "Write a newsletter intro about our neighborhood park's cleanup day." (~250 words)
     13. "Write an essay about starting piano lessons as an adult." (~600 words)
     14. "Write a blog post with practical tips for sticking to a monthly budget." (~700 words)
     15. "Write a travel piece about a long train journey." (~1,000 words)
     16. "Write a reflective essay about what a vegetable garden teaches you." (~500 words)
3. **Write `bench/eval/llm-heldout/README.md`** recording:
   - the model, the date, and each file's exact prompt;
   - that the text is original and MIT licensed;
   - that the set is for **evaluation only**: never train, tune or choose a bundled
     contrast corpus with it.
4. **Keep it out of the package.** Make sure the set is not in the wheel, and add a test
   that checks it isn't. (The sdist includes `bench/`; that's fine.)

**Done when.** The 16 drafts and the README are merged. The PR body lists each prompt, the
draft's word count, and confirms rule 1 was kept.

---

## WP-10b: Generic contrast, second attempt (L)

Start after WP-10a merges. Reuse #37's plumbing (`build --generic-contrast`, library
`generic_contrast=True`, combining with `--contrast`, the "generic LLM drafts" label, and
the wheel resources) from branch `codex/onboarding-wp-10`. Replace its corpus and its
anchor.

**Write the design down first.** Before measuring anything on the test set, commit a
section of `docs/method.md` that states:
- the corpus design (step 1);
- the anchor (step 2);
- the data sets (step 3);
- the bars ("Done when").

The PR's commit order must show that this came first. Evaluate on the test set **once**,
after the corpus and the code are frozen in a commit, and report whatever it shows.

1. **Corpus: 60 original drafts, under 400 KB,** in
   `src/styleprofile/data/generic-contrast/<register>/`, with a README giving provenance and
   license.
   - **Length:** 15 drafts each of 150–300, 300–600, 600–1,000 and 1,000–1,500 words.
     Before any likeness evaluation, check that length alone separates the corpus from
     `examples/writer` and from the `medium` writer documents with an AUC of at most 0.70.
     If it doesn't, fix the lengths only.
   - **Six registers of 10 drafts each:**
     1. a headed or bulleted explainer;
     2. a plain-prose blog post;
     3. a personal essay written on request;
     4. a newsletter or email;
     5. an opinion piece;
     6. a "make it sound human" rewrite style: no em dashes, varied sentence lengths, no
        stock phrases.
   - **Topics:** no two drafts share a topic, and at least half are everyday or personal
     rather than technical.
   - **Variety:** no opening or closing formula appears more than twice in the corpus.
     Report the largest number of drafts that share any 8-word sequence.
   - Don't read `bench/eval/llm-heldout/` while writing.
2. **A fair anchor.** For generic contrast, compute the contrast's calibration (its median,
   its minimum and the AUC) with folds that leave out **a whole register** at a time, not
   one draft. The anchor then reflects LLM text in a style the weights haven't seen.
   - Implement it as contrast groups: a generic draft's group is its register folder. For
     `--contrast` drafts the user supplies, the group stays the document, and profiles built
     without generic contrast must be **byte-identical** to `main`'s. Test that.
3. **Data sets, fixed in advance:**
   - **Dev**, which you may iterate on: `examples/llm-drafts`, and the `medium` corpus's
     contrast documents.
   - **Test**, used once: `bench/eval/llm-heldout/`, the 16 drafts from WP-10a.
   - **Writer held-out:** each `examples/writer` essay held out in turn, and the `medium`
     writer documents that are every fifth when sorted, against profiles built without
     them.

**Done when**, with these bars written down before the test run:
- **Dev:** all 5 drafts in `examples/llm-drafts` read "leans LLM" or beyond.
- **Test:** at least 13 of the 16 test drafts read "leans LLM" or beyond. The AUC of test
  drafts against writer held-out documents is at least 0.90, and length alone gives at most
  0.70.
- **Writer:** at least 95% of the held-out writer documents (both sets together) read "like
  the reference".
- `make check`, the bench gate and `make dist-check` pass. The report gets a minor version
  for the new optional fields (rule 6), unless measurement changed.

**If the test fails**, keep the PR as a draft with the full results. Don't change the corpus,
the anchor or the bars after seeing the test set. List the product alternatives for the
maintainer to decide, for example showing generic-contrast likeness as a number marked
"indicative", without verdict words. Don't implement one.
