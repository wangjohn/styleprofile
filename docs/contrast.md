# Make contrast drafts from your own briefs

A contrast set answers a limited question: which habits distinguish this writer's texts
from these particular drafts? Drafts made from the writer's own briefs keep topics and
lengths similar, reducing the chance that likeness measures those differences instead of
writing habits. The bundled generic set is a convenience candidate with a failed held-out
acceptance check; it is not a validated substitute. See [Method](method.md#generic-contrast-candidate).

Choose independent texts from the same genre as the reference. Describe the subject,
audience, purpose and approximate length of each in a brief. Avoid pasting the writer's
actual sentences or asking a model to imitate the writer: that erases the difference the
comparison is meant to measure. Keep the finished draft separate from its prompt and
retain only the draft in the contrast folder.

Copy and adapt this prompt for each brief:

```text
Write an original [GENRE] for [AUDIENCE].
Subject and purpose: [DESCRIBE THE BRIEF WITHOUT COPYING THE WRITER'S TEXT].
Include these points: [POINTS THAT THE WRITER ALSO ADDRESSES].
Use approximately [N] words, matching the reference's typical length.
Write in your ordinary assistant voice. Do not imitate a particular writer.
Use headings or lists only if they fit this genre and audience.
Return only the draft, without a prefatory explanation or a discussion of the prompt.
```

Use several models or model versions when you can, and record which produced each draft
outside the text folder. Generate at least 10 drafts from different briefs for a first
comparison; aim for 20–30 or more when interpreting held-out ranges. The program needs at
least two independent writer documents to learn a contrast, but that minimum does not
make a reliable detector. More drafts cannot replace variety or a suitable reference.

Check actual lengths after generation. Match the distribution of writer lengths instead
of asking every draft for the same round number. If the writer uses both short posts and
long essays, include both kinds or build separate profiles. Keep the reference to one
genre where possible. Remove prompt echoes, refusal messages, and empty files, but do not
select drafts according to whether they receive your preferred verdict.

```sh
styleprofile build writer/ --contrast contrast/ -o writer.json
styleprofile score held-out-drafts/ writer.json
```

Keep a fresh set of writer texts and model drafts outside both training folders. Check
those before relying on the profile. Training AUC can be high while unseen model drafts
still receive weak verdicts. A likeness score describes resemblance to a contrast set; it
does not establish authorship or prove that a model wrote a text.
