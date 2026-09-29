# Provenance of the forecast date

This is the detailed record behind the README's section [What the 13 June date actually rests on](../README.md#what-the-13-june-date-actually-rests-on). It was moved out of the README on 29 September 2026 without changing any fact. What does not depend on any date, the inputs' unchanged bytes and the absence of any result after 11 June, stays in the README.

## What the 13 June date actually rests on

**There is no independent timestamp for 13 June.** The date rests on git metadata written on the author's own machine. Everything below should be read with that in mind.

**History was rewritten on 23 September 2026.** An internal notes file was removed from every commit and the repository was recreated on GitHub. No other file changed. Every commit hash changed as a result. The forecast commit was `32ffb4c` and is now `99a2305`. Every file in it other than the removed one is byte-identical to the original. Author and committer dates were preserved. The rewrite also means the GitHub creation date of this repository attests nothing about the forecast.

Before the rewrite the forecast commit reached GitHub only on 18 September, 90 days after its author date. So no server ever saw it before the tournament ended.

What remains is local evidence. All of it was produced on the same machine and all of it could be forged. Only the first item below can be checked from a clone of this repository. The other two are the author's account of objects that were never pushed: the pre-rewrite commits (`f296b05`, `eaddeb6`, `32ffb4c`) and the reflog are not in the published history, and a fresh clone has neither.

- The **author date** of `99a2305` is 13 June 2026 12:09:22 +02:00. Author dates are plain metadata and can be set to any value.
- *Not verifiable from a clone.* The author reports that a pre-rebase original of the commit, `f296b05` in the pre-rewrite history, is kept in a local copy of the original repository. Its **committer date matches its author date**. Its `data/predictions.csv` blob is byte-identical to the one in `99a2305`.
- *Not verifiable from a clone.* The author reports that the local **reflog** of that copy records that original at `HEAD@{2026-06-13 12:09:22 +0200}`. Git writes reflog timestamps at the moment of the operation, so they are not author-set. They are local only and can be edited.

## Why the committer date is 18 September

The committer date of `99a2305` is 18 September because the commit was replayed by a rebase that day. According to the author, the reflog of their local copy of the original repository records that rebase under the pre-rewrite hashes (`eaddeb6` and `32ffb4c`), and the pre-rebase original has both dates at 13 June and an identical `predictions.csv` blob. That is the author's explanation for the committer date. None of those objects are in the published history, so a reader cannot check it.
