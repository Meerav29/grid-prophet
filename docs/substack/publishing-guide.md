# Publishing the Grid Prophet post on Substack

I couldn't open Substack's editor while writing this, so button names below come from Substack's help pages and may differ slightly on screen. Where I'm unsure I say so.

## What Substack will and won't take

- **No file or zip upload for a single post.** Substack's importer exists to move a whole publication over from another platform (Medium, WordPress, Ghost and similar), not to load one file. A single post is typed or pasted into the editor.
- **Markdown is not reliable.** Substack's help page says its post editor doesn't support Markdown, though that page is old, so it may have changed. Don't rely on pasting the `.md` file.
- **Images are uploaded one at a time** (drag into the draft, or use the image icon). Accepted formats are avif, gif, jpg, jpeg, png and webp. **SVG is not on the list**, so the routines infographic must be uploaded as the PNG (`06_routines_infographic.png`), which is already in `docs/blog-assets/`.
- **Alt text** is set per image: hover the image, open the three-dot menu in its corner, choose "Edit alt text".

Sources: [media in posts](https://support.substack.com/hc/en-us/articles/360037832971-How-do-I-embed-media-in-my-post-e-g-images-video-GIFs), [Markdown support](https://support.substack.com/hc/en-us/articles/360037463132-Do-you-support-Markdown), [importing](https://support.substack.com/hc/en-us/articles/360037830351-How-do-I-import-my-posts-from-another-platform-such-as-Mailchimp-WordPress-Medium-or-Ghost), [alt text](https://support.substack.com/hc/en-us/articles/48404618511508-How-do-I-add-alt-text-to-an-image-on-a-Substack-post), [RSS feed](https://support.substack.com/hc/en-us/articles/360038239391-Is-there-an-RSS-feed-for-my-publication).

## The easy path: paste the rendered text, then add images

1. Open `docs/substack/post-paste.html` in your browser. It is the post with the title removed, each image replaced by a numbered orange placeholder, and the footnote markers shown as superscript [1] to [4].
2. Select everything under "Copy from here" and copy.

## Steps in Substack

**A. One-time setup (you have an account, so check for a publication)**
1. Go to substack.com and, if you haven't yet, start a publication (writing happens inside a publication, not just an account).
2. Pick a name and a URL. The URL becomes `yourname.substack.com`, and every post lives at `yourname.substack.com/p/<post-slug>`. It's hard to change later, so choose it deliberately.
3. Write a one-line description. Skip payments and paid tiers for now.

**B. The post**
1. Start a new post. Put the title in the title field:
   `I rebuilt my F1 model, let routines build out parts of it, and graded it on Singapore`
2. The subtitle field is optional. A suggestion, to edit: `A backtest, a Singapore scorecard, and one thing that broke`.
3. Click into the body and paste. Check that headings, bullets, bold and the code-style words kept their formatting.
4. Replace each orange placeholder with the image (table below), then set its alt text. Delete the placeholder text.
5. Footnotes: Substack has a native footnote feature that shows the note on hover. I couldn't confirm the menu path, so look for "Footnote" in the editor's insert (+) menu or the text-selection toolbar. Replace each superscript [n] with a real footnote using the text in the table below. Fallback if you can't find it: leave the superscripts and add the four notes as a "Notes" list at the end.
6. Preview the post, and send yourself a test email if the option is offered, so you see what a reader sees on desktop and phone.
7. Publish. You have no subscribers yet, so there's no need to email anyone; look for an option to publish to the web only. Scheduling for a set time is available from the publish step if you'd rather post in the morning. (Exact labels unconfirmed.)
8. Copy the live URL. You'll need it for the X thread and the website.

## Images, in the order they appear

| # | Upload this file | Alt text |
|---|---|---|
| 1 | `06_routines_infographic.png` | Scheduled routines build, review and summarize work on a side branch, so main stays untouched until I review one pull request. |
| 2 | `01_baselines.png` | Pre-weekend model beats three baselines; adding the qualifying results cuts winner log loss from 1.80 to 1.26 (lower is better). |
| 3 | `02_paired_delta.png` | Conditioning on qualifying improved log loss in 75 of 88 rounds and worsened it in 13, with a few large misses. |
| 4 | `03_by_season.png` | Every season improved after conditioning on qualifying, with 2022 improving the least. |
| 5 | `04_warm_start.png` | All 88 warm-start refits finished under the 120-second target, averaging about 73 seconds; none needed the full-refit fallback. |
| 6 | `05_calibration.png` | Pre-weekend model is too timid at the front, too generous at the back: ~55% podium picks happened ~77% of the time; ~3% picks, ~0.5%. |
| 7 | `08_singapore_podium.png` | The model's top three podium odds, Verstappen, Leclerc and Antonelli, finished 1st to 3rd; Hamilton, at 51%, finished 6th. |
| 8 | `07_singapore_range.png` | 18 of 22 drivers landed inside the forecast range; Hadjar, Lindblad, Colapinto and Bortoleto did not. |

All eight are in `docs/blog-assets/`.

## Footnote text

1. Winner log loss is a score for how much probability the model gave the driver who actually won, where lower is better (for Singapore, ~69% on the winner scored 0.376).
2. These are the "routines" in the title: scheduled cloud agents, meaning AI jobs that start on a timer on someone else's machine and each do one job, like building a slice or reviewing a pull request.
3. Spearman is a score from -1 to 1 for how closely a predicted order matches the real one, where 1 is identical.
4. The Brier score is the average squared gap between a forecast probability and what happened (counted as 1 or 0), so lower is better and 0 would be perfect.

## Before you hit publish

- [ ] Title and subtitle read right
- [ ] All 8 images are in, in order, with alt text
- [ ] The infographic is the PNG, not the SVG
- [ ] Footnotes work (or the Notes fallback is in)
- [ ] No orange placeholder or blue instruction box is left in the body
- [ ] Preview looks right on a phone-width window
- [ ] The post still says "@memeerav" at the end (and the subscribe line makes sense now the publication exists)

## After publishing

- Your live post URL will look like `https://<publication>.substack.com/p/<slug>`.
- Your feed is at `https://<publication>.substack.com/feed`. That feed is what lets your website pick up new posts automatically (see `portfolio-agent-prompt.md`).
