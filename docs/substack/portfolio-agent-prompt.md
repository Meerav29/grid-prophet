# Starter prompt for the agent in the portfolio repo

Fill in the two `<...>` values near the top (after you've made the publication and published the post), then paste everything inside the box into a Claude session opened in `C:\Users\meera\Github-Projects\MyPortfolio`.

```
You are working in my portfolio site repo (Next.js, deployed on Vercel at meeravshah.vercel.app). I now publish long posts on Substack, and I want them to show up in my site's /sidequests listing exactly like my other posts, except that clicking one opens the Substack post instead of a page on my site.

Inputs (I will fill these in):
- Substack publication URL: <https://PUBLICATION.substack.com>
- First post to feature: <https://PUBLICATION.substack.com/p/POST-SLUG>
- The post's source and images are in C:\Users\meera\Github-Projects\grid-prophet\docs\ (blog-grid-prophet-overhaul.md and blog-assets\). Read them from there; do not copy the whole folder into this repo.

First, orient yourself and report back before changing anything:
1. Check which repo is the deployed one. There is MyPortfolio and a nested MyPortfolio-2\MyPortfolio. Use the one whose git origin is Meerav29/MyPortfolio unless you find evidence otherwise, and tell me which you chose and why.
2. My MyPortfolio checkout has uncommitted work on branch animation-rework (components/Hero.tsx, components/PlanetCanvas.tsx, package-lock.json) and an untracked draft post (content/sidequests/2026-08-18-re-sike-beta-broken-in-prod.mdx, status "draft"). Do not stash, reset, commit, delete or publish any of that. Do all your work in a new git worktree branched from origin/main.
3. Read README_SIDEQUESTS.md, lib/sidequests.ts (zod frontmatter schema), app/sidequests/page.tsx, app/sidequests/[slug], components/sidequests/SidequestCard.tsx, and the /sidequests.xml feed route. Read any CLAUDE.md or AGENTS.md and follow it.

Then do this in one small PR (never push to main):
A. Add an optional externalUrl to the Sidequest type and frontmatter schema. When it is set, the card links to that URL in a new tab (rel="noopener noreferrer") with a small "on Substack" marker, the internal /sidequests/<slug> page is not generated for it, and /sidequests.xml and any sitemap use the external URL. Posts without externalUrl must behave exactly as today.
B. Make new Substack posts appear automatically: fetch <PUBLICATION_URL>/feed at build time (revalidate hourly if the site supports it), map each item to a Sidequest (title, publishedAt, excerpt from the description, thumbnail from the feed image if present else a default, tags from feed categories else ["Substack"], status "published", externalUrl from the item link) and merge it into the listing, de-duplicating on URL. If the feed fetch fails, the site must still build and show the local posts. No secrets are needed and none should be added.
C. If the feed doesn't yet contain the first post, add one MDX stub for it using externalUrl, with title "I rebuilt my F1 model, let routines build out parts of it, and graded it on Singapore", an excerpt drawn from the post's TL;DR, tags ["F1", "AI", "Project"], and the publish date from the Substack page. Copy ONE image from the blog assets (08_singapore_podium.png) to public/images/sidequests/ as its thumbnail.
D. Run the repo's lint, type check and production build, and fix what you broke. Open the PR with a description that lists what changed, how I can review it in 10 minutes, and what you did not do. Show screenshots of the listing page.

Verify in a browser (use the browser tools available to you):
- Open the Substack post URL and confirm it loads without logging in, the 8 images render, and the footnotes show. Report anything broken rather than fixing it.
- Open the local dev server's /sidequests and confirm the Substack card appears in the right order, opens Substack in a new tab, and that an existing post still opens normally.
- Repeat on the Vercel preview URL for the PR once it exists.

Hard rules:
- Never enter, request or store any password, token or credential. If a page asks me to log in, stop and tell me.
- Do not publish, schedule, send or edit anything on Substack, and do not change any account or site settings. Reading public pages is fine.
- Do not merge the PR. I will.
- If anything above is ambiguous or doesn't match what you find, ask me before guessing.
```

## Why it's built this way

- **The RSS feed does the repeat work.** Every Substack publication has a feed at `/feed`, so once part B is merged, future posts appear on the site with no commits and no prompts.
- **The stub in step C is a fallback** for the gap between publishing and the feed updating, and for anything the feed describes poorly.
- **It works in its own worktree** so your in-progress animation changes and the draft post stay untouched.
- **It can't publish or log in for you.** Publishing to Substack stays your action.
