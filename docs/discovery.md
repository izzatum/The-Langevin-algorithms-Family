# Maintaining search and AI discoverability

The public documentation is at
<https://izzatum.github.io/The-Langevin-algorithms-Family/>.
GitHub Pages deployment was verified on 2026-10-17 after the owner enabled it.
The repository and documentation serve different needs: GitHub hosts the code,
while the site exposes explanations, equations, citations and executed tutorial
outputs as ordinary HTML. Neither publication nor a sitemap submission proves
that a search engine has indexed or ranked a page.

## Build and publish

```bash
python -m venv /tmp/lafa-docs-venv
/tmp/lafa-docs-venv/bin/python -m pip install -r requirements-docs.txt
/tmp/lafa-docs-venv/bin/python -m mkdocs build --strict
/tmp/lafa-docs-venv/bin/python scripts/check_docs.py
```

The build converts the six committed notebooks without executing them. Their
recorded CPU results and limitations remain visible. Update notebook outputs
through the normal validation workflow before publishing changed experiments.
Source-only benchmark files link back to GitHub. Plans and this maintainer guide
are excluded from the public site.

## Repository About settings

In the repository's **About** panel, use this concise description:

> Langevin algorithms in PyTorch: ULA, MALA, kinetic and adaptive samplers, SVGD
> and KSD for Bayesian inference and uncertainty quantification. Six executed
> tutorials and cited algorithm sources.

Suggested topics describe implemented capabilities:

```text
langevin-algorithms langevin-dynamics pytorch mcmc bayesian-inference
uncertainty-quantification inverse-problems monte-carlo metropolis-hastings
ula mala svgd stein-variational-inference kernel-stein-discrepancy
replica-exchange bayesian-neural-networks scientific-computing python
```

Set the About website to the public documentation URL. README, package and
citation links now point to the live site; its templates use the same canonical
URLs. Repository administration permission is required to change About settings.
The integration used for this update can push code and deploy enabled Pages,
but About edits still return HTTP 403; the owner must apply those settings.

In repository **Settings → Pages**, select **GitHub Actions** as the publishing
source. `.github/workflows/docs.yml` validates pull requests and deploys `main`.
Deployment needs Pages write permission and the `github-pages` environment's
approval requirements, if configured. A passing build is not a successful deploy;
inspect the deployment job and request the public URL after it completes.

Each public page has a distinct title and description, a self-referencing
canonical URL, sharing metadata and navigable headings. The home page describes
the software using `SoftwareSourceCode` JSON-LD. The build emits `sitemap.xml`
without invented modification dates. `scripts/check_docs.py` checks the sitemap,
metadata, local assets, fragments, citation anchors and software description.

Keep version, author, license and capability statements consistent across
`CITATION.cff`, `pyproject.toml`, `mkdocs.yml`, README and the documentation.
Structured data must describe the visible software accurately; it does not
promise a rich result or a ranking benefit.

## Verify ownership and request discovery

These steps require the owner's Google/Bing account. Do not invent verification
values or treat a GitHub login as a Search Console login.

1. In [Google Search Console](https://search.google.com/search-console), add the
   **URL-prefix** property
   `https://izzatum.github.io/The-Langevin-algorithms-Family/`. A Domain property
   would require control of `github.io` DNS, which a project owner does not have.
2. Use the HTML verification file supplied by Google. Put its exact filename and
   contents in `docs/`, publish, confirm the supplied URL returns that file, and
   click Verify. Keep the file after verification. An issued HTML meta tag is an
   alternative; add only that actual tag to `docs/overrides/main.html`.
3. Submit
   `https://izzatum.github.io/The-Langevin-algorithms-Family/sitemap.xml`.
   Inspect the home page and algorithm guide with URL Inspection, run the live
   test and request indexing where offered. Review crawl and indexing errors.
4. In [Bing Webmaster Tools](https://www.bing.com/webmasters/), import the verified
   Search Console property or use Bing's supplied HTML verification file/tag.
   Submit the same sitemap and inspect important URLs.
5. Review Google's generative-AI inclusion setting, including inherited
   exclusions, and Bing's AI Performance reports where available.

The project path cannot set origin-wide crawler rules: a robots file only applies
at `https://izzatum.github.io/robots.txt`. A file under this project's path would
not control crawling. No blocking robots directive or `noindex` is added to
public content. Google does not require `llms.txt` or special AI markup.
The site also supplies a public [IndexNow ownership file](https://izzatum.github.io/The-Langevin-algorithms-Family/29313ffdf7d22db2fc472f7fa744bf8a.txt). This project-scoped
key allows notification of the published sitemap URLs through IndexNow, which
participating engines including Bing support. The key is intentionally public;
it is not a private API credential. Keep the file available after submission.
An HTTP 200 or 202 response confirms receipt, not indexing or ranking, and does
not submit pages to Google. Search Console/Bing ownership verification and their
reporting accounts remain separate from this notification.

## Measure useful outcomes

Record a baseline after account verification. Monitor query impressions, clicks,
click-through rate, position and indexed pages in Google/Bing. Include the
actual queries people use: `Langevin algorithms`, `Langevin algorithms PyTorch`,
`ULA MALA Python`, and relevant inverse-problem terms. Separate brand queries
from general algorithm queries, and search exposure from AI citations.

Use Search Console's AI impressions and Bing's AI Performance citation reports
when available. Rankings vary by query, country, device and date; a manual
search is not a reproducible global ranking measurement. Changes can take days
to weeks to be recrawled, and indexing or ranking is never guaranteed.

Maintain original, useful method explanations and compare against appropriate
references. Link the project from relevant author profiles, papers, teaching
materials and research pages that you actually control. Avoid purchased links,
fabricated endorsements, duplicated keyword pages and unsupported performance
claims. Preserve unresolved GPU validation and finite-run limitations.

## Official guidance

- [Google: optimizing for AI experiences](https://developers.google.com/search/docs/fundamentals/ai-optimization-guide)
  — AEO/GEO share the foundations of useful, indexable search content.
- [Google: titles](https://developers.google.com/search/docs/appearance/title-link),
  [snippets](https://developers.google.com/search/docs/appearance/snippet) and
  [canonical URLs](https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls).
- [Google: sitemap guidance](https://developers.google.com/search/docs/crawling-indexing/sitemaps/overview)
  and [requesting recrawls](https://developers.google.com/search/docs/crawling-indexing/ask-google-to-recrawl).
- [Google: ownership verification](https://support.google.com/webmasters/answer/9008080?hl=en),
  [AI inclusion settings](https://support.google.com/webmasters/answer/16908024?hl=en)
  and [AI impressions](https://support.google.com/webmasters/answer/16984139?hl=en).
- [Google: FAQ rich results removed in May 2026](https://developers.google.com/search/updates#faq-deprecation).
  The site's questions help readers; they do not claim FAQ rich-result eligibility.
- [Bing: sitemaps in AI-powered search](https://blogs.bing.com/webmaster/2025/7/Keeping-Content-Discoverable-with-Sitemaps-in-AI-Powered-Search/)
  and [AI Performance reporting](https://blogs.bing.com/webmaster/2026/2/Introducing-AI-Performance-in-Bing-Webmaster-Tools-Public-Preview/).
- [Google: robots.txt location](https://developers.google.com/crawling/docs/robots-txt/create-robots-txt)
  and [IndexNow documentation](https://www.indexnow.org/documentation).
- [GitHub: configuring the Pages publishing source](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

Guidance checked against these primary sources on 2026-10-17. Recheck changing
search features before making claims about eligibility or reporting.
