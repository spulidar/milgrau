# MILGRAU website — Lotus Docs + Hugo

**The sole maintained scientific and operational documentation source is docs/.** The new website does not introduce an additional editable copy. The build script reads the original Markdown and prepares temporary Hugo pages, adding navigation metadata, page titles and source-aware links. Original source documents and their GitHub relative links are unchanged.

## Local preview

Install Python 3.12 or newer, Go 1.21+, Git and Hugo Extended 0.140+ (GitHub Pages workflow uses Hugo 0.152.2).

~~~bash
python docs/_site/build.py
hugo --source docs/_site server --baseURL http://localhost:1313/milgrau/ --appendPort=false
~~~

Preview at http://localhost:1313/milgrau/. To produce the same static website as GitHub Pages:

~~~bash
python docs/_site/build.py
hugo --source docs/_site --gc --minify --baseURL https://spulidar.github.io/milgrau/
~~~

The generated directories `docs/_site/.hugo-content/` and `docs/_site/public/` are ignored by Git. Do not edit or commit their contents: make all scientific text changes in docs/ instead, then regenerate.

Lotus Docs is a pinned Hugo module in `docs/_site/hugo.toml`, with an explicitly pinned Bootstrap SCSS dependency. The site includes a landing page, grouped navigation, searchable pages, light/dark mode and KaTeX mathematical typesetting. The landing page branding lives in `docs/_site/data/landing.yaml` and has no duplicate scientific documentation.

## GitHub Pages deployment

The workflow `.github/workflows/docs-pages.yml` builds and checks the site when docs/ or the site configuration changes. On main it also uploads the generated site and attempts publication.

A repository administrator may need to configure the one-time GitHub setting:

1. Visit https://github.com/spulidar/milgrau/settings/pages
2. Under **Build and deployment**, set **Source** to **GitHub Actions**
3. Review the **Documentation — Lotus Docs / Hugo / GitHub Pages** workflow in Actions

After a successful deployment, the intended public URL is **https://spulidar.github.io/milgrau/**.

If Pages is disabled by repository or organization policy, the build may pass while deployment remains blocked. In that case do not interpret the intended URL as already online.

## Adding and organizing documents

- Write new documents only in docs/ and link to other Markdown files with ordinary GitHub-compatible relative links.
- The build script maps LIBIDS, LIPANCORA, LEBEAR, atmosphere, scientific validation and tools into a hierarchical Lotus Docs sidebar.
- New Markdown files that are not assigned to a category appear under **Reference** automatically, not silently omitted.
- Historical documentation appears in a distinct archived section; the current operating instructions remain authoritative.
- Edit the page-category mapping in `docs/_site/build.py` only when reorganizing website navigation, never to duplicate source files.

HTML, sidebar cards and internal Hugo URLs are generated automatically from the sources. The build fails when a relative source link points to a missing file, and its tests confirm that the source Markdown is never edited.

All website tooling (Hugo settings, theme modules, landing-page design, templates, generator and its tests) lives in `docs/_site/`. The GitHub Actions workflow is the only exception outside `docs/` because GitHub requires workflows under `.github/workflows/`. The root project README and scientific configuration remain in their conventional locations.

The original Licel observations and NetCDF products do **not** belong in this public website. If we publish scientific plots, they should be explicitly selected, non-sensitive figures placed in docs/ alongside context and interpretation.

## Current status

The GitHub Actions build checks representative pages and produces a downloadable HTML artifact. Publication depends on GitHub Pages account/repository settings and successful deployment. Continue scientific validation and documentation on the current tracker; site publication does not validate the underlying lidar physics.
