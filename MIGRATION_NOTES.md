# Hugo -> Astro migration notes

## PR preview URLs (before merge)

This repository now includes a pull-request preview workflow.

- Every PR build deploys to `gh-pages` under `pr-<number>/`.
- URL format is:
  - `https://<user>.github.io/<repo>/pr-<number>/`
- A sticky PR comment is added/updated with the preview URL.

## What to configure after switching to Astro

1. Keep the default branch as `main`.
2. Ensure Astro builds into `dist/` (`npm run build`).
3. In **GitHub Pages settings**, serve from `gh-pages` branch root (`/`).
4. Delete the old Hugo workflow (`.github/workflows/hugo.yml`) once Astro is fully migrated.
