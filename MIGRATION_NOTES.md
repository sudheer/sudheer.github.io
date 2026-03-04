# Hugo -> Astro migration notes

## Status

This branch now includes a real Astro project (`package.json`, `astro.config.mjs`, `src/`) and Astro CI/deploy workflows.

## PR preview URLs (before merge)

- Every PR build deploys to `gh-pages` under `preview/pr-<number>/`.
- URL format:
  - `https://<user>.github.io/<repo>/preview/pr-<number>/`
- The workflow comments preview status and URL on the PR.

## Production deploy

- Pushes to `main`/`master` build Astro and deploy `dist/` to `gh-pages` root.
- `clean-exclude: preview` preserves existing PR preview folders.

## Required GitHub Pages settings

In **Settings → Pages**:

1. Source: `Deploy from a branch`
2. Branch: `gh-pages`
3. Folder: `/(root)`

## Workflow files

- `.github/workflows/astro-ci.yml`
- `.github/workflows/astro-pages.yml`
- `.github/workflows/astro-preview.yml`
- `.github/workflows/bootstrap-pages-branch.yml`
