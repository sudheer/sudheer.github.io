# Hugo -> Astro migration notes

## Why no Astro build happened earlier

The previous workflows were gated by:

- `if: ${{ hashFiles('package-lock.json') != '' }}`

Since this Hugo repository does not yet have `package-lock.json`, jobs were skipped and no build ran.

## PR preview URLs (before merge)

This repository now follows the same preview approach as `marcus-astro`:

- Every PR build deploys to `gh-pages` under `preview/pr-<number>/`.
- URL format is:
  - `https://<user>.github.io/<repo>/preview/pr-<number>/`
- The workflow verifies the URL and comments it in the PR.

## Production deploy

- Pushes to `main`/`master` build with Astro and deploy `dist/` to `gh-pages` root.
- `clean-exclude: preview` keeps PR previews while updating production.

## Required GitHub settings

In **Settings → Pages**:

1. Source: `Deploy from a branch`
2. Branch: `gh-pages`
3. Folder: `/(root)`
