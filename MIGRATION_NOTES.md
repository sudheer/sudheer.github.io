# Hugo -> Astro migration notes

## Status

This branch now includes an Astro monorepo with npm workspaces:

- `apps/blog` contains the deployable blog app, content, routes, Astro config, and TypeScript config.
- `packages/marcus-astro-theme` contains the reusable Marcus Astro theme components, layouts, and styles.
- The blog imports theme code through the local workspace package `@sudheer/marcus-astro-theme`.

## PR preview URLs (before merge)

- Every PR build deploys to `gh-pages` under `preview/pr-<number>/`.
- URL format:
  - `https://<user>.github.io/<repo>/preview/pr-<number>/`
- The workflow comments preview status and URL on the PR.

## Production deploy

- Pushes to `main`/`master` build `apps/blog` and deploy `apps/blog/dist/` to `gh-pages` root.
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
