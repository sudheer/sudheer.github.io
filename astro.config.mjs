import { defineConfig } from 'astro/config';

const repository = process.env.GITHUB_REPOSITORY?.split('/')[1];
const isUserSiteRepo = repository === 'sudheer.github.io';
const isGithubActions = process.env.GITHUB_ACTIONS === 'true';
const deployTarget = process.env.DEPLOY_TARGET;
const previewPath = process.env.DEPLOY_PREVIEW_PATH;

const repoBase = repository && !isUserSiteRepo ? `/${repository}` : '';

const normalizeBase = (value) => {
  const withLeadingSlash = value.startsWith('/') ? value : `/${value}`;
  return withLeadingSlash.endsWith('/') ? withLeadingSlash : `${withLeadingSlash}/`;
};

const base = !isGithubActions
  ? '/'
  : deployTarget === 'dev'
    ? normalizeBase(`${repoBase}/preview`)
    : deployTarget === 'pr' && previewPath
      ? normalizeBase(`${repoBase}/preview/${previewPath}`)
      : normalizeBase(repoBase || '/');

export default defineConfig({
  site: 'https://sudheer.github.io',
  base,
  markdown: {
    shikiConfig: {
      theme: 'github-dark'
    }
  }
});
