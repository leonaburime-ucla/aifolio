# AIfolio Vue (Nuxt 3)

## Vercel Deployment Settings

| Field            | Value                          |
|------------------|--------------------------------|
| Root Directory   | `web/vue`                      |
| Framework Preset | Nuxt.js                        |
| Install Command  | `cd .. && npm install --force` |
| Build Command    | `nuxt build`                   |
| Output Directory | _(blank)_                      |

### Why `cd .. && npm install --force`

- `cd ..` runs install from `web/` where the workspace `package.json` lives. This resolves `file:../packages/*` dependencies (contracts, frontend-core) correctly.
- `--force` is needed because `vue-echarts@7` has a peer dep on `echarts@^5` but we use `echarts@^6` (compatible in practice).
- The workspace `package.json` also lists linux-x64 native bindings (oxc-parser, oxc-transform, oxc-minify, rollup, esbuild, unrs-resolver) as explicit optional deps to work around an npm bug that skips platform-specific optional deps in workspaces.

### Environment Variables

| Variable                    | Description              |
|-----------------------------|--------------------------|
| `NUXT_PUBLIC_API_BASE_URL`  | Backend API base URL     |
