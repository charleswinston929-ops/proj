# Deploying

## 1. Dashboard on Netlify (static, ~2 minutes)
1. Push this repository to GitHub.
2. Netlify: **Add new site → Import an existing project → GitHub →** pick the repo. `netlify.toml` already sets the publish directory (`apps/dashboard`) and security headers; leave the build settings as detected.
3. Deploy. The site is live at `https://<name>.netlify.app`. (Site configuration → Domain management to add your own domain.)

## 2. Keep it fresh (GitHub Actions, free)
`.github/workflows/refresh.yml` runs daily (06:00 UTC): re-pulls the current season, pulls availability (FPL API, if reachable), re-forecasts, rebuilds `apps/dashboard/index.html`, commits it; Netlify redeploys on that commit.
It is an **untested scaffold**: trigger it once via *Actions → daily-refresh → Run workflow* and read the log. Repository Settings → Actions → General must allow workflow write permissions.

## 3. First live check (do this once on your machine)
```
make setup
cd services/prediction-engine && python -m app.pipeline.availability_fetch   # verifies the FPL endpoint and team-name resolution
cd ../.. && make forecast ui && open apps/dashboard/index.html
```
If it reports `unresolved teams`, add the missing names to `MANUAL_ALIASES` in `app/entities.py`.

## 4. Optional: the API
The API (`make api`, or `docker compose up`) is not needed by the static dashboard. To host it, deploy `services/prediction-engine/Dockerfile` to any container host (Render, Fly.io, Railway, Cloud Run) with the `artifacts/` folder mounted or baked in. Set `RATE_LIMIT_PER_MIN`. Put it behind HTTPS; the app has no authentication, so keep it read-only and public-data only.

## Before you promote it
Statistical estimates only. No odds or edge claims are made. If you ever add odds, subscriptions or ads linked to betting, get legal advice (Germany: GlüStV), add an 18+ gate and responsible-gambling messaging, and publish an Impressum and privacy policy.
