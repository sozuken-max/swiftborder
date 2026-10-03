# Keep the Hosting client in git and deploy it by hand

## Status

Proposed

## Context

The graded demo is the static site at `https://swiftborder-92b45.web.app` (same bytes on `https://swiftborder-92b45.firebaseapp.com`). Travel times stay on the public JSON file ([0001](0001-firebase-client-api-calls.md)). This note decides where the page source lives and how a later push is deployed.

This repo has no `firebase.json`, `.firebaserc`, `package.json`, or Hosting directory. `.github/workflows/tests.yml` is the only workflow. `firebase` is not on PATH. `gcloud firebase` only offers the Test Lab group.

`C:\Users\vi_ci\Downloads` has `app.js` (42,084 bytes, written 2026-10-03 10:36 SGT) and no `index.html`, `style.css`, `firebase.json`, or `.firebaserc` beside it. The script is not a Firebase project. It has no Firebase SDK. On `DOMContentLoaded` it reads elements that the page must already define (`live-cam-grid`, `ai-detection-panel`, `lc-chart-area`, the transit cards, the toll calculator).

## Verified

Commands were read-only. The GCP account used for the read-only check had a different active gcloud project; every `gcloud` call below passed `--project=swiftborder`. Nothing was deployed and no API was enabled.

| Check | Command | Result |
| --- | --- | --- |
| Firebase CLI | `firebase --version` | Command not found. |
| gcloud Firebase group | `gcloud firebase --help` | Only `test` (Test Lab). `gcloud firebase projects list` is an invalid choice. |
| Enabled APIs | `gcloud services list --enabled --project=swiftborder --filter="config.name:firebase OR config.name:firebaserules OR config.name:firebasehosting"` | No rows. |
| Firebase project resource | `GET https://firebase.googleapis.com/v1beta1/projects/swiftborder` with `x-goog-user-project: swiftborder` | HTTP 403, `SERVICE_DISABLED` (Firebase Management API). A call without that header was billed to project number `681255809395` and is not evidence about `swiftborder`. |
| Hosting sites in `swiftborder` | `GET https://firebasehosting.googleapis.com/v1beta1/projects/swiftborder/sites` with the same header | HTTP 403, `SERVICE_DISABLED` (Firebase Hosting API). |
| Site id `swiftborder-92b45` | `GET https://firebasehosting.googleapis.com/v1beta1/sites/swiftborder-92b45` with the same header | HTTP 404, empty body. |
| Project id `swiftborder-92b45` | `gcloud projects describe swiftborder-92b45` | Permission denied, or the project is not visible to this account. |
| Live site | `HEAD` `/` on both Hosting hostnames, **2026-10-03 10:57 SGT** | HTTP 200, `Content-Type: text/html`, 15,989 bytes, `Last-Modified: Sat, 19 Sep 2026 05:54:38 GMT`, `Vary` includes `x-fh-requested-host`. |
| Live assets | `GET /app.js` and `GET /style.css` | HTTP 200. Same `Last-Modified` as `/`. `app.js` 42,084 bytes, sha256 `b99c25a5df956976808206626cd10af89e177fecb773158188539c35b057898e`, byte-identical to `Downloads\app.js`. `style.css` 44,139 bytes. The HTML also loads Google Fonts and one inline script. Those two files are not in Downloads. |
| Cloud Build | `gcloud builds triggers list --project=swiftborder` | One trigger: `76bbca35-c1b4-4836-9f34-d7adda53ea17`, GitHub `sozuken-max/swiftborder`, push to `^main$`, no `filename` (inline config). |
| Trigger scope | `gcloud builds triggers describe 76bbca35-c1b4-4836-9f34-d7adda53ea17 --project=swiftborder` | `includedFiles` is only `camdetect/main.py`, `camdetect/requirements.txt`, `camdetect/Dockerfile`, `camdetect/cloudbuild.yaml`, and `camdetect/*.{yaml,yml,json,toml}`. |
| GitHub Actions | `gh workflow list` in this repo | One workflow, `tests`, id `371431724`, active. `tests.yml` sets `permissions: contents: read` and has no deploy step and no secrets. |
| Public bucket | `gcloud storage buckets describe gs://swiftborder-public --project=swiftborder` | Location `ASIA-SOUTHEAST1`. CORS `GET` and `HEAD` from `https://swiftborder-92b45.web.app`, `https://swiftborder-92b45.firebaseapp.com`, and `http://localhost:5000`. `maxAgeSeconds` 3600. Response header `Content-Type`. |
| Public object | `gcloud storage objects describe gs://swiftborder-public/traffic-24h.json` | `application/json`, `Cache-Control: public, max-age=300`, size 18,480, `update_time` **2026-10-03T02:55:06Z** (10:55 SGT). |
| Fetcher | `gcloud run services describe gmap-woodlands-fetcher --project=swiftborder --region=asia-southeast1` | Revision `gmap-woodlands-fetcher-00004-msb`. Image `asia-southeast1-docker.pkg.dev/swiftborder/cloud-run-source-deploy/gmap-woodlands-fetcher`. Service account `1095552466513-compute@developer.gserviceaccount.com`. One env var, name `GOOGLE_MAPS_API_KEY`. The value is not recorded here. No bucket name in the env list. Source is not in git. |
| Scheduler | `gcloud scheduler jobs list --project=swiftborder --location=asia-southeast1` | One job, `Gmap-Woodlands`, `*/5 * * * *`, `Asia/Singapore`, `ENABLED`, HTTP `GET` to `https://gmap-woodlands-fetcher-1095552466513.asia-southeast1.run.app/`. |

The object timestamp sits on that 5-minute grid. This pass did not read the container or its logs, so the write to `traffic-24h.json` is still inferred from that cadence and from [0001](0001-firebase-client-api-calls.md).

## Decision

Put the Hosting client in this repo, and keep the deploy manual.

Do it in a later commit, when these files are in one directory (name it `hosting/`):

| File | Where it is today | This pass |
| --- | --- | --- |
| `app.js` | Live site and `Downloads\app.js`, same sha256 | Not copied |
| `index.html` | Live `/` only (15,989 bytes) | Not copied |
| `style.css` | Live `/style.css` only (44,139 bytes) | Not copied |
| `firebase.json` and `.firebaserc` | Not on disk and not served by Hosting | Not invented |

`app.js` alone is not a project `firebase deploy` can publish. A guessed `firebase.json` would invent the public directory and the site id.

No GitHub Action and no Cloud Build step deploys Hosting. The next push stays a manual upload from the machine that already has the Firebase login, after the four files above are in git. The exact command waits on `.firebaserc`: this pass could not read which Firebase project owns site `swiftborder-92b45`.

A Hosting workflow is a later decision. It would need a new credential that can deploy that site only. It must not be trigger `76bbca35`, and it must not receive `ROBOFLOW_API_KEY`. This repo has no such credential, and creating one is out of scope while the Hosting API on project `swiftborder` is disabled and the site's project is unconfirmed.

| | Include source and CI on merge | Include source, deploy by hand | Leave the client out |
| --- | --- | --- | --- |
| Reproducible next push | Needs a Hosting token this repo does not have, and a `firebase.json` nobody has | The static files can be saved from the live site; the Firebase project file still has to come from the last deploy machine | The next push stays an untracked upload of whatever is on that machine |
| Blast radius | A workflow secret that can publish the demo. Easy to over-grant into `swiftbackend` | Same as today: whoever is logged into the Firebase CLI | Same as today, and the demo source stays outside the code history |
| Fits this tree | `tests.yml` is `contents: read`. The only deploy trigger is `camdetect` to `swiftbackend` | A docs-only change does not match `includedFiles`, so it does not redeploy `swiftbackend` | Nothing to deploy from git |

## Consequences

- Chad collects `index.html`, `style.css`, and the `firebase.json` / `.firebaserc` from the machine that last deployed (the live HTML and CSS match the 19 Sep 2026 release; Downloads already matches live `app.js`). He does not commit `app.js` by itself.
- Until that commit, the demo source of truth for the page is the live Hosting release of 19 Sep 2026 05:54:38 GMT, not this repo.
- `tests.yml` stays tests-only. Trigger `76bbca35` stays limited to `camdetect` runtime files.
- CORS on `swiftborder-public` is unchanged. `TRAFFIC_API` and `BACKEND_URL` are unchanged.
- This pass did not deploy, did not enable `firebase.googleapis.com` or `firebasehosting.googleapis.com`, and did not copy the client into git.

## What this pass did not verify

- A Firebase release record (the Hosting API is disabled on project `swiftborder`, so there is no release list). The 19 Sep 2026 timestamp is the HTTP `Last-Modified` on the three files.
- Which GCP project owns site `swiftborder-92b45`.
- Scheduler `lastAttemptTime`, and the fetcher source that writes the object.
- Any Hosting file other than `/`, `/app.js`, and `/style.css`.
