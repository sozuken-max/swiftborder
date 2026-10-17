# Services

Cloud Run service and job source that this repo deploys. Firebase Hosting stays in [`../hosting/`](../hosting/index.html). Dated revisions, schedules, and row counts are in [docs/inventory.md](../docs/inventory.md). Trigger paths below were read from project `swiftborder` on 2026-10-17 (about 13:55 SGT).

| Directory | Resource | Kind and region | Runtime service account | Cloud Build |
| --- | --- | --- | --- | --- |
| [`camdetect/`](camdetect/README.md) | `swiftbackend` (`detect`) | Cloud Run service, `europe-west1` | `1095552466513-compute@developer.gserviceaccount.com` | [`camdetect/cloudbuild.yaml`](camdetect/cloudbuild.yaml). Trigger `76bbca35-c1b4-4836-9f34-d7adda53ea17` (`rmgpgab-swiftbackend-europe-west1-sozuken-max-swiftborder--mtkc`), region **global**, push to `^main$`. |
| [`camdetect/`](camdetect/README.md) | `cam2701-local-backfill` | Cloud Run Job, `asia-southeast1` | `1095552466513-compute@developer.gserviceaccount.com` | [`camdetect/jobs/backfill.cloudbuild.yaml`](camdetect/jobs/backfill.cloudbuild.yaml). Manual trigger `03aefce3-94e7-4840-afab-1d5797fdff76` (`cam2701-local-backfill-image`), region **asia-southeast1**. Image `cloud-run-source-deploy/cam2701-local-backfill`. |
| [`forecastapi/`](forecastapi/README.md) | `forecast-api` (`forecast`) | Cloud Run service, `asia-southeast1` | `forecast-api@swiftborder.iam.gserviceaccount.com` | [`forecastapi/cloudbuild.yaml`](forecastapi/cloudbuild.yaml). Trigger `949ff029-31c9-4521-8ff4-d0e8d16dfa25` (`forecast-api`), region **global**, push to `^main$`. |
| [`gmapfetcher/`](gmapfetcher/README.md) | `gmap-woodlands-fetcher` (`fetchTravelTimes`) | Cloud Run service, `asia-southeast1` | `1095552466513-compute@developer.gserviceaccount.com` | [`gmapfetcher/cloudbuild.yaml`](gmapfetcher/cloudbuild.yaml). Trigger `33e4b77e-dbce-426e-a701-de234ec16d60` (`rmgpgab-gmap-woodlands-fetcher-asia-southeast1-sozuken-max-szsd`), region **global**, push to `^main$`. |

`traffic-backfill` is a separate Cloud Run Job in `asia-southeast1` (same default compute service account). Its image is `asia-southeast1-docker.pkg.dev/swiftborder/traffic-backfill/traffic-backfill`, not the `cam2701-local-backfill` image this folder builds. `weather-backfill-tmp` is a live Cloud Run service in `asia-southeast1` whose source is not in this tree.

Included-file filters for the three push triggers are in the 2026-10-17 inventory note. All four triggers build as `1095552466513-compute@developer.gserviceaccount.com`.
