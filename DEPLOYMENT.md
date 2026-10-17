# Deployment

There is no real server. A merge into `staging` or `main` builds the Docker image for
that environment (see `.github/workflows/cd.yml`).

## Environments and secrets

In GitHub (Settings -> Environments) we created two environments:

| Environment | Branch | Secret |
|---|---|---|
| staging | staging | `API_KEY` (staging key) |
| production | main | `API_KEY` (a different key) |

## What the CD workflow does

1. Writes a `.env` file with `APP_ENV` and the `API_KEY` of the environment.
2. Builds the image (`sms-spam:staging` or `sms-spam:production`).
3. Starts the container with `--env-file .env` and calls `/health` and `/predict`
   with the key to check that it works.

The `.env` file is in `.dockerignore` and `.gitignore`, so the key is never copied into
the image or pushed to GitHub. It is only given to the container when it starts.

## Run an environment on your computer

```bash
cp .env.example .env      # then put APP_ENV=staging and some API_KEY in it
docker build -t sms-spam:staging .
docker run --rm -p 8000:8000 --env-file .env sms-spam:staging
```