---
name: setup
description: Initial setup for new users who just cloned this repo
disable-model-invocation: true
allowed-tools: Bash, Read
---

# Initial Setup

Run the full initial setup for a new developer who just cloned this repo. Execute each step in order, stopping on any error.

## Pre-flight checks

Verify these are installed (fail with a clear message if any are missing):
- Node.js >= 20
- npm
- Docker

## Steps

1. **Start PostgreSQL** via Docker Compose:
   ```
   docker compose up -d
   ```
   Wait for the container to be healthy.

2. **Install npm dependencies** from the repo root:
   ```
   npm install
   ```

3. **Configure environment**: Check if `app/.env.development` exists. If not, create it from this template:
   ```
   LOCAL=true
   PORT=4111
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=detection_db
   DB_USERNAME=postgres
   DB_PASSWORD=password
   AWS_REGION=ap-northeast-1
   S3_BUCKET=
   ```
   Then ask the user for their S3 bucket name and fill it in. Also check if `db/.env.development` exists and create it if needed with the DB credentials.

4. **Run database migration**:
   ```
   cd db && npx drizzle-kit generate
   ```
   Then apply the latest migration SQL:
   ```
   docker exec -i detection-db psql -U postgres -d detection_db < db/migrations/<latest>.sql
   ```

5. **Verify**: Start the server and hit the health endpoint:
   ```
   cd app && npx tsx --env-file=.env.development src/hono-server.ts
   ```
   Then `curl http://localhost:4111/health` to confirm `{"status":"ok"}`.
   After verification, stop the server.

6. **Print summary** of what was set up and how to start developing (point them to `/dev`).
