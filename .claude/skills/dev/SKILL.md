---
name: dev
description: Start the development environment (PostgreSQL + Hono server)
disable-model-invocation: true
allowed-tools: Bash, Read
---

# Start Development Environment

Quick-start for developers who already have dependencies installed.

## Current state check

Before starting, check:
- Docker is running: !`docker info > /dev/null 2>&1 && echo "Docker: running" || echo "Docker: NOT running"`
- PostgreSQL container: !`docker ps --filter name=detection-db --format "PostgreSQL: {{.Status}}" 2>/dev/null || echo "PostgreSQL: not found"`
- Port 4111: !`curl -s http://localhost:4111/health 2>/dev/null && echo "Server: already running" || echo "Server: not running"`

## Steps

1. **Ensure PostgreSQL is running**. If the `detection-db` container is not running:
   ```
   docker compose up -d
   ```
   If it's already running, skip this step.

2. **Start the Hono server** (if not already running on port 4111):
   ```
   cd app && npx tsx --env-file=.env.development src/hono-server.ts
   ```
   Run this in the background.

3. **Verify** by hitting `http://localhost:4111/health`. Report the result.

4. **Print available endpoints**:
   - `POST /detection/detect` - Upload image for detection
   - `GET  /detection/list` - List detections
   - `GET  /detection/:id` - Detection detail
   - `GET  /detection/:id/image` - Image redirect (presigned URL)
   - `POST /detection/feedback` - Submit label feedback
   - `GET  /detection/:id/feedback` - Get feedback
   - `DELETE /detection/:id` - Delete detection
   - `GET  /detection/analysis/summary` - Accuracy stats
   - `GET  /detection/analysis/misclassifications` - Misclassification pairs
   - `GET  /detection/analysis/confidence-distribution` - Confidence buckets
   - `POST /detection/stress-test` - Run stress test
   - `GET  /detection/stress-tests` - List stress tests
