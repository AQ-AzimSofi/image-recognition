# Phase 2.0.1: Post-Setup Fixes and Developer Experience

## Changes

### Fix: S3 LocalStack flag decoupled from LOCAL env var

**File: `app/src/lib/s3.ts`**

Previously, `IS_LOCAL` was derived from `LOCAL=true`, which forced all S3 calls through LocalStack even when using a real S3 bucket for local development. Changed to a dedicated `USE_LOCALSTACK` env var.

- Before: `const IS_LOCAL = process.env.LOCAL === "true" || process.env.NODE_ENV !== "production";`
- After: `const USE_LOCALSTACK = process.env.USE_LOCALSTACK === "true";`

This allows `LOCAL=true` (for DB and auth bypass) while still using real AWS S3 for image storage.

### Fix: ESM import hoisting breaks dotenv loading

**File: `app/src/hono-server.ts`**

`dotenv.config()` placed after `import` statements was ineffective because ESM hoists all `import` declarations above executable code. By the time `dotenv.config()` ran, `s3.ts` and other modules had already read `process.env` and found empty values.

- Removed: inline `dotenv.config()` from `hono-server.ts`
- Added: `--env-file=.env.development` flag to the `tsx` command

**File: `app/package.json`**

- Before: `"dev": "tsx watch src/hono-server.ts"`
- After: `"dev": "tsx watch --env-file=.env.development src/hono-server.ts"`

Node.js 20+ `--env-file` flag loads environment variables before any module evaluation, bypassing the ESM hoisting issue entirely.

### New: Docker Compose for local PostgreSQL

**File: `docker-compose.yml`**

PostgreSQL 16 Alpine container (`detection-db`) on port 5432 with a named volume for data persistence.

### New: Environment files

**File: `db/.env.development`** - Database credentials for Drizzle CLI (generate, push, studio).

**File: `app/.env.development`** - Full app config: DB credentials, AWS region, S3 bucket name (`image-recognition-detection-dev`).

### New: S3 bucket created

Bucket `image-recognition-detection-dev` created in `ap-northeast-1` via AWS CLI for development image storage.

### New: Claude Code skills

**File: `.claude/skills/setup/SKILL.md`** - `/setup` command for initial repo setup (Docker, npm install, env config, DB migration, health check).

**File: `.claude/skills/dev/SKILL.md`** - `/dev` command for daily development startup (ensure PostgreSQL running, start Hono server, list endpoints).

## File Summary

| File | Change Type |
|------|-------------|
| `app/src/lib/s3.ts` | Modified (USE_LOCALSTACK flag) |
| `app/src/hono-server.ts` | Modified (removed dotenv import) |
| `app/package.json` | Modified (--env-file flag) |
| `docker-compose.yml` | New |
| `db/.env.development` | New |
| `app/.env.development` | New |
| `.claude/skills/setup/SKILL.md` | New |
| `.claude/skills/dev/SKILL.md` | New |
