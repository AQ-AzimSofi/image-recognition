# Project Context

## About This Repo
This is a **personal project** by Azim Sofi. It is not team-assigned, has no sprint schedule, and no daily meetings. The goal is to build an image recognition API that orchestrates multiple detection providers for maximum accuracy.

## Vision
Orchestrate an image recognition workflow:
1. **Bounding box detection** via AWS Rekognition or Google Cloud Vision (structured coordinates)
2. **Contextual identification** via Gemini or Claude Vision (accurate object naming, product recognition, text reading)
3. Combine both for bounding boxes + high-accuracy labels

See `docs/image-recognition-tech-comparison.md` for the full technology evaluation.

## Relationship to sitelens-ai-agent
- **sitelens-ai-agent** (separate repo) is the team's production AI agent (SiteLens Assistant)
- If this project proves valuable, detection features may be proposed to the PM for integration
- This repo is personal until approved

## Repository Structure
```
image-recognition/
├── src/
│   ├── server.ts          # Hono HTTP server entry point
│   ├── db/
│   │   ├── connection.ts  # PostgreSQL connection (postgres.js + Drizzle)
│   │   ├── index.ts       # Barrel export
│   │   └── models/        # Drizzle ORM table definitions
│   ├── lib/
│   │   ├── db-helper.ts   # DB connection setup and useDB()
│   │   ├── detection-helpers.ts  # CRUD operations for detections
│   │   ├── logger.ts      # Simple console logger with prefixes
│   │   ├── register-api-route.ts # Route registration helper
│   │   ├── rekognition.ts # AWS Rekognition client
│   │   └── s3.ts          # AWS S3 client (upload, presigned URLs)
│   ├── routes/
│   │   ├── health.ts
│   │   └── detection/     # All detection CRUD + analysis routes
│   └── utils/
│       └── format-error-message.ts
├── drizzle.config.ts      # Drizzle Kit config (schema in src/db/models/)
├── package.json           # Single package, no workspaces
├── tsconfig.json
├── docker-compose.yml     # PostgreSQL 16
├── docs/                  # Research documents
├── datasets/              # Scraped image datasets for testing
├── scripts/               # Utility scripts (batch detection, download)
└── infra/                 # CDK IAM permissions
```

## Architecture Decisions
- **Single package** (no monorepo/workspaces) -- simplicity over sitelens compatibility
- Hono for HTTP (14KB, fast)
- Drizzle ORM for database (lightweight, type-safe)
- PostgreSQL with cuid2 text IDs and jsonb columns
- AWS Rekognition for object detection (current, will expand to multi-provider)
- No AI agent framework -- direct function calls, pipeline orchestration

## Tech Stack
- **Runtime:** Node.js 20+, TypeScript, ESM
- **HTTP:** Hono + @hono/node-server
- **Database:** PostgreSQL 16 + Drizzle ORM + postgres.js
- **Detection:** AWS Rekognition (bounding boxes + labels)
- **Storage:** AWS S3 (image storage, presigned URLs)
- **Dev tools:** tsx (dev server), tsup (build), drizzle-kit (migrations)

## Development
```bash
docker compose up -d          # Start PostgreSQL
npm run dev                   # Start dev server (port 4111)
npm run db:generate           # Generate migration
npm run db:push               # Push schema to DB
npm run db:studio             # Drizzle Studio UI
```

## Environment Variables
Create `.env.development` at project root:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=detection_db
DB_USERNAME=postgres
DB_PASSWORD=password
AWS_REGION=ap-northeast-1
S3_BUCKET=
PORT=4111
```
