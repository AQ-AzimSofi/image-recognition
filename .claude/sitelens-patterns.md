# sitelens-ai-api Patterns Reference

Patterns extracted from the `sitelens-ai-api` repository at `kjm/sitelens/sitelens-ai-api/`.
Follow these exactly for migration compatibility.

## Project Structure

```
sitelens-ai-api/
├── package.json              # Root - npm workspaces
├── docker-compose.yaml       # PostgreSQL (pgvector) + pgAdmin
├── app/                      # Main application workspace
│   ├── package.json
│   ├── tsconfig.json
│   ├── .env.development.example
│   └── src/
│       ├── mastra/
│       │   ├── index.ts              # Mastra instance & auth middleware
│       │   ├── agents/
│       │   │   └── base-agent.ts     # Single AI agent (Gemini 2.5 Flash)
│       │   ├── custom-routes/
│       │   │   ├── chat-adapter.ts   # SSE streaming endpoint
│       │   │   ├── warmup.ts         # Lambda cold-start prevention
│       │   │   ├── img-redirect.ts   # S3 presigned URL redirect
│       │   │   └── rag/
│       │   │       ├── upload.ts     # Single file RAG upload
│       │   │       └── batch-upload.ts
│       │   ├── db/
│       │   │   ├── vector-db.ts      # PgVector connection
│       │   │   ├── vector-query.ts   # RAG search tool
│       │   │   ├── helpers.ts        # Document extractors
│       │   │   ├── pdf-image-converter.ts
│       │   │   ├── pdf-page-chunker.ts
│       │   │   └── store-embedding-with-store-images.ts
│       │   └── utils/
│       │       └── image-registry.ts # Anti-hallucination image ID mapping
│       └── lib/
│           ├── logger.ts             # Pino-based logging
│           ├── cognito-verifier.ts   # JWT validation
│           ├── s3.ts                 # S3 + presigned URLs
│           └── chat-route/
│               ├── stream-adapter.ts           # Mastra -> AI SDK v5
│               └── image-marker-transformer.ts # {{IMG:id}} -> URL
├── docker/
│   ├── pg/Dockerfile         # pgvector/pgvector:pg17
│   └── pgadmin/servers.json
├── scripts/
│   └── .env.deploy.example
└── docs/
```

## Tech Stack

- **Runtime:** Node.js 22+, TypeScript 5.8, ESM modules
- **Framework:** Mastra 1.3 (Hono-based AI agent framework)
- **AI Model:** Google Gemini 2.5 Flash (`@ai-sdk/google` v2.0)
- **AI SDK:** `ai` v5.0 (streaming, structured outputs)
- **Embeddings:** `gemini-embedding-001` (3072 dimensions)
- **Vector DB:** PgVector via `@mastra/pg` v1.6
- **Agent Memory:** LibSQL via `@mastra/libsql` v1.6 (in-memory for dev)
- **RAG:** `@mastra/rag` v2.1 (recursive chunking, 640 tokens, 50 overlap)
- **Auth:** AWS Cognito JWT (`aws-jwt-verify` v5.1)
- **Storage:** AWS S3 + CloudFront (`@aws-sdk/client-s3` v3.943)
- **Logging:** Pino via `@mastra/loggers`
- **Validation:** Zod v3.25
- **PDF:** `pdfjs-dist` v5.3 + Poppler (pdftoppm)
- **Documents:** `mammoth` (DOCX), `xlsx` (Excel), `papaparse` (CSV)

## Route Pattern (Mastra/Hono)

Routes use `registerApiRoute` from `@mastra/core`:
```typescript
export const myRoute = {
  path: "/my-endpoint",
  method: "POST",
  handler: async (c) => {
    // c is Hono Context
    const body = await c.req.json();
    return c.json({ success: true });
  },
} satisfies RegisterApiRouteOptions<string, object>;
```

Registration happens in `mastra/index.ts` via the Mastra constructor's `server.apiRoutes` array.

## API Endpoints

| Path | Method | Auth | Purpose |
|------|--------|------|---------|
| `/stream` | POST | Yes | Chat with SSE streaming |
| `/rag/upload` | POST | Yes | Single file RAG upload |
| `/rag/batch-upload` | POST | Yes | Multi-file upload (max 10) |
| `/img/:encodedKey` | GET | No | S3 presigned URL redirect (302) |
| `/warmup` | POST | No | Lambda cold-start prevention |
| `/swagger-ui` | GET | No | OpenAPI docs |

## Auth Middleware

Centralized in `mastra/index.ts`. Cognito JWT validated via `cognito-verifier.ts`.
- Production: requires `COGNITO_USER_POOL_ID` env var
- Development: auth bypassed when `COGNITO_USER_POOL_ID` is empty
- Token extracted from `Authorization` header or request body

## Agent Pattern

Single agent (`base-agent.ts`):
```typescript
new Agent({
  id: "base-agent",
  model: google("gemini-2.5-flash"),
  tools: { vectorQueryTool },
  memory: new Memory({ storage: new LibSQLStore(...) }),
  maxSteps: 10,
  instructions: "..." // Japanese system prompt
})
```

System prompt is entirely in Japanese, optimized for SiteLens manual Q&A.

## Mastra Tool Pattern

```typescript
import { createTool } from "@mastra/core/tools";

export const myTool = createTool({
  id: "my-tool",
  description: "...",
  inputSchema: z.object({ query: z.string() }),
  execute: async ({ context }) => { ... },
});
```

Currently only one tool: `vectorQueryTool` (RAG semantic search, returns top-7 results).

## RAG Pipeline

1. File upload -> buffer extraction
2. PDF special processing: page-based chunking + Poppler image generation
3. Text chunking: recursive strategy, 640 tokens max, 50 token overlap
4. Embedding: Gemini embedding model, batches of 100
5. Vector storage: PgVector upsert in batches of 100
6. Metadata: text, fileName, fileType, pageNumber, imageFullPath, totalPages

## ImageRegistry (Anti-Hallucination)

Per-request `ImageRegistry` maps sequential IDs (`img-001`, `img-002`) to actual URLs.
- AI agent only sees `{{IMG:id}}` placeholders in tool results
- `image-marker-transformer.ts` resolves `{{IMG:id}}` to real URLs during stream output
- Invalid/unregistered IDs are filtered out

## S3 Pattern

```typescript
import { S3Client } from "@aws-sdk/client-s3";
// Module-level client
const s3 = new S3Client({ region: "ap-northeast-1" });
// Dual-mode: local filesystem (dev) vs S3 + CloudFront (prod)
```

Functions: `uploadToS3(key, body, contentType)`, `generatePresignedUrl(key, expiresIn)`

## Logger Pattern

```typescript
import { PinoLogger } from "@mastra/loggers";
// Custom logger with prefix
export const ragLogger = new CustomLogger("[RAG]");
```

Prefixes defined in `LOG_MSG_PREFIXES` constant.

## Streaming Pattern

Chat responses flow through:
1. Mastra agent `fullStream` iterator
2. `stream-adapter.ts` converts Mastra stream -> AI SDK v5 format
3. `image-marker-transformer.ts` resolves `{{IMG:id}}` markers -> real URLs
4. SSE output to client

## Database

- **Vector store:** PgVector on PostgreSQL 17
  - Index: `mastra_rag`, dimension: 3072, schema: `sitelens`
  - Connection: individual env vars (`DB_HOST`, etc.) or `POSTGRES_CONNECTION_STRING`
  - Dev fallback: `postgresql://mastra_user:mastra_password@localhost:5434/mastra_db`
  - Production: RDS Aurora with SSL (`rds-ca-bundle.pem`)
- **Agent memory:** LibSQL (`:memory:` in dev, file-based in prod)

## Docker

```yaml
# docker-compose.yaml
postgres:  # pgvector/pgvector:pg17, port 5434
pgadmin:   # dpage/pgadmin4, port 5051
```

## Commands

```bash
# Development
npm install              # All workspaces
npm run mastra:dev       # Dev server (localhost:4111)
npm run host:images      # Local image server (localhost:3001)
docker compose up -d     # PostgreSQL + pgAdmin

# Build
cd app && mastra build   # Production build
cd app && mastra start   # Production server
cd app && npx tsc --noEmit  # Type checking
```

## TypeScript Config

- Target: ES2022, Module: ES2022 (ESM)
- Module resolution: bundler
- Strict mode enabled
- Path alias: `@/*` -> `src/*`

## Key Dependencies (app/package.json)

```
@ai-sdk/google: ^2.0.23
ai: ^5.0.0
@mastra/core: ^1.6.0
@mastra/rag: ^2.1.0
@mastra/pg: ^1.6.0
@mastra/libsql: ^1.6.0
@mastra/memory: ^1.5.0
@mastra/loggers: ^1.0.2
@aws-sdk/client-s3: ^3.943.0
@aws-sdk/client-secrets-manager: ^3.943.0
@aws-sdk/s3-request-presigner: ^3.943.0
aws-jwt-verify: ^5.1.1
pdfjs-dist: ^5.3.93
mammoth: ^1.9.1
xlsx: ^0.18.5
papaparse: ^5.5.3
zod: ^3.25.67
dayjs: ^1.11.13
```

## Environment Variables

```bash
# Required
GOOGLE_GENERATIVE_AI_API_KEY=     # Gemini API key
GEMINI_MODEL_ID=gemini-2.5-flash

# Database (local)
POSTGRES_CONNECTION_STRING=postgresql://mastra_user:mastra_password@localhost:5434/mastra_db

# Database (production - individual vars)
DB_HOST=  DB_PORT=5432  DB_NAME=sitelens  DB_USERNAME=  DB_PASSWORD=

# Auth (empty = bypass in dev)
COGNITO_USER_POOL_ID=
COGNITO_CLIENT_IDS=

# Storage (production only)
S3_BUCKET=
CLOUDFRONT_DOMAIN=
IMAGE_BUCKET_PATH=rag/images
```
