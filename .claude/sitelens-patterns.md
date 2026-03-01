# sitelens-ai-agent Patterns Reference

Patterns extracted from the `sitelens-ai-agent` repository (separate repo, location varies by machine).
Follow these exactly for migration compatibility.

## Database Models (Drizzle ORM)
- Location: `db/src/models/`
- PK: `text("id").primaryKey().$defaultFn(() => createId())` (cuid2)
- FK: `.references(() => table.id, { onDelete: "set null" })`
- Always export: InsertSchema, UpdateSchema (partial), SelectSchema, inferred type
- Schemas: `createInsertSchema()` / `createSelectSchema()` from `drizzle-zod`
- Relations: separate file (e.g., `detection-relations.ts`)
- All exports barrel'd through `db/src/index.ts`

## Route Pattern (Hono)
- Type: `satisfies RegisterApiRouteOptions<string, object>`
- Fields: `path`, `method`, `openapi`, `handler`
- Registration: `registerHonoApiRoute(route, app)` in `hono-server.ts`
- Body parsing: `c.req.json()` for JSON, `c.req.parseBody()` for multipart
- Auth: `getCognitoUsername(c)` from middleware context
- DB: `const db = useDB()` at module level

## Logger Pattern
- `CustomLogger` class extending `PinoLogger` from `@mastra/loggers`
- Prefixes defined in `LOG_MSG_PREFIXES` constant
- Named exports: `export const detectionLogger = new CustomLogger("[Detection]")`

## S3 Pattern
- Module-level client: `new S3Client({ region })` with LocalStack fallback
- `uploadToS3(key, body, contentType)`, `generatePresignedUrl(key, expiresIn)`
- Image redirect: decode S3 key -> presigned URL -> 302 redirect

## Rekognition Pattern
- Same module-level client pattern as S3
- `new RekognitionClient({ region })` with matching version range
- Match `@aws-sdk/client-s3` version: `^3.997.0`

## Mastra Tool Pattern
- `createTool({ id, description, inputSchema, execute })`
- Register in `base-agent.ts` tools object
- Use zod schema for input validation

## Helper Pattern (DB operations)
- Module-level `const db = useDB()`
- Named async exports for CRUD operations
- Use `drizzle-orm` operators: `eq`, `desc`, `sql`, etc.
- Return raw query results

## CDK Infrastructure
- Lambda: `this.function.addToPrincipalPolicy(new iam.PolicyStatement(...))`
- ECS: `taskRole.addToPrincipalPolicy(new iam.PolicyStatement(...))`
- Rekognition needs: `actions: ["rekognition:DetectLabels"], resources: ["*"]`

## Key Dependencies (app/package.json)
- `@aws-sdk/client-s3`: `^3.997.0`
- `@aws-sdk/client-rekognition`: `^3.997.0`
- `@aws-sdk/s3-request-presigner`: `^3.997.0`
- `drizzle-orm`: `^0.37.0`
- `@mastra/core`: `^1.0.0`
- `@mastra/hono`: `^1.0.1`
- `ai`: `^5.0.70`
- `hono` (via @mastra/hono)
