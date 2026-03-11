# Image Recognition API

Multi-provider image recognition API that orchestrates detection services (AWS Rekognition, Google Cloud Vision, Claude/Gemini Vision) for object detection with feedback tracking and accuracy analysis.

## Prerequisites

- Node.js 20+
- Docker (for PostgreSQL)
- AWS credentials with Rekognition and S3 access

## Setup

```bash
# Start PostgreSQL
docker compose up -d

# Install dependencies
npm install

# Configure environment
cp .env.example .env.development
# Edit .env.development with your credentials

# Push database schema
npm run db:push

# Start dev server (port 4111)
npm run dev
```

### Environment Variables

Create `.env.development` at the project root:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=detection_db
DB_USERNAME=postgres
DB_PASSWORD=password
AWS_REGION=ap-northeast-1
S3_BUCKET=<bucket-name>
PORT=4111
```

## Commands

```bash
npm run dev           # Dev server with hot reload (tsx watch, port 4111)
npm run build         # Production build (tsup)
npm run start         # Run production build
npm run typecheck     # Type check (tsc --noEmit)

# Database (Drizzle Kit)
npm run db:push       # Push schema to DB (dev workflow)
npm run db:generate   # Generate migration files
npm run db:migrate    # Apply migrations
npm run db:studio     # Open Drizzle Studio UI
```

## Tech Stack

- **Runtime:** Node.js 20+, TypeScript 5.8, ESM
- **HTTP:** Hono 4.7 + @hono/node-server
- **Database:** PostgreSQL 16 + Drizzle ORM + postgres.js
- **Detection:** AWS Rekognition (primary), multi-provider planned
- **Storage:** AWS S3 (presigned URLs for image access)
- **Validation:** Zod
- **IDs:** cuid2 (collision-resistant text IDs)
- **Build:** tsup (bundler), tsx (dev runner)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/detection/detect` | Upload image for object detection |
| GET | `/detection/list` | List detection history |
| GET | `/detection/:detectionId` | Get detection detail |
| GET | `/detection/:detectionId/image` | Get detection image (presigned URL) |
| DELETE | `/detection/:detectionId` | Delete a detection |
| POST | `/detection/feedback` | Submit label feedback |
| GET | `/detection/:detectionId/feedback` | Get feedback for a detection |
| GET | `/detection/analysis/summary` | Accuracy statistics |
| GET | `/detection/analysis/confidence-distribution` | Confidence distribution data |
| GET | `/detection/analysis/misclassifications` | Misclassification analysis |
| POST | `/detection/stress-test` | Run image degradation test |
| GET | `/detection/stress-tests` | List stress test results |

## Project Structure

```
src/
  server.ts                # Hono app entry point, route registration, CORS
  db/
    connection.ts          # postgres.js + Drizzle connection
    index.ts               # DB exports
    models/                # Drizzle table definitions
  lib/
    db-helper.ts           # useDB() singleton, initDB() setup
    detection-helpers.ts   # Detection CRUD + analysis queries
    register-api-route.ts  # Typed route registration helper
    rekognition.ts         # AWS Rekognition DetectLabels wrapper
    s3.ts                  # S3 upload + presigned URL generation
    logger.ts              # Console logger with category prefixes
  routes/
    health.ts              # GET /health
    detection/             # All detection endpoints
  utils/
    format-error-message.ts
```

**Detection flow:** Image upload -> S3 storage -> Rekognition DetectLabels -> parse labels + bounding boxes -> save to DB (detection + labels + raw JSON response) -> return results.

## Python (Experimental)

The `src/` directory also contains Python modules from the original prototype, used for multi-provider evaluation and local model experimentation.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m src.main --mode api    # FastAPI server (port 8000)
python -m src.main --mode ui     # Gradio UI (port 7860)
python -m src.main --mode both   # Both

streamlit run src/dashboard/app.py  # Pipeline comparison dashboard
```

The **Streamlit dashboard** provides a side-by-side comparison UI: upload images, select detection pipelines, and compare results across providers by latency, cost, and detection count.

Providers available in Python: AWS Rekognition, Google Cloud Vision, Claude Vision, Gemini Vision, OpenAI Vision, OpenRouter.
