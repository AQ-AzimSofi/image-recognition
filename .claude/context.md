# Project Context

## About This Repo
This is a **personal project** by Azim Sofi. It is not team-assigned, has no sprint schedule, and no daily meetings. The goal is to build a proof-of-concept for object detection using AWS Rekognition. If the product is promising, it will be proposed to the PM and potentially migrated into the team's production system.

## Relationship to sitelens-ai-agent
- **sitelens-ai-agent** (separate repo, location varies by machine) is the team's production Mastra-based AI agent (SiteLens Assistant)
- It is a RAG chatbot deployed as a Lambda function via ECR
- This POC repo (`image-recognition`) must **never modify** sitelens-ai-agent directly
- Instead, the same patterns and stack are replicated here so migration is seamless
- sitelens-ai-agent is strictly for the sitelens team; this repo is personal until approved

## Migration Plan
When approved, the contents of `db/`, `app/`, `infra/` from this repo can be copied into sitelens-ai-agent with minimal changes:
- Table definitions, routes, and helpers are designed to slot in directly
- Same dependency versions, same code patterns, same folder structure
- Key migration change: add `getCognitoUsername(c)` authentication to routes (currently unauthenticated in POC)

## Why Rekognition Fits the Existing Lambda
- Rekognition processing happens on AWS's side -- the Lambda just sends an API call
- The Lambda's compute load is minimal (send image bytes, receive JSON)
- Single codebase, single deployment, no cross-Lambda communication needed

## Repository Structure
- `src/` - Original Python POC (FastAPI + Gradio + SQLite, runs locally)
- `db/` - Drizzle ORM schema (mirrors sitelens-ai-agent/db/)
- `app/` - Hono/Mastra backend (mirrors sitelens-ai-agent/app/)
- `infra/` - CDK IAM permissions (mirrors sitelens-ai-agent/infra/)

## Architecture Decisions
- Monorepo with npm workspaces: `db/`, `app/` (matches sitelens-ai-agent)
- Hono + Drizzle ORM + Mastra (same stack as sitelens-ai-agent)
- PostgreSQL with cuid2 text IDs and jsonb columns
- AWS Rekognition for object detection (managed service, no model hosting)
- Turborepo for task orchestration

## What the Python POC Does
- FastAPI backend on localhost:8000
- Gradio web UI on localhost:7860
- AWS Rekognition integration via boto3
- SQLite database (PostgreSQL in production)
- Feedback system, stress testing, accuracy analysis
