import * as dotenv from "dotenv";
import { defineConfig } from "drizzle-kit";

if (!process.env.GITHUB_ACTIONS) {
  const envFile =
    process.env.NODE_ENV === "production" ? ".env" : ".env.development";
  dotenv.config({ path: envFile });
}

export default defineConfig({
  schema: "./src/db/models/**/*.ts",
  dialect: "postgresql",
  dbCredentials: {
    database: process.env.DB_NAME || "detection_db",
    host: process.env.DB_HOST || "localhost",
    password: process.env.DB_PASSWORD || "password",
    port: Number(process.env.DB_PORT || 5432),
    user: process.env.DB_USERNAME || "postgres",
    ssl: process.env.NODE_ENV === "production",
  },
  verbose: true,
  strict: true,
  out: "./migrations",
});
