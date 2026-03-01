import * as dotenv from "dotenv";
import { defineConfig } from "drizzle-kit";

if (!process.env.GITHUB_ACTIONS) {
  const envFile =
    process.env.NODE_ENV === "production" ? ".env" : ".env.development";
  dotenv.config({ path: envFile });
}

export const DEVOptions = {
  database: process.env.DATABASE_NAME || "postgres",
  host: "127.0.0.1",
  user: process.env.DATABASE_USERNAME || "postgres",
  password: process.env.DATABASE_PASSWORD,
  port: Number(process.env.DB_PORT) || 5432,
  ssl: {
    checkServerIdentity: () => undefined,
    require: true,
  },
};

export default defineConfig({
  schema: "./src/models/**/*.ts",
  dialect: "postgresql",
  dbCredentials: {
    ...(process.env.ENVIRONMENT === "dev"
      ? DEVOptions
      : {
          database: process.env.DB_NAME!,
          host: process.env.DB_HOST!,
          password: process.env.DB_PASSWORD,
          port: Number(process.env.DB_PORT || 5432),
          user: process.env.DB_USERNAME,
          ssl: false,
        }),
  },
  verbose: true,
  strict: true,
  out: "./migrations",
});
