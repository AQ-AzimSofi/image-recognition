import useDBFromRepo from "@repo/db/lib/db";
import * as fs from "fs";

import { logger } from "./logger";

export const getDBData = () => {
  if (!process.env.DATABASE_URL && !process.env.DB_HOST) {
    console.info("No Database Info found returning default");
    return {
      certString: undefined,
      dbHost: "localhost",
      dbPort: Number(process.env.DB_PORT) || 5432,
      dbName: "detection_db",
      dbUsername: "postgres",
      dbPassword: "password",
    };
  }

  const dbHost = process.env.DB_HOST;
  const dbPort = Number(process.env.DB_PORT) || 5432;
  const dbName = process.env.DB_NAME;
  const dbUsername = process.env.DB_USERNAME;
  const dbPassword = process.env.DB_PASSWORD;

  if (!dbHost || !dbName || !dbUsername || !dbPassword) {
    throw new Error(
      "Database configuration incomplete. All of: DB_HOST, DB_NAME, DB_USERNAME, DB_PASSWORD",
    );
  }

  let certString: string | undefined = undefined;
  if (process.env.NODE_ENV === "production") {
    try {
      certString = fs.readFileSync("ap-northeast-1-bundle.pem").toString();
    } catch {
      // Ignore, proceed without cert
    }
  }

  return {
    certString,
    dbHost,
    dbPort,
    dbName,
    dbUsername,
    dbPassword,
  };
};

let connection: ReturnType<typeof useDBFromRepo>;

const setup = () => {
  const { certString, dbHost, dbPort, dbName, dbUsername, dbPassword } =
    getDBData();
  if (!dbHost || !dbPort || !dbName || !dbUsername || !dbPassword) {
    logger.error("Database configuration is not set.");
    throw new Error("Database configuration is not set.");
  }

  try {
    connection = useDBFromRepo(
      dbHost,
      dbPort,
      dbName,
      dbUsername,
      dbPassword,
      certString,
    );
    logger.info("Database connected successfully.");
  } catch (error) {
    logger.error("Failed to connect to the database.", {
      error: { message: (error as Error).message },
    });
    throw error;
  }
};

setup();

export const useDB = () => {
  return connection;
};
