import { drizzle } from "drizzle-orm/postgres-js";
import type { Sql } from "postgres";
import postgres from "postgres";

const schema = {};

let connection: Sql<{}>;

const useDB = (
  dbHost: string,
  dbPort: number,
  dbName: string,
  dbUsername: string,
  dbPassword: string,
  certString?: string,
) => {
  const isRDSProxy = isRDSProxyEndpoint(dbHost);
  try {
    if (process.env.NODE_ENV === "production") {
      connection = postgres({
        host: dbHost,
        password: dbPassword,
        user: dbUsername,
        database: dbName,
        port: dbPort,
        ssl: certString
          ? {
              ca: isRDSProxy ? undefined : certString,
              checkServerIdentity: () => undefined,
              require: isRDSProxy ? true : undefined,
            }
          : undefined,
      });
    } else {
      const globalConnection = global as typeof globalThis & {
        connection: Sql<{}>;
      };

      if (!globalConnection.connection) {
        globalConnection.connection = postgres({
          host: dbHost,
          password: dbPassword,
          user: dbUsername,
          database: dbName,
          port: dbPort,
          ssl: certString
            ? {
                ca: isRDSProxy ? undefined : certString,
                checkServerIdentity: () => undefined,
                require: isRDSProxy ? true : undefined,
              }
            : undefined,
        });
      }

      connection = globalConnection.connection;
    }
    console.info("Database connected successfully.");
    return drizzle(connection, {
      schema,
    });
  } catch (error) {
    console.error("Failed to connect to the database.", {
      error: {
        message: (error as Error).message,
      },
    });
    throw error;
  }
};

export function isRDSProxyEndpoint(hostOrUrl: string): boolean {
  return /\.proxy-[a-z0-9]+\..*\.rds\.amazonaws\.com/i.test(hostOrUrl);
}

export default useDB;

export type DrizzleClient = ReturnType<typeof useDB>;
