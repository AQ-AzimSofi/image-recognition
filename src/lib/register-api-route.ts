import type { Handler, Hono } from "hono";

export type Methods = "GET" | "POST" | "PUT" | "DELETE" | "PATCH" | "OPTIONS";

export type RegisterApiRouteOptions<P extends string, R extends object> = {
  path: P;
  method: Methods;
  openapi?: Record<string, unknown>;
  handler: Handler;
};

export const registerHonoApiRoute = <P extends string, R extends object>(
  options: RegisterApiRouteOptions<P, R>,
  honoApp: Hono,
) => {
  const { method, handler, path } = options;

  switch (method) {
    case "GET":
      honoApp.get(path, handler);
      break;
    case "POST":
      honoApp.post(path, handler);
      break;
    case "PUT":
      honoApp.put(path, handler);
      break;
    case "DELETE":
      honoApp.delete(path, handler);
      break;
    case "PATCH":
      honoApp.patch(path, handler);
      break;
    case "OPTIONS":
      honoApp.options(path, handler);
      break;
    default:
      throw new Error(`Unsupported method: ${method}`);
  }
};
