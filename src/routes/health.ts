import type { RegisterApiRouteOptions } from "@/lib/register-api-route";

export const healthRoute = {
  path: "/health",
  method: "GET",
  handler: async (c) => {
    return c.json({ status: "ok", timestamp: new Date().toISOString() });
  },
} satisfies RegisterApiRouteOptions<string, object>;
