import type { Context } from "hono";

import { detectionLogger as logger } from "@/lib/logger";
import type { RegisterApiRouteOptions } from "@/lib/register-api-route";
import { getDetections } from "@/lib/detection-helpers";
import { formatErrorForLogging } from "@/utils/format-error-message";

export const getDetectionsRoute = {
  path: "/detection/list",
  method: "GET",
  openapi: {
    summary: "List detections with pagination and filters",
    tags: ["Detection"],
  },
  handler: async (c: Context) => {
    try {
      const limit = Number(c.req.query("limit")) || 20;
      const offset = Number(c.req.query("offset")) || 0;
      const userId = c.req.query("userId") || undefined;
      const labelFilter = c.req.query("label") || undefined;
      const dateFrom = c.req.query("dateFrom")
        ? new Date(c.req.query("dateFrom")!)
        : undefined;
      const dateTo = c.req.query("dateTo")
        ? new Date(c.req.query("dateTo")!)
        : undefined;

      const items = await getDetections({
        userId,
        limit,
        offset,
        labelFilter,
        dateFrom,
        dateTo,
      });

      logger.info("Fetched detections list", { count: items.length });

      return c.json({ items, limit, offset });
    } catch (e) {
      logger.error("Failed to list detections", {
        error: formatErrorForLogging(e),
      });
      return c.json({ error: "Database error" }, 500);
    }
  },
} satisfies RegisterApiRouteOptions<string, object>;
