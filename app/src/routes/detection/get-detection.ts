import type { Context } from "hono";

import { detectionLogger as logger } from "@/lib/logger";
import type { RegisterApiRouteOptions } from "@/lib/register-api-route";
import { getDetection } from "@/lib/detection-helpers";
import { formatErrorForLogging } from "@/utils/format-error-message";

export const getDetectionRoute = {
  path: "/detection/:detectionId",
  method: "GET",
  openapi: {
    summary: "Get detection detail with labels",
    tags: ["Detection"],
  },
  handler: async (c: Context) => {
    const detectionId = c.req.param("detectionId");

    if (!detectionId) {
      return c.json({ error: "Detection ID is required" }, 400);
    }

    try {
      const detection = await getDetection(detectionId);

      if (!detection) {
        return c.json({ error: "Detection not found" }, 404);
      }

      return c.json(detection);
    } catch (e) {
      logger.error("Failed to get detection", {
        error: formatErrorForLogging(e),
      });
      return c.json({ error: "Database error" }, 500);
    }
  },
} satisfies RegisterApiRouteOptions<string, object>;
