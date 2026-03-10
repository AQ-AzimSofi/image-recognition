import type { Context } from "hono";

import { detectionLogger as logger } from "@/lib/logger";
import type { RegisterApiRouteOptions } from "@/lib/register-api-route";
import { deleteDetection } from "@/lib/detection-helpers";
import { formatErrorForLogging } from "@/utils/format-error-message";

export const deleteDetectionRoute = {
  path: "/detection/:detectionId",
  method: "DELETE",
  openapi: {
    summary: "Delete a detection and its labels",
    tags: ["Detection"],
  },
  handler: async (c: Context) => {
    const detectionId = c.req.param("detectionId");

    if (!detectionId) {
      return c.json({ error: "Detection ID is required" }, 400);
    }

    try {
      const deleted = await deleteDetection(detectionId);

      if (!deleted) {
        return c.json({ error: "Detection not found" }, 404);
      }

      logger.info("Detection deleted", { detectionId });
      return c.json({ deleted: true, id: detectionId });
    } catch (e) {
      logger.error("Failed to delete detection", {
        error: formatErrorForLogging(e),
      });
      return c.json({ error: "Database error" }, 500);
    }
  },
} satisfies RegisterApiRouteOptions<string, object>;
