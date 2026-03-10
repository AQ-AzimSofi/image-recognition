import type { Context } from "hono";

import { detectionLogger as logger } from "@/lib/logger";
import type { RegisterApiRouteOptions } from "@/lib/register-api-route";
import { getDetectionFeedback } from "@/lib/detection-helpers";
import { formatErrorForLogging } from "@/utils/format-error-message";

export const getDetectionFeedbackRoute = {
  path: "/detection/:detectionId/feedback",
  method: "GET",
  openapi: {
    summary: "Get feedback for a detection",
    tags: ["Detection Feedback"],
  },
  handler: async (c: Context) => {
    const detectionId = c.req.param("detectionId");

    if (!detectionId) {
      return c.json({ error: "Detection ID is required" }, 400);
    }

    try {
      const feedback = await getDetectionFeedback(detectionId);
      return c.json(feedback);
    } catch (e) {
      logger.error("Failed to get feedback", {
        error: formatErrorForLogging(e),
      });
      return c.json({ error: "Database error" }, 500);
    }
  },
} satisfies RegisterApiRouteOptions<string, object>;
