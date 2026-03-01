import type { Context } from "hono";

import { detectionLogger as logger } from "@/lib/logger";
import type { RegisterApiRouteOptions } from "@/lib/register-api-route";
import { getMisclassifications } from "@/lib/detection-helpers";
import { formatErrorForLogging } from "@/utils/format-error-message";

export const analysisMisclassificationsRoute = {
  path: "/detection/analysis/misclassifications",
  method: "GET",
  openapi: {
    summary: "Get common misclassification pairs",
    tags: ["Detection Analysis"],
  },
  handler: async (c: Context) => {
    try {
      const misclassifications = await getMisclassifications();
      return c.json(misclassifications);
    } catch (e) {
      logger.error("Failed to get misclassifications", {
        error: formatErrorForLogging(e),
      });
      return c.json({ error: "Database error" }, 500);
    }
  },
} satisfies RegisterApiRouteOptions<string, object>;
