import type { Context } from "hono";

import { detectionLogger as logger } from "@/lib/logger";
import type { RegisterApiRouteOptions } from "@/lib/register-api-route";
import { getConfidenceDistribution } from "@/lib/detection-helpers";
import { formatErrorForLogging } from "@/utils/format-error-message";

export const analysisConfidenceRoute = {
  path: "/detection/analysis/confidence-distribution",
  method: "GET",
  openapi: {
    summary: "Get confidence score distribution",
    tags: ["Detection Analysis"],
  },
  handler: async (c: Context) => {
    try {
      const distribution = await getConfidenceDistribution();
      return c.json(distribution);
    } catch (e) {
      logger.error("Failed to get confidence distribution", {
        error: formatErrorForLogging(e),
      });
      return c.json({ error: "Database error" }, 500);
    }
  },
} satisfies RegisterApiRouteOptions<string, object>;
