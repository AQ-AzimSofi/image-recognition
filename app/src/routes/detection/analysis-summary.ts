import type { Context } from "hono";

import { detectionLogger as logger } from "@/lib/logger";
import type { RegisterApiRouteOptions } from "@/lib/register-api-route";
import { getAnalysisStats } from "@/lib/detection-helpers";
import { formatErrorForLogging } from "@/utils/format-error-message";

export const analysisSummaryRoute = {
  path: "/detection/analysis/summary",
  method: "GET",
  openapi: {
    summary: "Get detection accuracy and review stats",
    tags: ["Detection Analysis"],
  },
  handler: async (c: Context) => {
    try {
      const stats = await getAnalysisStats();
      return c.json(stats);
    } catch (e) {
      logger.error("Failed to get analysis stats", {
        error: formatErrorForLogging(e),
      });
      return c.json({ error: "Database error" }, 500);
    }
  },
} satisfies RegisterApiRouteOptions<string, object>;
