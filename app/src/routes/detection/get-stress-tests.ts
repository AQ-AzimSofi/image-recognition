import type { Context } from "hono";

import { detectionLogger as logger } from "@/lib/logger";
import type { RegisterApiRouteOptions } from "@/lib/register-api-route";
import { getStressTests } from "@/lib/detection-helpers";
import { formatErrorForLogging } from "@/utils/format-error-message";

export const getStressTestsRoute = {
  path: "/detection/stress-tests",
  method: "GET",
  openapi: {
    summary: "List stress test results",
    tags: ["Detection Stress Test"],
  },
  handler: async (c: Context) => {
    try {
      const sourceDetectionId =
        c.req.query("sourceDetectionId") || undefined;

      const stressTests = await getStressTests(sourceDetectionId);
      return c.json(stressTests);
    } catch (e) {
      logger.error("Failed to get stress tests", {
        error: formatErrorForLogging(e),
      });
      return c.json({ error: "Database error" }, 500);
    }
  },
} satisfies RegisterApiRouteOptions<string, object>;
