import type { Context } from "hono";

import { detectionLogger as logger } from "@/lib/logger";
import type { RegisterApiRouteOptions } from "@/lib/register-api-route";
import { saveFeedback } from "@/lib/detection-helpers";
import { formatErrorForLogging } from "@/utils/format-error-message";

export const submitFeedbackRoute = {
  path: "/detection/feedback",
  method: "POST",
  openapi: {
    summary: "Submit feedback for a detection label",
    tags: ["Detection Feedback"],
  },
  handler: async (c: Context) => {
    try {
      const body = (await c.req.json()) as {
        labelId: string;
        detectionId: string;
        isCorrect: number;
        isWrongReason?: string;
        expectedLabel?: string;
        reviewerNotes?: string;
      };

      if (!body.labelId || !body.detectionId) {
        return c.json(
          { error: "labelId and detectionId are required" },
          400,
        );
      }

      if (body.isCorrect !== 0 && body.isCorrect !== 1) {
        return c.json({ error: "isCorrect must be 0 or 1" }, 400);
      }

      const feedback = await saveFeedback(body);

      logger.info("Feedback saved", {
        labelId: body.labelId,
        isCorrect: body.isCorrect,
      });

      return c.json(feedback, 201);
    } catch (e) {
      logger.error("Failed to save feedback", {
        error: formatErrorForLogging(e),
      });
      return c.json({ error: "Failed to save feedback" }, 500);
    }
  },
} satisfies RegisterApiRouteOptions<string, object>;
