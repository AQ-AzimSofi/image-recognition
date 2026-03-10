import type { Context } from "hono";

import { detectionLogger as logger } from "@/lib/logger";
import type { RegisterApiRouteOptions } from "@/lib/register-api-route";
import { getDetection } from "@/lib/detection-helpers";
import { generatePresignedUrl } from "@/lib/s3";
import { formatErrorForLogging } from "@/utils/format-error-message";

export const getDetectionImageRoute = {
  path: "/detection/:detectionId/image",
  method: "GET",
  openapi: {
    summary: "Redirect to detection image via presigned URL",
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

      const presignedUrl = await generatePresignedUrl(detection.imageS3Key);
      return c.redirect(presignedUrl, 302);
    } catch (e) {
      logger.error("Failed to get detection image", {
        error: formatErrorForLogging(e),
      });
      return c.json({ error: "Failed to generate image URL" }, 500);
    }
  },
} satisfies RegisterApiRouteOptions<string, object>;
