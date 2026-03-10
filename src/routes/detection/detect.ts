import type { Context } from "hono";

import { detectionLogger as logger } from "@/lib/logger";
import type { RegisterApiRouteOptions } from "@/lib/register-api-route";
import { detectLabels } from "@/lib/rekognition";
import { saveDetection } from "@/lib/detection-helpers";
import { uploadToS3 } from "@/lib/s3";
import { formatErrorForLogging } from "@/utils/format-error-message";

const ALLOWED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/gif",
  "image/bmp",
];
const MAX_FILE_SIZE = 15 * 1024 * 1024; // 15MB

export const detectRoute = {
  path: "/detection/detect",
  method: "POST",
  openapi: {
    summary: "Upload image and detect objects",
    tags: ["Detection"],
  },
  handler: async (c: Context) => {
    try {
      const body = await c.req.parseBody();
      const file = body["file"];

      if (!file || !(file instanceof File)) {
        return c.json({ error: "File is required" }, 400);
      }

      if (!ALLOWED_TYPES.includes(file.type)) {
        return c.json(
          { error: `Unsupported file type: ${file.type}` },
          400,
        );
      }

      if (file.size > MAX_FILE_SIZE) {
        return c.json({ error: "File exceeds 15MB limit" }, 400);
      }

      const arrayBuffer = await file.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);

      const minConfidence = Number(body["minConfidence"]) || 50;
      const maxLabels = Number(body["maxLabels"]) || 50;
      const notes = typeof body["notes"] === "string" ? body["notes"] : undefined;
      const userId = typeof body["userId"] === "string" ? body["userId"] : undefined;

      const timestamp = Date.now();
      const s3Key = userId
        ? `detections/${userId}/${timestamp}_${file.name}`
        : `detections/anonymous/${timestamp}_${file.name}`;

      logger.info("Uploading image to S3", { s3Key, size: file.size });
      await uploadToS3(s3Key, buffer, file.type);

      logger.info("Running Rekognition detection", { minConfidence, maxLabels });
      const result = await detectLabels(buffer, {
        minConfidence,
        maxLabels,
        s3Key,
      });

      const detection = await saveDetection({
        userId,
        imageFilename: file.name,
        imageS3Key: s3Key,
        minConfidence,
        maxLabels,
        rawResponse: result.rawResponse,
        labels: result.labels,
        notes,
      });

      logger.info("Detection saved", {
        detectionId: detection.id,
        labelCount: result.labelCount,
      });

      return c.json(
        {
          id: detection.id,
          imageFilename: file.name,
          labelCount: result.labelCount,
          labels: result.labels,
        },
        201,
      );
    } catch (e) {
      logger.error("Detection failed", {
        error: formatErrorForLogging(e),
      });
      return c.json({ error: "Detection failed" }, 500);
    }
  },
} satisfies RegisterApiRouteOptions<string, object>;
