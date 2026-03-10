import type { Context } from "hono";

import { detectionLogger as logger } from "@/lib/logger";
import type { RegisterApiRouteOptions } from "@/lib/register-api-route";
import {
  getDetection,
  saveDetection,
  saveStressTest,
} from "@/lib/detection-helpers";
import { detectLabels } from "@/lib/rekognition";
import { formatErrorForLogging } from "@/utils/format-error-message";

export const stressTestRoute = {
  path: "/detection/stress-test",
  method: "POST",
  openapi: {
    summary: "Run image degradation stress test",
    tags: ["Detection Stress Test"],
  },
  handler: async (c: Context) => {
    try {
      const body = (await c.req.json()) as {
        sourceDetectionId: string;
        degradationType: string;
        degradationLevel: number;
        imageBase64: string;
      };

      if (!body.sourceDetectionId || !body.imageBase64) {
        return c.json(
          { error: "sourceDetectionId and imageBase64 are required" },
          400,
        );
      }

      const source = await getDetection(body.sourceDetectionId);
      if (!source) {
        return c.json({ error: "Source detection not found" }, 404);
      }

      const imageBytes = Buffer.from(body.imageBase64, "base64");

      const result = await detectLabels(imageBytes, {
        minConfidence: source.minConfidence ?? 50,
        maxLabels: source.maxLabels ?? 50,
      });

      const resultDetection = await saveDetection({
        userId: source.userId,
        imageFilename: `stress_${body.degradationType}_${body.degradationLevel}_${source.imageFilename}`,
        imageS3Key: `stress-tests/${source.id}/${body.degradationType}_${body.degradationLevel}`,
        minConfidence: source.minConfidence ?? 50,
        maxLabels: source.maxLabels ?? 50,
        rawResponse: result.rawResponse,
        labels: result.labels,
        notes: `Stress test: ${body.degradationType} at level ${body.degradationLevel}`,
      });

      const sourceLabels = new Set(source.labels.map((l) => l.name));
      const resultLabels = new Set(result.labels.map((l) => l.name));
      const added = result.labels.filter((l) => !sourceLabels.has(l.name));
      const removed = source.labels.filter(
        (l) => !resultLabels.has(l.name),
      );

      const labelDiff = { added, removed };

      const stressTest = await saveStressTest({
        sourceDetectionId: body.sourceDetectionId,
        degradationType: body.degradationType,
        degradationLevel: body.degradationLevel,
        resultDetectionId: resultDetection.id,
        labelDiff,
      });

      logger.info("Stress test completed", {
        stressTestId: stressTest.id,
        addedLabels: added.length,
        removedLabels: removed.length,
      });

      return c.json(
        {
          id: stressTest.id,
          resultDetectionId: resultDetection.id,
          labelDiff,
          resultLabels: result.labels,
        },
        201,
      );
    } catch (e) {
      logger.error("Stress test failed", {
        error: formatErrorForLogging(e),
      });
      return c.json({ error: "Stress test failed" }, 500);
    }
  },
} satisfies RegisterApiRouteOptions<string, object>;
