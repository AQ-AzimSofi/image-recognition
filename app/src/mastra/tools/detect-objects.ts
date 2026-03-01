import { createTool } from "@mastra/core/tools";
import { z } from "zod";

import { logger } from "@/lib/logger";
import { detectLabels } from "@/lib/rekognition";

const inputSchema = z.object({
  imageBase64: z
    .string()
    .describe("Base64-encoded image data"),
  minConfidence: z
    .number()
    .optional()
    .describe("Minimum confidence threshold (default: 50)"),
  maxLabels: z
    .number()
    .optional()
    .describe("Maximum number of labels to return (default: 50)"),
});

export const detectObjectsTool = createTool({
  id: "detect_objects_rekognition",
  description:
    "AWS Rekognitionを使用して画像内の物体を検出します。Base64エンコードされた画像データを受け取り、検出されたラベル、信頼度スコア、バウンディングボックスを返します。",
  inputSchema: inputSchema as any,
  execute: async (input: z.infer<typeof inputSchema>) => {
    logger.debug("[detectObjects] Starting detection");

    try {
      const imageBytes = Buffer.from(input.imageBase64, "base64");

      const result = await detectLabels(imageBytes, {
        minConfidence: input.minConfidence,
        maxLabels: input.maxLabels,
      });

      const formattedLabels = result.labels.map((label) => ({
        name: label.name,
        confidence: `${label.confidence.toFixed(1)}%`,
        categories: label.categories,
        parents: label.parents,
        instanceCount: label.instances.length,
        boundingBoxes: label.instances
          .filter((i) => i.boundingBox)
          .map((i) => i.boundingBox),
      }));

      return {
        labelCount: result.labelCount,
        labels: formattedLabels,
        message: `${result.labelCount}件の物体が検出されました。`,
      };
    } catch (error) {
      logger.error("[detectObjects] Error", {
        error: { message: error instanceof Error ? error.message : String(error) },
      });
      return {
        labelCount: 0,
        labels: [],
        message: "物体検出中にエラーが発生しました。",
        error: error instanceof Error ? error.message : String(error),
      };
    }
  },
});
