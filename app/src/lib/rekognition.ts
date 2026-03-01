import {
  RekognitionClient,
  DetectLabelsCommand,
  type DetectLabelsCommandOutput,
  type Label,
  type Instance,
} from "@aws-sdk/client-rekognition";

import { rekognitionLogger as logger } from "./logger";
import { uploadToS3, getS3Bucket } from "./s3";

const REGION = process.env.AWS_REGION || "ap-northeast-1";
const MAX_INLINE_BYTES = 5 * 1024 * 1024; // 5MB

const client = new RekognitionClient({ region: REGION });

export class RekognitionError extends Error {
  code: string;

  constructor(message: string, code: string) {
    super(message);
    this.name = "RekognitionError";
    this.code = code;
  }
}

export interface DetectedInstance {
  confidence: number;
  boundingBox?: {
    left: number;
    top: number;
    width: number;
    height: number;
  };
}

export interface DetectedLabel {
  name: string;
  confidence: number;
  categories: string[];
  parents: string[];
  instances: DetectedInstance[];
}

export interface DetectionResult {
  labels: DetectedLabel[];
  labelCount: number;
  rawResponse: DetectLabelsCommandOutput;
}

function parseLabel(label: Label): DetectedLabel {
  const instances: DetectedInstance[] = (label.Instances || []).map(
    (inst: Instance) => ({
      confidence: inst.Confidence || 0,
      boundingBox: inst.BoundingBox
        ? {
            left: inst.BoundingBox.Left || 0,
            top: inst.BoundingBox.Top || 0,
            width: inst.BoundingBox.Width || 0,
            height: inst.BoundingBox.Height || 0,
          }
        : undefined,
    }),
  );

  return {
    name: label.Name || "Unknown",
    confidence: label.Confidence || 0,
    categories: (label.Categories || []).map((c) => c.Name || ""),
    parents: (label.Parents || []).map((p) => p.Name || ""),
    instances,
  };
}

export async function detectLabels(
  imageBytes: Buffer,
  options?: {
    minConfidence?: number;
    maxLabels?: number;
    s3Key?: string;
  },
): Promise<DetectionResult> {
  const minConfidence = options?.minConfidence ?? 50;
  const maxLabels = options?.maxLabels ?? 50;

  logger.info("Starting label detection", {
    imageSizeBytes: imageBytes.length,
    minConfidence,
    maxLabels,
  });

  let imageParam: { Bytes: Buffer } | { S3Object: { Bucket: string; Name: string } };

  if (imageBytes.length > MAX_INLINE_BYTES) {
    const bucket = getS3Bucket();
    if (!bucket) {
      throw new RekognitionError(
        `Image exceeds ${MAX_INLINE_BYTES} bytes and S3 is not configured for fallback`,
        "IMAGE_TOO_LARGE",
      );
    }

    const s3Key =
      options?.s3Key || `rekognition-tmp/${Date.now()}_oversized`;
    await uploadToS3(s3Key, imageBytes, "application/octet-stream");
    imageParam = { S3Object: { Bucket: bucket, Name: s3Key } };
    logger.info("Image exceeds inline limit, using S3 reference", { s3Key });
  } else {
    imageParam = { Bytes: imageBytes };
  }

  try {
    const command = new DetectLabelsCommand({
      Image: imageParam,
      MinConfidence: minConfidence,
      MaxLabels: maxLabels,
    });

    const response = await client.send(command);
    const labels = (response.Labels || []).map(parseLabel);

    logger.info("Detection completed", { labelCount: labels.length });

    return {
      labels,
      labelCount: labels.length,
      rawResponse: response,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    logger.error("Rekognition API error", { error: { message } });
    throw new RekognitionError(message, "REKOGNITION_API_ERROR");
  }
}
