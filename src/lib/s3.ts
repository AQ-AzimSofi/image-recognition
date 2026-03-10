import {
  S3Client,
  PutObjectCommand,
  GetObjectCommand,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const REGION = process.env.AWS_REGION || "ap-northeast-1";
const BUCKET = process.env.S3_BUCKET;
const USE_LOCALSTACK = process.env.USE_LOCALSTACK === "true";

if (!BUCKET && process.env.NODE_ENV === "production") {
  throw new Error("S3_BUCKET is required in production");
}

const s3Client = BUCKET
  ? new S3Client({
      region: REGION,
      ...(USE_LOCALSTACK && {
        endpoint: `http://localhost:${process.env.LOCALSTACK_HOST_PORT || "4566"}`,
        forcePathStyle: true,
        credentials: { accessKeyId: "test", secretAccessKey: "test" },
      }),
    })
  : null;

export async function uploadToS3(
  key: string,
  body: Buffer,
  contentType: string,
): Promise<string> {
  if (!s3Client || !BUCKET) {
    throw new Error("S3 is not configured (S3_BUCKET not set)");
  }

  await s3Client.send(
    new PutObjectCommand({
      Bucket: BUCKET,
      Key: key,
      Body: body,
      ContentType: contentType,
    }),
  );

  return key;
}

export async function generatePresignedUrl(
  key: string,
  expiresIn: number = 3600,
): Promise<string> {
  if (!s3Client || !BUCKET) {
    throw new Error("S3 is not configured (S3_BUCKET not set)");
  }

  const command = new GetObjectCommand({
    Bucket: BUCKET,
    Key: key,
  });

  return getSignedUrl(s3Client, command, { expiresIn });
}

export function isS3Enabled(): boolean {
  return !!BUCKET;
}

export function getS3Bucket(): string | undefined {
  return BUCKET;
}

export function encodeS3Key(key: string): string {
  return Buffer.from(key).toString("base64url");
}

export function decodeS3Key(encoded: string): string | null {
  try {
    return Buffer.from(encoded, "base64url").toString("utf-8");
  } catch {
    return null;
  }
}
