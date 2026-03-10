import {
  detectionFeedback,
  detectionLabels,
  detections,
  stressTests,
} from "@/db/models/detection";
import { and, desc, eq, gte, lte, like, sql, count } from "drizzle-orm";

import { useDB } from "@/lib/db-helper";
import type { DetectedLabel } from "@/lib/rekognition";

const db = useDB();

export interface SaveDetectionInput {
  userId?: string | null;
  imageFilename: string;
  imageS3Key: string;
  imageWidth?: number | null;
  imageHeight?: number | null;
  minConfidence?: number;
  maxLabels?: number;
  rawResponse?: unknown;
  labels: DetectedLabel[];
  notes?: string;
}

export const saveDetection = async (input: SaveDetectionInput) => {
  const [detection] = await db
    .insert(detections)
    .values({
      userId: input.userId,
      imageFilename: input.imageFilename,
      imageS3Key: input.imageS3Key,
      imageWidth: input.imageWidth,
      imageHeight: input.imageHeight,
      minConfidence: input.minConfidence,
      maxLabels: input.maxLabels,
      rawResponse: input.rawResponse,
      labelCount: input.labels.length,
      notes: input.notes,
    })
    .returning();

  if (input.labels.length > 0) {
    const labelRows = input.labels.flatMap((label) => {
      if (label.instances.length > 0) {
        return label.instances.map((instance) => ({
          detectionId: detection.id,
          name: label.name,
          confidence: label.confidence,
          category: label.categories.join(", ") || null,
          parents: label.parents.join(", ") || null,
          hasBoundingBox: instance.boundingBox ? 1 : 0,
          bboxLeft: instance.boundingBox?.left ?? null,
          bboxTop: instance.boundingBox?.top ?? null,
          bboxWidth: instance.boundingBox?.width ?? null,
          bboxHeight: instance.boundingBox?.height ?? null,
          instanceConfidence: instance.confidence as number | null,
        }));
      }

      return [
        {
          detectionId: detection.id,
          name: label.name,
          confidence: label.confidence,
          category: label.categories.join(", ") || null,
          parents: label.parents.join(", ") || null,
          hasBoundingBox: 0,
          bboxLeft: null as number | null,
          bboxTop: null as number | null,
          bboxWidth: null as number | null,
          bboxHeight: null as number | null,
          instanceConfidence: null as number | null,
        },
      ];
    });

    await db.insert(detectionLabels).values(labelRows);
  }

  return detection;
};

export const getDetection = async (id: string) => {
  const [detection] = await db
    .select()
    .from(detections)
    .where(eq(detections.id, id));

  if (!detection) return null;

  const labels = await db
    .select()
    .from(detectionLabels)
    .where(eq(detectionLabels.detectionId, id));

  const feedback = await db
    .select()
    .from(detectionFeedback)
    .where(eq(detectionFeedback.detectionId, id));

  return { ...detection, labels, feedback };
};

export const getDetections = async (options?: {
  userId?: string;
  limit?: number;
  offset?: number;
  labelFilter?: string;
  dateFrom?: Date;
  dateTo?: Date;
}) => {
  const limit = options?.limit ?? 20;
  const offset = options?.offset ?? 0;

  const conditions = [];
  if (options?.userId) {
    conditions.push(eq(detections.userId, options.userId));
  }
  if (options?.dateFrom) {
    conditions.push(gte(detections.detectedAt, options.dateFrom));
  }
  if (options?.dateTo) {
    conditions.push(lte(detections.detectedAt, options.dateTo));
  }

  let query = db
    .select()
    .from(detections)
    .orderBy(desc(detections.detectedAt))
    .limit(limit)
    .offset(offset);

  if (conditions.length > 0) {
    query = query.where(and(...conditions)) as typeof query;
  }

  const items = await query;

  if (options?.labelFilter) {
    const detectionIds = items.map((d) => d.id);
    if (detectionIds.length > 0) {
      const matchingLabels = await db
        .select({ detectionId: detectionLabels.detectionId })
        .from(detectionLabels)
        .where(like(detectionLabels.name, `%${options.labelFilter}%`));

      const matchingIds = new Set(matchingLabels.map((l) => l.detectionId));
      return items.filter((d) => matchingIds.has(d.id));
    }
  }

  return items;
};

export const deleteDetection = async (id: string) => {
  const [deleted] = await db
    .delete(detections)
    .where(eq(detections.id, id))
    .returning();
  return deleted;
};

export const saveFeedback = async (input: {
  labelId: string;
  detectionId: string;
  isCorrect: number;
  isWrongReason?: string;
  expectedLabel?: string;
  reviewerNotes?: string;
}) => {
  const [existing] = await db
    .select()
    .from(detectionFeedback)
    .where(eq(detectionFeedback.labelId, input.labelId))
    .limit(1);

  if (existing) {
    const [updated] = await db
      .update(detectionFeedback)
      .set({
        isCorrect: input.isCorrect,
        isWrongReason: input.isWrongReason,
        expectedLabel: input.expectedLabel,
        reviewerNotes: input.reviewerNotes,
      })
      .where(eq(detectionFeedback.id, existing.id))
      .returning();
    return updated;
  }

  const [created] = await db
    .insert(detectionFeedback)
    .values(input)
    .returning();
  return created;
};

export const getDetectionFeedback = async (detectionId: string) => {
  return db
    .select()
    .from(detectionFeedback)
    .where(eq(detectionFeedback.detectionId, detectionId));
};

export const getAnalysisStats = async () => {
  const totalDetections = await db
    .select({ count: count() })
    .from(detections);

  const totalLabels = await db
    .select({ count: count() })
    .from(detectionLabels);

  const totalFeedback = await db
    .select({ count: count() })
    .from(detectionFeedback);

  const correctCount = await db
    .select({ count: count() })
    .from(detectionFeedback)
    .where(eq(detectionFeedback.isCorrect, 1));

  const incorrectCount = await db
    .select({ count: count() })
    .from(detectionFeedback)
    .where(eq(detectionFeedback.isCorrect, 0));

  const feedbackTotal = totalFeedback[0].count;
  const correct = correctCount[0].count;
  const incorrect = incorrectCount[0].count;
  const accuracyRate = feedbackTotal > 0 ? correct / feedbackTotal : null;

  return {
    totalDetections: totalDetections[0].count,
    totalLabels: totalLabels[0].count,
    totalReviewed: feedbackTotal,
    correct,
    incorrect,
    accuracyRate,
  };
};

export const getMisclassifications = async () => {
  const res = await db.execute(sql`
    SELECT
      dl.name AS "detectedLabel",
      df.expected_label AS "expectedLabel",
      COUNT(*) AS "count"
    FROM detection_feedback df
    JOIN detection_labels dl ON df.label_id = dl.id
    WHERE df.is_correct = 0 AND df.expected_label IS NOT NULL
    GROUP BY dl.name, df.expected_label
    ORDER BY COUNT(*) DESC
    LIMIT 20
  `);

  return res as unknown as Array<{
    detectedLabel: string;
    expectedLabel: string;
    count: number;
  }>;
};

export const getConfidenceDistribution = async () => {
  const res = await db.execute(sql`
    SELECT
      CASE
        WHEN confidence >= 90 THEN '90-100'
        WHEN confidence >= 80 THEN '80-90'
        WHEN confidence >= 70 THEN '70-80'
        WHEN confidence >= 60 THEN '60-70'
        WHEN confidence >= 50 THEN '50-60'
        ELSE 'below-50'
      END AS "bucket",
      COUNT(*) AS "count"
    FROM detection_labels
    GROUP BY "bucket"
    ORDER BY "bucket" DESC
  `);

  return res as unknown as Array<{ bucket: string; count: number }>;
};

export const saveStressTest = async (input: {
  sourceDetectionId: string;
  degradationType: string;
  degradationLevel: number;
  resultDetectionId?: string;
  labelDiff?: unknown;
}) => {
  const [created] = await db.insert(stressTests).values(input).returning();
  return created;
};

export const getStressTests = async (sourceDetectionId?: string) => {
  if (sourceDetectionId) {
    return db
      .select()
      .from(stressTests)
      .where(eq(stressTests.sourceDetectionId, sourceDetectionId))
      .orderBy(desc(stressTests.createdAt));
  }
  return db
    .select()
    .from(stressTests)
    .orderBy(desc(stressTests.createdAt))
    .limit(50);
};
