import { relations } from "drizzle-orm";

import {
  detectionFeedback,
  detectionLabels,
  detections,
  stressTests,
} from "./detection";
import { users } from "./user";

export const detectionsRelations = relations(detections, ({ one, many }) => ({
  user: one(users, {
    fields: [detections.userId],
    references: [users.id],
  }),
  labels: many(detectionLabels),
  feedback: many(detectionFeedback),
  stressTestsAsSource: many(stressTests, { relationName: "sourceDetection" }),
  stressTestsAsResult: many(stressTests, { relationName: "resultDetection" }),
}));

export const detectionLabelsRelations = relations(
  detectionLabels,
  ({ one, many }) => ({
    detection: one(detections, {
      fields: [detectionLabels.detectionId],
      references: [detections.id],
    }),
    feedback: many(detectionFeedback),
  }),
);

export const detectionFeedbackRelations = relations(
  detectionFeedback,
  ({ one }) => ({
    label: one(detectionLabels, {
      fields: [detectionFeedback.labelId],
      references: [detectionLabels.id],
    }),
    detection: one(detections, {
      fields: [detectionFeedback.detectionId],
      references: [detections.id],
    }),
  }),
);

export const stressTestsRelations = relations(stressTests, ({ one }) => ({
  sourceDetection: one(detections, {
    fields: [stressTests.sourceDetectionId],
    references: [detections.id],
    relationName: "sourceDetection",
  }),
  resultDetection: one(detections, {
    fields: [stressTests.resultDetectionId],
    references: [detections.id],
    relationName: "resultDetection",
  }),
}));
