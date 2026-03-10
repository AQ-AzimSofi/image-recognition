import { createId } from "@paralleldrive/cuid2";
import {
  integer,
  jsonb,
  pgTable,
  real,
  text,
  timestamp,
} from "drizzle-orm/pg-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";

import { users } from "./user";

export const detections = pgTable("detections", {
  id: text("id")
    .primaryKey()
    .$defaultFn(() => createId()),
  userId: text("user_id").references(() => users.id, {
    onDelete: "set null",
  }),
  imageFilename: text("image_filename").notNull(),
  imageS3Key: text("image_s3_key").notNull(),
  imageWidth: integer("image_width"),
  imageHeight: integer("image_height"),
  detectedAt: timestamp("detected_at").defaultNow().notNull(),
  minConfidence: real("min_confidence").default(50),
  maxLabels: integer("max_labels").default(50),
  rawResponse: jsonb("raw_response"),
  labelCount: integer("label_count").notNull().default(0),
  notes: text("notes"),
});

export const DetectionInsertSchema = createInsertSchema(detections);
export const DetectionUpdateSchema = DetectionInsertSchema.partial();
export const DetectionSelectSchema = createSelectSchema(detections);

export type Detection = typeof detections.$inferSelect;

export const detectionLabels = pgTable("detection_labels", {
  id: text("id")
    .primaryKey()
    .$defaultFn(() => createId()),
  detectionId: text("detection_id")
    .references(() => detections.id, { onDelete: "cascade" })
    .notNull(),
  name: text("name").notNull(),
  confidence: real("confidence").notNull(),
  category: text("category"),
  parents: text("parents"),
  hasBoundingBox: integer("has_bounding_box").notNull().default(0),
  bboxLeft: real("bbox_left"),
  bboxTop: real("bbox_top"),
  bboxWidth: real("bbox_width"),
  bboxHeight: real("bbox_height"),
  instanceConfidence: real("instance_confidence"),
});

export const DetectionLabelInsertSchema = createInsertSchema(detectionLabels);
export const DetectionLabelSelectSchema = createSelectSchema(detectionLabels);

export type DetectionLabel = typeof detectionLabels.$inferSelect;

export const detectionFeedback = pgTable("detection_feedback", {
  id: text("id")
    .primaryKey()
    .$defaultFn(() => createId()),
  labelId: text("label_id")
    .references(() => detectionLabels.id, { onDelete: "cascade" })
    .notNull(),
  detectionId: text("detection_id")
    .references(() => detections.id, { onDelete: "cascade" })
    .notNull(),
  isCorrect: integer("is_correct"),
  isWrongReason: text("is_wrong_reason"),
  expectedLabel: text("expected_label"),
  reviewerNotes: text("reviewer_notes"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const DetectionFeedbackInsertSchema =
  createInsertSchema(detectionFeedback);
export const DetectionFeedbackSelectSchema =
  createSelectSchema(detectionFeedback);

export type DetectionFeedback = typeof detectionFeedback.$inferSelect;

export const stressTests = pgTable("stress_tests", {
  id: text("id")
    .primaryKey()
    .$defaultFn(() => createId()),
  sourceDetectionId: text("source_detection_id")
    .references(() => detections.id, { onDelete: "cascade" })
    .notNull(),
  degradationType: text("degradation_type").notNull(),
  degradationLevel: real("degradation_level").notNull(),
  resultDetectionId: text("result_detection_id").references(
    () => detections.id,
    { onDelete: "set null" },
  ),
  labelDiff: jsonb("label_diff"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const StressTestInsertSchema = createInsertSchema(stressTests);
export const StressTestSelectSchema = createSelectSchema(stressTests);

export type StressTest = typeof stressTests.$inferSelect;
