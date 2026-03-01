import { relations } from "drizzle-orm";

import { detections } from "./detection";
import { users } from "./user";

export const usersRelations = relations(users, ({ many }) => ({
  detections: many(detections),
}));
