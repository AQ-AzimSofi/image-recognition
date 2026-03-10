import { createId } from "@paralleldrive/cuid2";
import {
  boolean,
  pgTable,
  text,
  timestamp,
  varchar,
} from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";

export const users = pgTable("users", {
  id: text("id")
    .primaryKey()
    .$defaultFn(() => createId()),
  profileText: text("profile_text").notNull(),
  useProfileTextForAI: boolean("use_profile_text_for_ai").default(false),
  createdAt: timestamp("created_at").defaultNow(),
  cognitoUsername: varchar("cognito_username", { length: 255 })
    .unique()
    .notNull(),
  locale: varchar("locale", { length: 10 }).default("ja"),
  updatedAt: timestamp("updated_at").defaultNow(),
  email: text("email"),
  fullName: text("full_name"),
});

export const insertUserSchema = createInsertSchema(users);
export const selectUserSchema = createInsertSchema(users);

export type User = typeof users.$inferSelect;
