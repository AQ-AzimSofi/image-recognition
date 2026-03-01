import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { cors } from "hono/cors";

import { registerHonoApiRoute } from "./lib/register-api-route";
import { healthRoute } from "./routes/health";
import {
  detectRoute,
  getDetectionRoute,
  getDetectionsRoute,
  getDetectionImageRoute,
  submitFeedbackRoute,
  getDetectionFeedbackRoute,
  deleteDetectionRoute,
  analysisSummaryRoute,
  analysisMisclassificationsRoute,
  analysisConfidenceRoute,
  stressTestRoute,
  getStressTestsRoute,
} from "./routes/detection";

const app = new Hono();

app.use(
  "*",
  cors({
    origin: "*",
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allowHeaders: ["Content-Type", "Authorization", "x-authorization"],
    exposeHeaders: ["Content-Disposition"],
  }),
);

registerHonoApiRoute(healthRoute, app);

registerHonoApiRoute(detectRoute, app);
registerHonoApiRoute(getDetectionsRoute, app);
registerHonoApiRoute(getStressTestsRoute, app);
registerHonoApiRoute(analysisSummaryRoute, app);
registerHonoApiRoute(analysisMisclassificationsRoute, app);
registerHonoApiRoute(analysisConfidenceRoute, app);
registerHonoApiRoute(submitFeedbackRoute, app);
registerHonoApiRoute(stressTestRoute, app);

registerHonoApiRoute(getDetectionRoute, app);
registerHonoApiRoute(getDetectionImageRoute, app);
registerHonoApiRoute(getDetectionFeedbackRoute, app);
registerHonoApiRoute(deleteDetectionRoute, app);

const port = Number.parseInt(process.env.PORT ?? "4111", 10);

serve({ fetch: app.fetch, port }, () => {
  console.log(`Server running on port ${port}`);
});
