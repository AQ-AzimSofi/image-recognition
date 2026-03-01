import * as iam from "aws-cdk-lib/aws-iam";

export function grantRekognitionPermissions(grantable: iam.IGrantable) {
  grantable.grantPrincipal.addToPrincipalPolicy(
    new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: ["rekognition:DetectLabels"],
      resources: ["*"],
    }),
  );
}

// Usage in sitelens-ai-agent after migration:
//
// Lambda (infra/lib/constructs/lambda-backend-construct.ts):
//   After line ~397 (after secret grants):
//   grantRekognitionPermissions(this.function);
//
// ECS (infra/lib/constructs/ecs-backend-construct.ts):
//   After line ~577 (after secret grants):
//   grantRekognitionPermissions(taskRole);
