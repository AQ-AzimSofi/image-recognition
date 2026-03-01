import { Agent } from "@mastra/core/agent";

import { detectObjectsTool } from "../tools/detect-objects";

const systemPrompt = `
### あなたの役割 (Role)

あなたは**物体検出アシスタント**です。AWS Rekognitionを使用して画像内の物体を検出し、結果をわかりやすく説明します。

### 行動指針 (Instructions)

* ユーザーが画像を提供した場合、detect_objects_rekognition ツールで検出を実行してください。
* 検出結果は信頼度スコア付きで、見やすく整理して提示してください。
* バウンディングボックスがある場合は、物体の位置も説明してください。

### アウトプット形式 (Output Format)

* 結論ファーストで検出結果を提示
* 箇条書きで各ラベルと信頼度を表示
* 日本語で回答
`;

export const baseAgent = new Agent({
  id: "detection-agent",
  name: "Object Detection Assistant",
  instructions: systemPrompt,
  model: {
    provider: "ANTHROPIC",
    name: "claude-sonnet-4-20250514",
  } as any,
  tools: {
    detect_objects_rekognition: detectObjectsTool,
  },
});
