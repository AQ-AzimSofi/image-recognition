# Mislabel Analysis & Kaizen (改善)

## Project Context

**Domain:** Office and construction site environments.

Workers take photos of objects in the field and upload them. The AI's job is strictly **object detection/recognition** -- identify what objects are in the image.

- Input: An image
- Output: A list of detected objects (e.g., `["vacuum", "mop"]`, `["purse"]`, `["towel"]`)

The AI does NOT handle intent classification (e.g., "clean this up", "忘れ物"). Intent/action is determined by a separate system or by humans. The AI only answers: **"What objects are in this image?"**

---

## Technology Selection: AWS Rekognition

### Why Rekognition

- Fully managed service -- no model training, hosting, or infrastructure to manage for baseline usage
- Team has AWS expertise (AIF-C01, MLA-C01 certified) and company is an AWS partner
- Provides object detection with bounding boxes and confidence scores out of the box
- Supports custom model training (Custom Labels) when the built-in model is insufficient
- Pay-per-image pricing with no upfront commitment

### How DetectLabels Works

The `DetectLabels` API accepts an image (from S3 or raw bytes) and returns structured results:

```json
{
  "Labels": [
    {
      "Name": "Vacuum Cleaner",
      "Confidence": 97.5,
      "Categories": [{ "Name": "Home and Indoors" }],
      "Instances": [
        {
          "BoundingBox": {
            "Width": 0.35,
            "Height": 0.6,
            "Left": 0.2,
            "Top": 0.3
          },
          "Confidence": 97.5
        }
      ],
      "Parents": [{ "Name": "Appliance" }]
    },
    {
      "Name": "Mop",
      "Confidence": 82.1,
      "Instances": [
        {
          "BoundingBox": {
            "Width": 0.1,
            "Height": 0.7,
            "Left": 0.65,
            "Top": 0.15
          },
          "Confidence": 82.1
        }
      ],
      "Parents": [{ "Name": "Tool" }]
    }
  ]
}
```

Key fields in the response:

| Field | Description | Use for Kaizen |
|-------|-------------|----------------|
| `Name` | The detected object label | Primary output of the system |
| `Confidence` | 0-100 score of how certain the AI is | Flag low-confidence results for human review |
| `Instances[].BoundingBox` | Rectangle coordinates showing WHERE in the image the object was detected | Verify the AI looked at the correct object (Problem 2 detection) |
| `Parents` | Label hierarchy (e.g., Vacuum Cleaner → Appliance) | Understand misclassification patterns |
| `Categories` | Broader category grouping | Filter/organize results |

**Important:** Not all labels include bounding boxes. Concrete physical objects (vacuum, chair, helmet) typically have them. Abstract/scene labels (e.g., "Indoor", "Urban") do not -- they have empty `Instances` arrays.

### BoundingBox for "Right Answer, Wrong Reason" Detection

Since Rekognition is a black-box service, we cannot use techniques like Grad-CAM that require access to model internals. However, the **BoundingBox** in the response serves as a practical alternative:

- BoundingBox tells us exactly which region of the image Rekognition associated with a given label
- If the bounding box for "towel" is positioned over the background shirt instead of the actual towel, we have detected a "right answer, wrong reason" case
- This can be verified manually during the kaizen review or automated by comparing bounding box positions against known ground-truth annotations

**Verification approach:**
1. For each prediction, overlay the bounding box on the original image
2. Human reviewer (or automated check) confirms: does the box cover the correct object?
3. Log cases where the box does not match -- these are "correct label, wrong region" incidents
4. Analyze patterns in these mismatches to understand model weaknesses

### Custom Labels: Training for Special Equipment

The built-in DetectLabels model knows thousands of common objects, but it may not recognize **domain-specific equipment** such as:
- Specialized construction tools unique to a specific site
- Company-specific cleaning equipment or PPE
- Custom machinery or proprietary devices

**Amazon Rekognition Custom Labels** allows training a custom model using your own labeled images.

#### How Custom Labels Works

```
1. Collect images
   - Gather photos of the special equipment (minimum ~50 images per label recommended,
     more is better)
   - Include variety: different angles, lighting, backgrounds, partial occlusion

2. Upload & annotate in Rekognition Console
   - Upload images to an S3 bucket
   - Use the built-in labeling tool to draw bounding boxes around target objects
   - Assign labels to each bounding box

3. Train
   - Rekognition trains a custom model on your annotated dataset
   - Training time varies by dataset size (typically 30 min to several hours)
   - The model learns to detect your specific objects

4. Deploy & run inference
   - Start the model (provisions an inference endpoint)
   - Call DetectCustomLabels API with your images
   - Response format is similar to DetectLabels (labels + confidence + bounding boxes)

5. Iterate
   - Review results, identify misclassifications
   - Add more training images for problem cases
   - Retrain and redeploy
```

#### Custom Labels Considerations

| Aspect | Details |
|--------|---------|
| Minimum training data | ~50 images per label (more = better accuracy) |
| Bounding box support | Yes -- custom models can output bounding boxes if trained with them |
| Retraining | Upload additional images and retrain to improve accuracy |
| Versioning | Multiple model versions can exist; deploy the best one |
| Limitation | Requires a running inference endpoint (cost implications -- see below) |

#### Combined Strategy: DetectLabels + Custom Labels

For best results, use both APIs together:

```
Image
  ├── DetectLabels API    → common objects (vacuum, mop, helmet, purse, etc.)
  └── DetectCustomLabels  → domain-specific objects (custom equipment, proprietary tools)

Merge results → final object list with confidence scores and bounding boxes
```

---

## Cost Analysis (Image Only, No Video)

### DetectLabels API (Built-in Model)

Pricing is per-image, tiered by monthly volume:

| Monthly Volume | Cost per Image | Example Monthly Cost |
|---------------|----------------|---------------------|
| First 5,000 (Free Tier*) | $0.00 | $0.00 |
| Up to 1M images | $0.001 | 10,000 images = $10.00 |
| 1M - 5M images | $0.0008 | 2M images = $1,600 |
| 5M - 35M images | $0.0006 | -- |
| 35M+ images | $0.00025 | -- |

*Free Tier: 5,000 images/month for the first 12 months of the AWS account.

**Realistic scenario estimates:**

| Use Case | Images/Month | Monthly Cost |
|----------|-------------|-------------|
| Small office pilot | 500 | ~$0.50 |
| Single construction site | 5,000 | ~$5.00 |
| Multi-site deployment (10 sites) | 50,000 | ~$50.00 |
| Enterprise scale | 500,000 | ~$500.00 |

### Custom Labels (Custom-Trained Model)

Custom Labels has a fundamentally different cost structure because it requires a **running inference endpoint**:

| Component | Cost |
|-----------|------|
| Training | Free Tier: 2 hours/month (first 12 months). After: varies by dataset/training time |
| Inference | **$4.00 per inference unit per hour** |

**What this means in practice:**
- The inference endpoint must be running to process images
- 1 inference unit handles ~5 images/second (varies by model complexity)
- If you run the endpoint 24/7: **$4 x 24 x 30 = $2,880/month**
- If you run it on-demand (e.g., 2 hours/day): **$4 x 2 x 30 = $240/month**

**Cost optimization strategies for Custom Labels:**
- Batch processing: collect images throughout the day, spin up the endpoint once, process all images, shut down
- Schedule-based: only run during business hours
- Use AWS Lambda + EventBridge to automate start/stop
- Keep Custom Labels only for objects that DetectLabels cannot handle; use DetectLabels for everything else

### Total Cost Comparison

| Approach | Monthly Cost (10,000 images) | Pros | Cons |
|----------|------------------------------|------|------|
| DetectLabels only | ~$10 | Cheapest, zero setup | Cannot detect custom equipment |
| DetectLabels + Custom Labels (on-demand 2hr/day) | ~$250 | Handles everything | Higher cost, endpoint management |
| DetectLabels + Custom Labels (batch 1hr/day) | ~$130 | Good balance | Slight delay in processing |

---

## Core Problems

### Problem 1: Mislabeling (誤ラベル)

The AI identifies an object incorrectly.

**Examples:**
- A vacuum cleaner is labeled as "mop"
- A purse is labeled as "backpack"
- A hard hat is labeled as "bucket"

**Why this matters:** Downstream systems and humans rely on correct object identification. Wrong labels cascade into wrong actions.

**Detection with Rekognition:**
- Review the `Confidence` score -- low confidence predictions are more likely to be wrong
- Set a confidence threshold (e.g., 80%) below which results are flagged for human review
- Track and log all predictions with their confidence scores for trend analysis

### Problem 2: Right Answer, Wrong Reason (正解・誤根拠)

The AI arrives at the correct label, but based on the wrong object or feature in the image.

**Example:**
- A photo of a towel with a deer print on it. The AI does not recognize the towel itself, but notices a shirt in the background. It labels the image as "towel" based on the shirt's presence -- correct answer, but wrong reasoning.

**Why this matters:**
- Fragile correctness: the model will fail on similar images where the "lucky" background object is absent
- Cannot trust the model's consistency
- Makes improvement/debugging much harder because accuracy metrics look fine on the surface

**Detection with Rekognition:**
- Compare the BoundingBox coordinates against the expected object location
- If the bounding box for "towel" covers a region where the towel is NOT located, flag it
- Build a review pipeline where sampled predictions are visually verified (bounding box overlay on image)

---

## Analysis Dimensions

### Why does mislabeling happen?

| Category | Description | Example |
|----------|-------------|---------|
| Object confusion | AI confuses visually similar objects | Backpack vs. purse |
| Context blindness | AI ignores scene context that could help identification | Failing to use surrounding clues to distinguish similar objects |
| Background noise | Irrelevant objects influence the prediction | Shirt in background causes "towel" label |
| Occlusion | Target object is partially hidden | Half-visible item behind a door |
| Multiple objects | Image contains several objects; AI must detect all of them correctly | Photo with a milk carton and cookies should output both |
| Low image quality | Blurry, dark, or poorly framed photos | Construction site in low light |
| Unknown object | Object is not in Rekognition's built-in label set | Specialized proprietary tool |

### How to detect "Right Answer, Wrong Reason" with Rekognition

Since Rekognition is a black-box (no access to model internals), Grad-CAM is not available. Instead:

1. **BoundingBox Verification** - Primary method. Overlay the returned bounding box on the original image. If the box covers the wrong region, the model is looking at the wrong thing.

2. **Occlusion Testing** - Crop or mask different regions of the image and re-run DetectLabels. If masking the target object does not change the prediction, the model relied on something else.

3. **Confidence Delta Analysis** - Run the same image with and without the target object region. Compare confidence scores. A small delta suggests the target object is not driving the prediction.

---

## Open Questions

- What label taxonomy are we using? Is it fixed or dynamic?
- How many categories/labels exist in the system?
- Is there a human review step in the current workflow?
- What is the acceptable error rate for production?
- What domain-specific equipment needs Custom Labels training?
- What is the expected image volume per month?
- How real-time does detection need to be? (Affects Custom Labels endpoint strategy)

---

## Potential Kaizen Strategies

### Short-term
- Integrate DetectLabels API and log all predictions with confidence scores and bounding boxes
- Set confidence threshold; flag low-confidence results for human review
- Build a simple review UI to overlay bounding boxes on images for manual verification
- Start collecting a "ground truth" dataset of correctly labeled images

### Mid-term
- Identify objects that DetectLabels consistently gets wrong or cannot detect
- Train Custom Labels model for those specific objects
- Implement occlusion testing for sampled predictions to detect "right answer, wrong reason"
- Build a feedback loop: human corrections feed into Custom Labels retraining data
- Curate a "tricky cases" test set for edge cases (visually similar objects, cluttered backgrounds, multiple objects)

### Long-term
- Automated "right answer, wrong reason" detection pipeline
- Continuous model evaluation dashboard (accuracy trends, confidence distributions, common failure modes)
- Active learning: model flags uncertain cases, human labels them, Custom Labels retrains

---

## Notes / Brainstorming Log

_Add ongoing thoughts and discussion points below._
