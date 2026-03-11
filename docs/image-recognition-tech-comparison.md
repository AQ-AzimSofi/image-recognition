# Image Recognition / Object Detection Tech Comparison

Comparison of all available technologies for image object detection, covering managed APIs, local models, and custom training platforms.

> **Context**: This project uses AWS Rekognition DetectLabels for object detection. Rekognition's fixed label vocabulary (~3,000 categories) and weak support for specific product identification has led to accuracy issues. This document evaluates alternatives.

---

## Quick Recommendation

| Goal | Best Pick |
|------|-----------|
| Highest accuracy, lowest effort | **Claude Vision API** or **Gemini Flash** |
| Cheapest managed API | **Gemini 2.5 Flash-Lite** (generous free tier) |
| Stay within AWS ecosystem | **Amazon Bedrock (Claude)** |
| Need bounding boxes from managed API | **Google Cloud Vision** or **AWS Rekognition** |
| Need bounding boxes + high accuracy | **Grounding DINO** (local) or **Roboflow** |
| Full data privacy, no cloud | **Grounding DINO**, **YOLO11**, or **Florence-2** |

---

## 1. Managed Cloud APIs

These are pay-per-use services requiring no infrastructure. Best for prototyping and production without ML expertise.

### 1.1 Traditional Vision APIs (Fixed Model)

These use pre-trained fixed models. You cannot customize what they detect.

#### AWS Rekognition DetectLabels (Current)

| Item | Details |
|------|---------|
| **What it does** | Label detection + object localization on ~3,000 categories |
| **Bounding boxes** | Yes -- for object instances (people, cars, furniture, etc.) |
| **Cost** | ~$0.001/image (first 1M). Cheapest option |
| **Free tier** | 1,000 images/month for 12 months |
| **Accuracy** | Low-medium for general objects. Poor on region-specific products, small objects, niche items |
| **Data used for training** | No by default. Opt-out via AWS Organizations Policy for full guarantee |
| **Context understanding** | None -- returns generic labels only ("Bottle", not "Meiji milk carton") |
| **Strengths** | Cheap, fast, native AWS integration, returns structured bounding boxes |
| **Weaknesses** | Fixed label set, cannot identify specific products/brands, misclassifies similar shapes, misses small objects |

#### Google Cloud Vision API

| Item | Details |
|------|---------|
| **What it does** | Label detection, object localization, OCR, face detection, landmark detection |
| **Bounding boxes** | Yes -- via Object Localization ($0.00225/image). Label Detection alone has no boxes |
| **Cost** | Label: $0.0015/image, Object Localization: $0.00225/image |
| **Free tier** | 1,000 units/month per feature |
| **Accuracy** | Medium. Broader label vocabulary than Rekognition, better at web-common objects |
| **Data used for training** | No -- explicitly stated in Cloud Vision docs |
| **Context understanding** | None -- generic labels only |
| **Strengths** | Good OCR, web entity detection (can identify products by visual search), solid documentation |
| **Weaknesses** | Still a fixed label set, no brand/product-level identification from labels alone |

#### Azure AI Vision (Computer Vision)

| Item | Details |
|------|---------|
| **What it does** | Image tagging, object detection, OCR, spatial analysis, image captioning |
| **Bounding boxes** | Yes -- object detection returns pixel-level bounding boxes |
| **Cost** | ~$0.0015/image (0-1M tier) |
| **Free tier** | F0 tier with limited transactions |
| **Accuracy** | Medium. Comparable to Google Cloud Vision |
| **Data used for training** | No -- images deleted after processing |
| **Context understanding** | Limited -- captioning provides natural language descriptions but detection uses fixed categories |
| **Strengths** | Image captioning feature, good OCR, GDPR/HIPAA/SOC compliant |
| **Weaknesses** | Similar fixed-label limitations as Rekognition and Cloud Vision |

### 1.2 Multimodal LLM APIs (Context-Aware)

These use large language models with vision capabilities. They understand context, can identify specific products, read text on packaging, and describe scenes in natural language. **However, they do not natively return structured bounding box coordinates.**

#### Claude Vision API (Anthropic)

| Item | Details |
|------|---------|
| **What it does** | Image understanding via multimodal LLM. Can identify, describe, and reason about image contents |
| **Bounding boxes** | No native structured output. Can describe locations in text but not pixel-accurate boxes |
| **Cost** | Haiku 3.5: ~$0.001/image, Sonnet 4.6: ~$0.003/image, Opus 4.6: ~$0.005/image |
| **Free tier** | Small initial API credits |
| **Accuracy** | High -- understands context, reads text on labels, identifies specific products and brands |
| **Data used for training** | No -- API data is not used for training |
| **Batch discount** | 50% off with batch API (results within 24 hours) |
| **Strengths** | Best contextual understanding, reads text/brands on products, strong multilingual support, structured JSON output via prompting |
| **Weaknesses** | No bounding boxes, higher cost than traditional vision APIs, slower response |

#### Amazon Bedrock (Claude)

| Item | Details |
|------|---------|
| **What it does** | Same Claude models accessible via AWS infrastructure |
| **Bounding boxes** | No (same as Claude API) |
| **Cost** | Sonnet 4.5: $3/MTok in, $15/MTok out (same as direct API, varies by region) |
| **Free tier** | None |
| **Accuracy** | Same as Claude API |
| **Data used for training** | No -- data stays within AWS, does not leave your VPC |
| **Strengths** | AWS-native (IAM, VPC, CloudTrail), data never leaves AWS, same accuracy as direct Claude API |
| **Weaknesses** | No bounding boxes, slightly more setup than direct API, regional pricing premium |

#### OpenAI GPT-4o / GPT-4.1

| Item | Details |
|------|---------|
| **What it does** | Image understanding via multimodal LLM |
| **Bounding boxes** | No native structured output |
| **Cost** | GPT-4o: ~$0.001-0.004/image, GPT-4.1: ~$0.002-0.004/image |
| **Free tier** | Initial credits for new accounts |
| **Accuracy** | High -- similar contextual understanding to Claude |
| **Data used for training** | No -- API data not used for training by default |
| **Strengths** | Strong general vision, good at OCR, wide ecosystem |
| **Weaknesses** | No bounding boxes, contextual understanding slightly behind Claude |

#### Google Gemini API

| Item | Details |
|------|---------|
| **What it does** | Image understanding via multimodal LLM |
| **Bounding boxes** | No native structured output (can output approximate coordinates when prompted) |
| **Cost** | Gemini 2.5 Flash: ~$0.0002/image, Flash-Lite: ~$0.00006/image, 3.1 Pro: ~$0.001/image |
| **Free tier** | Generous -- free input/output tokens on Flash-Lite and 2.5 Flash models |
| **Accuracy** | Medium-high -- good contextual understanding, competitive with Claude/GPT |
| **Data used for training** | Free tier: may be used for product improvement. Paid tier (Vertex AI): not used |
| **Strengths** | Cheapest LLM option by far, generous free tier, fast |
| **Weaknesses** | Free tier data may be used for training, contextual understanding slightly weaker, no structured bounding boxes |

### 1.3 Specialized Detection APIs

#### Roboflow API

| Item | Details |
|------|---------|
| **What it does** | Object detection, classification, segmentation with pre-built and custom models |
| **Bounding boxes** | Yes -- core feature, structured JSON output |
| **Cost** | Free (public projects), Starter $49/month, Growth $299/month |
| **Free tier** | Free for public projects with limited inference |
| **Accuracy** | High for trained domains, uses state-of-the-art models (YOLO, RF-DETR) |
| **Data used for training** | Public plan: projects are publicly visible. Paid plans: private |
| **Strengths** | End-to-end platform (annotate, train, deploy), great documentation, active community |
| **Weaknesses** | Requires custom training for best results, public plan exposes your data |

#### Hugging Face Inference API

| Item | Details |
|------|---------|
| **What it does** | Host any open-source model as an API (DETR, YOLO, Florence-2, etc.) |
| **Bounding boxes** | Depends on model -- detection models (DETR, YOLO) return boxes |
| **Cost** | Free with rate limits, PRO $9/month, Endpoints from $0.03/hr (CPU) |
| **Free tier** | Free serverless inference with rate limits |
| **Accuracy** | Depends on chosen model |
| **Data used for training** | No -- HF is a hosting platform, does not train on inference data |
| **Strengths** | Access to thousands of models, flexible, no vendor lock-in |
| **Weaknesses** | Requires ML knowledge to choose and configure models, free tier has rate limits |

---

## 2. Local / Self-Hosted Models

Run entirely on your hardware. Zero data sharing, no per-request cost (only compute cost). Requires GPU for good performance.

> **Note on accuracy**: Local models are NOT necessarily weaker than cloud APIs. For object detection with bounding boxes, models like YOLO11 and Grounding DINO are state-of-the-art and can outperform traditional cloud APIs (Rekognition, Cloud Vision). However, they lack the contextual understanding of multimodal LLMs (Claude, GPT-4o) for identifying specific products or reading text.

### 2.1 Real-Time Object Detection

#### YOLO11 / YOLOv8 (Ultralytics)

| Item | Details |
|------|---------|
| **What it does** | Real-time object detection, segmentation, classification, pose estimation |
| **Bounding boxes** | Yes -- core output with class labels and confidence scores |
| **License** | AGPL-3.0 (free for open-source projects). Commercial use requires Enterprise License |
| **Pre-trained classes** | 80 categories (COCO dataset) -- person, car, bottle, cup, etc. |
| **Accuracy** | High on COCO classes. State-of-the-art speed/accuracy tradeoff |
| **GPU required** | Recommended but CPU inference possible (slower) |
| **Data used for training** | Fully local -- nothing sent anywhere |
| **Strengths** | Fastest inference, excellent documentation, easy fine-tuning, large community, Python/CLI/web export |
| **Weaknesses** | Limited to 80 pre-trained classes without fine-tuning, AGPL license restricts commercial use |

#### RT-DETR / RT-DETRv4

| Item | Details |
|------|---------|
| **What it does** | Real-time end-to-end object detection using transformers |
| **Bounding boxes** | Yes |
| **License** | Apache 2.0 (fully permissive) |
| **Accuracy** | Competitive with YOLO, better at detecting overlapping objects |
| **Data used for training** | Fully local |
| **Strengths** | No NMS post-processing needed, transformer architecture handles complex scenes better |
| **Weaknesses** | Smaller community than YOLO, fewer tutorials/resources |

#### RF-DETR (Roboflow, 2025)

| Item | Details |
|------|---------|
| **What it does** | State-of-the-art object detection, #1 on COCO benchmark (ICLR 2026) |
| **Bounding boxes** | Yes |
| **License** | Apache 2.0 (base models). PML 1.0 for Plus/XL models |
| **Accuracy** | Current SOTA on COCO |
| **Data used for training** | Fully local |
| **Strengths** | Best accuracy on standard benchmarks, easy to fine-tune via Roboflow |
| **Weaknesses** | Very new, smaller community |

### 2.2 Open-Vocabulary Detection (Zero-Shot)

These models can detect arbitrary objects specified by text prompts, without fine-tuning. Most relevant as alternatives to cloud APIs.

#### Grounding DINO

| Item | Details |
|------|---------|
| **What it does** | Open-set object detection driven by text prompts. Give it "milk carton, AirPods, keys" and it finds them |
| **Bounding boxes** | Yes -- core feature |
| **License** | Apache 2.0 |
| **Accuracy** | High for text-specified objects. Can find objects not in any fixed category list |
| **GPU required** | Yes (transformer model) |
| **Data used for training** | Fully local |
| **Strengths** | Detects anything you describe in text, no training needed, combines with SAM for segmentation |
| **Weaknesses** | Requires GPU, slower than YOLO, accuracy depends on prompt quality |

#### OWL-ViT / OWLv2 (Google)

| Item | Details |
|------|---------|
| **What it does** | Zero-shot text-conditioned object detection |
| **Bounding boxes** | Yes |
| **License** | Apache 2.0 |
| **Accuracy** | Medium-high for zero-shot detection |
| **Data used for training** | Fully local |
| **Strengths** | Simple API, works well for common objects, lightweight compared to Grounding DINO |
| **Weaknesses** | Less accurate than Grounding DINO on complex queries, smaller community |

### 2.3 Multi-Task Vision Models

#### Florence-2 (Microsoft)

| Item | Details |
|------|---------|
| **What it does** | Unified model for detection, captioning, segmentation, OCR, visual grounding |
| **Bounding boxes** | Yes -- supports object detection and region-level captioning |
| **License** | MIT (fully permissive) |
| **Model sizes** | Base (~230M params), Large (~770M params) |
| **Data used for training** | Fully local |
| **Strengths** | Single model handles many vision tasks, lightweight, excellent license, reads text |
| **Weaknesses** | Less accurate than specialized models for pure detection |

#### Meta Detectron2 / DETR

| Item | Details |
|------|---------|
| **What it does** | Object detection, segmentation, keypoint detection |
| **Bounding boxes** | Yes |
| **License** | Apache 2.0 |
| **Data used for training** | Fully local |
| **Strengths** | Battle-tested, extensive model zoo, well-documented |
| **Weaknesses** | Older architecture, surpassed by YOLO/DETR variants in speed and accuracy |

### 2.4 Other Local Tools

#### OpenCV DNN / MediaPipe

| Item | Details |
|------|---------|
| **What it does** | OpenCV: general computer vision. MediaPipe: face/hand/pose detection |
| **Bounding boxes** | Yes (OpenCV DNN can load YOLO/SSD models) |
| **License** | Apache 2.0 (both) |
| **Strengths** | Extremely lightweight, runs on CPU, good for edge devices |
| **Weaknesses** | Not designed for general object detection, limited pre-trained models |

---

## 3. Custom Training Platforms

Train your own model on your own labeled data. Highest accuracy for specific use cases, but requires data collection, labeling, and ongoing maintenance.

> **Note**: These typically require organizational approval and labeled dataset preparation.

#### AWS Rekognition Custom Labels

| Item | Details |
|------|---------|
| **Cost** | Training: billed per hour. Inference: $4.00/hr per inference unit (always-on) |
| **Free tier** | 2 training hours/month for 12 months |
| **Bounding boxes** | Yes |
| **Min data** | ~50 labeled images per class recommended |
| **Data used for training** | Only your custom model, within your AWS account |
| **Strengths** | AWS-native, no ML expertise needed, simple API |
| **Weaknesses** | Expensive inference ($4/hr always-on), slow iteration, limited model customization |

#### Google AutoML Vision

| Item | Details |
|------|---------|
| **Cost** | Training: per node-hour. Inference: per node-hour or per prediction |
| **Free tier** | First 1,000 predictions/month free (pre-built features) |
| **Bounding boxes** | Yes |
| **Data used for training** | Only your custom model |
| **Strengths** | Easy to use, good documentation, edge model export |
| **Weaknesses** | Can be expensive at scale, limited model architecture choices |

#### Azure Custom Vision

| Item | Details |
|------|---------|
| **Cost** | Training: $10/hr. Prediction: $2.00/1,000 images. Storage: $0.70/1,000 images/month |
| **Free tier** | F0: 2 projects, 5,000 training images, 10,000 predictions/month |
| **Bounding boxes** | Yes |
| **Data used for training** | Only your custom model |
| **Strengths** | Generous free tier, easy export to edge devices (ONNX, TensorFlow Lite) |
| **Weaknesses** | Limited to 500 tags per project, Microsoft ecosystem |

#### Ultralytics Hub / Roboflow Train

| Item | Details |
|------|---------|
| **Cost** | Ultralytics Hub: free tier available. Roboflow: included in paid plans |
| **Bounding boxes** | Yes |
| **Data used for training** | Your data, on their infrastructure (or export to self-host) |
| **Strengths** | Best-in-class annotation tools, one-click training, deploy anywhere |
| **Weaknesses** | Requires labeled data, Roboflow free tier makes projects public |

---

## 4. Feature Comparison Matrix

| Technology | Type | Bounding Boxes | Context-Aware | Cost/Image | Data Used for Training | Product ID Accuracy |
|-----------|------|:--------------:|:-------------:|------------|:---------------------:|:-------------------:|
| AWS Rekognition | API | Yes | No | $0.001 | No | Poor |
| Google Cloud Vision | API | Yes (Object Localization) | No | $0.002 | No | Poor |
| Azure AI Vision | API | Yes | No | $0.002 | No | Poor |
| **Claude Vision** | **API** | **No** | **Yes** | **$0.001-0.005** | **No** | **Strong** |
| Bedrock (Claude) | API | No | Yes | $0.001-0.005 | No | Strong |
| OpenAI GPT-4o | API | No | Yes | $0.001-0.004 | No | Medium |
| **Gemini Flash** | **API** | **No** | **Yes** | **$0.0002** | **Paid: No, Free: Maybe** | **Medium** |
| Roboflow | API | Yes | No | $49+/month | Paid: No | Custom training dependent |
| YOLO11 | Local | Yes | No | Free (GPU cost) | Fully local | 80 classes only (without fine-tuning) |
| Grounding DINO | Local | Yes | Partial (text prompt) | Free (GPU cost) | Fully local | Prompt dependent |
| Florence-2 | Local | Yes | Partial | Free (GPU cost) | Fully local | Medium |
| OWLv2 | Local | Yes | Partial (text prompt) | Free (GPU cost) | Fully local | Prompt dependent |
| Rekognition Custom Labels | Custom | Yes | No | $4/hr inference | Your model only | Training dependent |
| Azure Custom Vision | Custom | Yes | No | $10/hr train | Your model only | Training dependent |

---

## 5. The Bounding Box Gap

A key tradeoff exists: **multimodal LLMs (Claude, GPT-4o, Gemini) have the best accuracy and contextual understanding but do not return structured bounding boxes.** Traditional vision APIs and local detection models return bounding boxes but lack contextual understanding.

### Possible hybrid approaches:

1. **LLM for identification + Traditional API for localization**: Use Claude to identify what's in the image, then use Rekognition/Cloud Vision for bounding boxes
2. **LLM for identification + Grounding DINO for localization**: Use Claude to list objects, feed those labels to Grounding DINO to get bounding boxes (runs locally)
3. **LLM with prompted coordinates**: Some LLMs (Gemini, Claude) can output approximate bounding box coordinates when explicitly prompted, though accuracy varies
4. **Roboflow + custom model**: Train a custom model on your specific objects for both identification and bounding boxes

---

## 6. Privacy Summary

| Category | Data Sent Externally | Used for Model Training |
|----------|:-------------------:|:-----------------------:|
| Cloud APIs (paid tier) | Yes -- to provider's servers | No (all major providers) |
| Gemini API (free tier) | Yes | Possibly (Google may use for improvement) |
| Amazon Bedrock | Yes -- but stays within AWS VPC | No |
| Local models | No | No |
| Custom training (cloud) | Yes -- training data uploaded | Only your custom model |

---

*Last updated: 2026-03-08*
