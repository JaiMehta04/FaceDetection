# Enhance Image Anonymization in an ETL Pipeline Using Transfer Learning

A production-grade face detection and anonymization system built as an ETL pipeline, comparing **RetinaFace** and **MTCNN** on the WIDER FACE dataset with detailed error analysis.

---

## Project Architecture

```
FaceDetection/
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── README.md
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # Centralized configuration (paths, thresholds, hyperparams)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                     # Image I/O, IoU computation, quality metrics, logging
│   ├── data_loader.py               # WIDER FACE annotation parser
│   ├── evaluation.py                # Detection matching & metrics (Precision, Recall, F1, IoU)
│   ├── quality_analysis.py          # Per-face brightness, blur, size analysis
│   ├── group_analysis.py            # Performance vs face density
│   ├── distance_analysis.py         # Pairwise face distance failure analysis
│   ├── anonymization.py             # Gaussian blur / pixelation / solid masking
│   ├── pipeline.py                  # ETL orchestrator (Extract → Transform → Load)
│   ├── visualization.py             # Matplotlib plots and detection overlays
│   │
│   └── detectors/
│       ├── __init__.py
│       ├── base.py                  # Abstract detector interface
│       ├── retinaface_detector.py   # RetinaFace wrapper (insightface or TF backend)
│       ├── mtcnn_detector.py        # MTCNN wrapper (facenet-pytorch)
│       └── factory.py               # Detector factory function
│
├── scripts/
│   ├── download_data.py             # Download WIDER FACE dataset
│   └── quick_test.py                # Smoke test with 10 images
│
├── data/
│   └── raw/                         # WIDER FACE images + annotations (downloaded)
│
└── outputs/
    ├── csv/                         # Evaluation results, quality metrics, distance analysis
    ├── plots/                       # Generated charts and visualizations
    ├── anonymized/                  # Anonymized images per detector
    └── results/                     # Aggregate results
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download WIDER FACE dataset

```bash
python scripts/download_data.py
```

This downloads the validation set (~3,226 images, ~39,708 faces) and annotations.

### 3. Verify installation

```bash
python scripts/quick_test.py
```

---

## Usage

### Full pipeline (both detectors)

```bash
python main.py
```

### Specific detector only

```bash
python main.py --detectors retinaface
python main.py --detectors mtcnn
```

### Quick test with subset

```bash
python main.py --max-images 100
```

### Custom anonymization

```bash
python main.py --anon-method pixelate
python main.py --anon-method solid
python main.py --anon-method gaussian
```

### Evaluation only (no anonymization)

```bash
python main.py --no-anonymize
```

### All options

```bash
python main.py \
    --detectors retinaface mtcnn \
    --max-images 500 \
    --iou-threshold 0.5 \
    --confidence 0.5 \
    --anon-method gaussian \
    --device auto
```

---

## Design Decisions

### Why these two models?

| Aspect | RetinaFace | MTCNN |
|---|---|---|
| Architecture | Single-stage (RetinaNet + FPN) | Three-stage cascade (P→R→O-Net) |
| Backbone | ResNet-50 | Lightweight CNNs |
| Multi-task | Classification + BBox + Landmarks | Classification + BBox + Landmarks |
| Speed | Fast (single forward pass) | Slower (3 sequential stages) |
| Small faces | Excellent (FPN multi-scale) | Moderate (limited by P-Net resolution) |
| Crowded scenes | Strong | Weaker (NMS between stages) |

### Evaluation Protocol

The matching algorithm follows the PASCAL VOC / WIDER FACE standard:

1. Sort predictions by confidence (descending)
2. Greedily match each prediction to the highest-IoU ground-truth box
3. A match is valid only if IoU > 0.5
4. Each GT box can only be matched once

This produces three categories:
- **Detected (TP)**: GT box matched with IoU > 0.5
- **Undetected (FN)**: GT box with no matching prediction
- **False Positive (FP)**: Prediction with no matching GT box

### ETL Architecture

```
EXTRACT                    TRANSFORM                         LOAD
  │                            │                               │
  ├─ Load annotations         ├─ Face detection               ├─ Quality CSV
  ├─ Load images              ├─ IoU matching                 ├─ Group-wise CSV
  └─ Validate paths           ├─ Quality analysis             ├─ Distance CSV
                              ├─ Anonymize faces              ├─ Metrics CSV
                              └─ Collect metrics              ├─ Anonymized images
                                                              └─ Plots
```

---

## Analysis Outputs

### 1. Metrics Comparison
Precision, Recall, F1 bar chart comparing both detectors.

### 2. Group-wise Analysis
Performance degradation curves as face density increases (0–10, 11–20, ..., 51+).

### 3. Quality Analysis
Box plots showing how face area, brightness, blur, and illumination affect detection:
- Small, dark faces → higher miss rate
- Blurry images → lower precision
- Well-lit, large faces → highest detection confidence

### 4. Distance Analysis
Pairwise Euclidean distances between face centers reveal:
- **Detected↔Undetected** (short distance) → NMS suppression / occlusion
- **Undetected↔Undetected** (clustered) → dense crowd failure
- **Detected↔Detected** (spread out) → isolated faces easier to detect

### 5. Anonymization Samples
Side-by-side original vs. anonymized images with three methods available.

---

## Strategies to Improve Recall in Crowded Images

1. **Multi-scale inference**: Run detection at multiple resolutions and merge results with soft-NMS
2. **Soft-NMS**: Replace hard NMS with Gaussian-weighted suppression to preserve overlapping faces
3. **Tiled inference**: Split large images into overlapping tiles, detect on each, merge with dedup
4. **Lower confidence threshold**: Accept more candidates then post-filter with a secondary classifier
5. **Ensemble/hybrid detection**: Run both RetinaFace + MTCNN, union their detections with NMS
6. **Image preprocessing**: Apply CLAHE histogram equalization for low-light, deblur for motion blur
7. **Fine-tune on domain data**: If production images differ from WIDER FACE, fine-tune the backbone
8. **Anchor-free detectors**: Consider CenterNet or FCOS which avoid anchor-NMS issues in dense scenes

### Adaptive/Hybrid Strategy

```python
# Pseudocode for adaptive detection
def adaptive_detect(image, detectors, quality_analyzer):
    blur = compute_blur(image)
    brightness = compute_brightness(image)

    if blur < BLUR_THRESHOLD:
        image = deblur(image)
    if brightness < LOW_LIGHT:
        image = enhance_lighting(image)

    # Run primary detector
    result = detectors["retinaface"].detect(image)

    # If few faces found in a large image, supplement with MTCNN
    if result.num_faces < expected_density(image):
        mtcnn_result = detectors["mtcnn"].detect(image)
        result = merge_detections(result, mtcnn_result)

    return result
```

---

## Key Metrics Computed

| Metric | Formula |
|---|---|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | 2 × P × R / (P + R) |
| IoU | Intersection / Union (per matched pair) |
| Face Brightness | Mean V-channel of face crop |
| Image Blur | Variance of Laplacian |
| Face Size | Bounding box area (pixels²) |

---

## License

This project is for educational and research purposes. The WIDER FACE dataset has its own license terms.
