## Business Problem

Autonomous vehicles and advanced driver-assistance systems depend on accurate scene perception. Vehicles, cyclists, pedestrians, and other road objects must be detected and localized reliably to support safer navigation and better road awareness.

At the same time, transportation leaders need data-driven insight into safety events involving automation features. Understanding fatality patterns, Autopilot-related indicators, vehicle models, locations, and collision types can help support safety reviews, product risk analysis, and public-sector transportation planning.

This project addresses both needs by combining computer vision with safety analytics.

---

## Dataset Description

This project uses two datasets:

### 1. Road Image Dataset

The image dataset contains road-scene images with annotation labels. Each annotation includes:

* Image ID
* Object class
* Bounding-box coordinates:

  * `xmin`
  * `ymin`
  * `xmax`
  * `ymax`

The dataset includes multiple annotations per image. For this baseline version, the largest bounding box per image is selected as the representative object for model training.

### 2. Tesla Fatality Dataset

The Tesla fatality dataset contains accident-event records with fields such as:

* Case number
* Year and date
* Country and state
* Accident description
* Number of deaths
* Tesla driver death indicator
* Tesla occupant death indicator
* Cyclist/pedestrian involvement
* Other vehicle involvement
* Tesla model
* Verified Tesla Autopilot deaths
* Source and notes

This dataset is used for exploratory safety analysis, not causal claims.

---

## Modeling Approach

The computer vision model is built using PyTorch.

The baseline model uses a custom CNN architecture with:

* Convolutional layers
* Batch normalization
* ReLU activation
* Max pooling
* Adaptive average pooling
* Dropout
* A classification head for object type prediction
* A regression head for bounding-box prediction

The model learns two tasks at the same time:

1. **Classification:** Predict the vehicle/object class.
2. **Localization:** Predict normalized bounding-box coordinates.

The total loss combines:

* Cross-entropy loss for classification
* Smooth L1 loss for bounding-box regression

---

## Project Workflow

The end-to-end workflow includes:

1. Locate and extract the image dataset.
2. Load and clean annotation labels.
3. Match labels to available images.
4. Normalize bounding-box coordinates.
5. Select a representative object per image.
6. Split data into training, validation, and test sets.
7. Train the CNN model.
8. Evaluate classification accuracy and bounding-box IoU.
9. Generate sample prediction images.
10. Clean and analyze Tesla fatality records.
11. Export metrics, charts, cleaned data, and model artifacts.

---

## Results

The project exports model and analysis outputs to the `outputs/` folder, including:

* Training, validation, and test metrics
* Model performance history
* Class distribution charts
* Training loss curve
* Validation performance curve
* Sample prediction images
* Cleaned Tesla fatality dataset
* Tesla safety EDA summary
* Fatality distribution charts

Key model metrics include:

* Classification accuracy
* Mean Intersection over Union, also called mean IoU
* Combined model loss

Mean IoU measures how well the predicted bounding box overlaps with the true bounding box. A higher value indicates better localization performance.

---

## Safety Analytics

The Tesla safety EDA explores questions such as:

* How many accident events occurred by year?
* Which countries and states appear most frequently?
* What is the distribution of deaths per accident?
* How often did Tesla drivers die?
* How often did Tesla occupants die?
* How often were cyclists or pedestrians involved?
* How often were other vehicles involved?
* Which Tesla models appear most often in the records?
* What is the distribution of verified Tesla Autopilot deaths?

This analysis is descriptive. It identifies patterns in the available dataset but does not prove that Autopilot caused or prevented any accident.

---

## Business Value

RoadVision AI demonstrates how AI can support transportation and autonomous systems use cases, including:

* Autonomous vehicle perception
* Advanced driver-assistance system development
* Traffic monitoring
* Vehicle counting
* Road incident response
* Transportation safety analysis
* Safety-risk reporting
* Smart city infrastructure planning

The project shows how technical model outputs can be connected to business and safety decision-making.

---

## Limitations

This is a baseline project, not a production-grade autonomous driving system.

Current limitations include:

* The image dataset contains multiple objects per image, but this baseline uses only the largest bounding box per image.
* The model performs single-object detection rather than full multi-object detection.
* The CNN is trained on resized 64x64 images, which reduces visual detail.
* The model is intended for learning and portfolio demonstration, not real-world vehicle deployment.
* The Tesla fatality dataset supports descriptive analysis only and should not be used to make causal safety claims.
* Autopilot-related fields depend on the accuracy and completeness of the source data.

---

## Next Steps

Recommended improvements include:

1. Convert the full annotation dataset to YOLO or COCO format.
2. Train a true multi-object detector such as YOLOv8, Faster R-CNN, or SSD.
3. Increase image resolution for better localization.
4. Add precision, recall, F1 score, and mean average precision.
5. Add confusion matrix analysis for object classification.
6. Improve bounding-box visualization outputs.
7. Add more robust safety analytics by normalizing accidents against miles driven, vehicle counts, or exposure data.
8. Build an interactive dashboard for safety trends and model results.
9. Package the project with a cleaner `src/` folder structure.
10. Add a Streamlit or FastAPI demo for inference.

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd roadvision-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the datasets

Place the following files in the project root folder:

```text
Images.zip
labels.csv
Tesla - Deaths.csv
```

The script will attempt to locate these files automatically. If found, it will extract the images and create the required project folders.

### 4. Run the notebook or script

For the notebook version, open:

```text
RoadVision_AI_Capstone.ipynb
```

For the script version, run:

```bash
python run_end_to_end_local.py
```

### 5. Review outputs

Generated files will be saved under:

```text
outputs/
outputs/figures/
outputs/sample_predictions/
models/
```

The trained PyTorch model is saved in:

```text
models/roadvision_tiny_detector.pt
```

---

## Repository Structure

```text
roadvision-ai/
│
├── data/
│   └── Images/
│
├── models/
│   └── roadvision_tiny_detector.pt
│
├── outputs/
│   ├── figures/
│   ├── sample_predictions/
│   ├── model_metrics.json
│   ├── sample_predictions.csv
│   ├── tesla_deaths_cleaned.csv
│   └── tesla_eda_summary.json
│
├── README.md
├── requirements.txt
└── run_end_to_end_local.py
```

---

## Portfolio Summary

RoadVision AI demonstrates an end-to-end machine learning workflow that connects computer vision, transportation safety analytics, and business-value framing. The project shows practical experience with deep learning, image preprocessing, bounding-box regression, model evaluation, exploratory data analysis, and autonomous systems use-case development.
