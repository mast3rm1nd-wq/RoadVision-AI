# RoadVision AI: Vehicle Detection and Autopilot Safety Analytics

## Executed local end-to-end result
This repository contains an executed baseline for the capstone: a PyTorch computer-vision model for vehicle-type prediction and bounding-box localization, plus exploratory safety analytics using the Tesla deaths dataset.

## Dataset validation
- Images extracted: 5626
- Total annotation rows: 351,549
- Annotation rows matching extracted images: 17,967
- Images with usable labels: 5,626
- Modeling subset used for this local run: 293
- Usable Tesla event records: 294

## Model test metrics
- Classification accuracy: 0.568
- Mean IoU: 0.054
- Combined loss: 1.349

## Business value
RoadVision AI demonstrates how perception AI and safety analytics can support AV/ADAS programs, intelligent transportation systems, traffic monitoring, road-incident response, and executive safety-risk reviews.

## Limitation
The image labels contain multiple objects per image, while this baseline trains on the largest object per image. For a stronger portfolio version, convert the full annotations to YOLO or COCO and train a true multi-object detector.
