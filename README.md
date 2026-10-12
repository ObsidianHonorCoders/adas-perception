# ADAS Perception

An experimental repository for object-detection models for Advanced Driver Assistance Systems (ADAS). The initial baseline is **YOLO11n (`v0`)**. Subsequent versions will explore automotive-domain fine-tuning, model comparison, evaluation, and deployment-oriented development.

> **Safety disclaimer:** This repository is a research and development project, not a safety-certified ADAS system. Its outputs must not be used to make safety-critical driving decisions.

## Motivation

ADAS perception systems must identify relevant road users and objects under diverse driving conditions, including varying illumination, object sizes, occlusions, and traffic densities.

This project explores how pretrained object-detection models can be adapted and evaluated for automotive perception tasks using publicly available driving datasets.

The project emphasizes reproducible experiments, measurable model performance, error analysis, software quality, and maintainable ML development workflows. It is intended to demonstrate practical skills in computer vision, dataset engineering, model fine-tuning, ML evaluation, and MLOps-oriented development.

The initial scope focuses on camera-based 2D object detection. Distance estimation, object tracking, and ADAS warning logic are potential extensions rather than initial requirements.

## Project Objectives

1. **Establish a reproducible baseline:** Run a pretrained object-detection model on automotive images and document its performance.
2. **Build an automotive dataset pipeline:** Prepare, validate, and version image datasets and annotations in YOLO format.
3. **Adapt models to driving scenarios:** Fine-tune pretrained detectors on automotive data and investigate improvements for selected object classes and challenging conditions.
4. **Measure detection performance:** Evaluate mAP, precision, recall, per-class performance, inference latency, and representative failure cases.
5. **Enable fair model comparisons:** Compare model versions using consistent dataset splits, evaluation settings, and documented configurations.
6. **Develop reliable ML tooling:** Provide reusable training, evaluation, prediction, and analysis scripts with automated tests and configuration management.
7. **Prepare for deployment-oriented development:** Investigate model export, inference optimization, and reproducible execution on CPU or GPU hardware.
8. **Explore ADAS integration:** Establish a perception interface that could support future functional analytics or warning-system prototypes.

## Definition of Done

The initial portfolio milestone will be considered complete when:

- [ ] A documented baseline model runs inference on automotive images.
- [ ] The dataset preparation pipeline produces valid YOLO-format images and annotations.
- [ ] Dataset splits are documented and avoid known recording-level data leakage.
- [ ] Training and evaluation can be reproduced using documented configurations.
- [ ] Pretrained and fine-tuned models are evaluated using the same held-out test set.
- [ ] Results include overall and per-class metrics, inference performance, and representative false positives and false negatives.
- [ ] Model versions, checkpoint sources, dataset revisions, and experiment configurations are recorded.
- [ ] Automated tests cover critical dataset-processing and evaluation functionality.
- [ ] The README documents environment setup, execution, results, and limitations.

Improvement over the baseline is an experimental objective, not a guaranteed outcome. All performance claims must be supported by the documented evaluation protocol.

## Current Status

- [x] Initial repository scaffold and README created.
- [x] Initial model selected: YOLO11n (`v0`).
- [x] YOLO11n pretrained checkpoint downloaded to `models/pretrained/`.
- [x] Initial dataset structure and YOLO annotation requirements documented.
- [x] Training, evaluation, prediction, and analysis command interfaces documented.
- [ ] Baseline inference verified on representative automotive images.
- [ ] End-to-end training and evaluation workflow verified.
- [ ] BDD100K object-detection subset selected and prepared.
- [ ] Fine-tuned model evaluated against the pretrained baseline.

A checked item indicates completed work. Creating a script or directory does not, by itself, mean the corresponding workflow has been verified.

## Requirements

- Windows 10/11
- Python 3.10
- Git
- Internet access for downloading pretrained weights and datasets
- Disk space for images, annotations, checkpoints, and experiment outputs

A CPU-only computer is sufficient for environment setup, dataset preparation, inference experiments, testing, and evaluation on manageable subsets. Full training may be slow on CPU; a cloud GPU is recommended for larger fine-tuning experiments.

### Environment Setup

Run the following command from the repository root:

```bat
pyenv\setup_env.bat
```

This script creates `pyenv\.venv`, installs dependencies from `pyenv\requirements.txt`, and downloads the YOLO11n pretrained checkpoint to `models/pretrained/`.

Activate the environment:

```bat
pyenv\.venv\Scripts\activate
```

Verify the installation and available hardware acceleration:

```python
import torch
import ultralytics

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Ultralytics:", ultralytics.__version__)
```

PyTorch/CUDA installation depends on the hardware and installed drivers. The requirements install the configured default PyTorch build. If GPU acceleration is required, install the appropriate build using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).

Ultralytics uses CUDA when a compatible GPU environment is available. On a CPU-only system, training and inference can run on the CPU, but training performance may be limited.

## Dataset Setup

### Supported Datasets

#### BDD100K — Primary Automotive Dataset

BDD100K is the primary candidate for automotive-domain fine-tuning because it contains diverse driving scenes and annotations for road users and other traffic-related objects.

- [Official BDD100K website](https://bdd-data.berkeley.edu/)
- [Official BDD100K repository](https://github.com/bdd100k/bdd100k)
- [Ultralytics BDD100K dataset documentation](https://docs.ultralytics.com/datasets/detect/bdd100k/)

Start with a manageable subset rather than downloading and processing the entire dataset immediately.

#### COCO — Pretraining Reference

COCO is relevant as a reference dataset for general-purpose object detection and pretrained model development.

- [Official COCO downloads](https://cocodataset.org/#download)
- [Explore COCO images](https://cocodataset.org/#explore)
- [Ultralytics COCO documentation](https://docs.ultralytics.com/datasets/detect/coco/)

Downloading COCO images is not required merely to use pretrained YOLO11n weights. The pretrained checkpoint already contains learned model parameters.

### Dataset Preparation Strategy

The initial dataset preparation workflow should:

- [ ] Review dataset access conditions, licenses, and annotation schemas.
- [ ] Download the required images and object-detection annotations.
- [ ] Select a representative subset with diverse classes and driving conditions.
- [ ] Convert annotations into YOLO format.
- [ ] Validate image-label correspondence and bounding-box coordinates.
- [ ] Check for missing labels, invalid annotations, and class-mapping errors.
- [ ] Preserve appropriate original splits and prevent recording-level data leakage.
- [ ] Document class frequencies, object-size distributions, and preprocessing decisions.

### Expected Dataset Structure

Place the processed dataset under `data/processed/`:

```text
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image should have a corresponding YOLO-format `.txt` annotation file. Each annotation row contains:

```text
class_id x_center y_center width height
```

Coordinates must be normalized to the range `[0, 1]`.

For driving footage, split by recording or drive whenever possible instead of randomly splitting adjacent frames. This reduces data leakage and overly optimistic evaluation results.

### Dataset Configuration

Update `configs/dataset.yaml` with the actual dataset paths and class names.

Example:

```yaml
path: data/processed

train: images/train
val: images/val
test: images/test

names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
```

This is an illustrative class mapping. Verify that the annotation conversion uses exactly the same class IDs and ordering. Do not assume that the source dataset's category IDs match YOLO class IDs.

## Model and Experiment Strategy

### Initial Baseline: YOLO11n (`v0`)

YOLO11n is the initial pretrained baseline. Its purpose is to establish a working and measurable reference before changing the model architecture.

The initial baseline should:

- Load the pretrained checkpoint.
- Run inference on representative automotive images.
- Save annotated predictions.
- Establish baseline metrics on a documented dataset split.
- Record model provenance, configuration, software versions, and hardware.

### Fine-Tuning Strategy

Fine-tuning will initially use YOLO11n and the prepared BDD100K subset.

The workflow will compare the pretrained model with the automotive-domain fine-tuned model. The experiment will investigate whether fine-tuning improves selected classes or difficult scenarios without unacceptable regression in other classes.

Potential areas of investigation include:

- Pedestrian and rider detection.
- Small and distant vehicles.
- Traffic lights and traffic signs.
- Occlusion and varying illumination.
- Class imbalance and data augmentation.

Training parameters should be selected using validation results. The test split must remain held out for final comparisons.

### Hardware Strategy

Local CPU execution will be used for:

- Environment and pipeline verification.
- Small smoke tests.
- Dataset preparation and validation.
- Inference and evaluation on manageable subsets.
- Unit and integration testing.

Cloud GPU execution may be used for:

- Complete training experiments.
- Larger dataset subsets.
- Hyperparameter experiments.
- Model comparisons requiring repeated training runs.

Save checkpoints and experiment artifacts persistently when using temporary cloud environments. Keep dataset and model paths configurable rather than hardcoding machine-specific locations.

## Roadmap and Next Steps

### Phase 1 — Baseline Verification (`v0`)

**Scope:** Verify that the existing YOLO11n checkpoint and repository scripts work correctly.

- [ ] Verify the Python environment and dependency versions.
- [ ] Confirm the checkpoint exists and loads successfully.
- [ ] Run inference on representative automotive images.
- [ ] Verify predicted classes, confidence scores, and bounding boxes.
- [ ] Run a small training smoke test.
- [ ] Verify evaluation outputs and metric files.
- [ ] Verify prediction and analysis scripts.
- [ ] Record checkpoint provenance and baseline configuration.

**Deliverable:** A verified inference pipeline and reproducible baseline configuration.

### Phase 2 — Automotive Dataset Preparation

**Scope:** Prepare a manageable BDD100K object-detection subset.

- [ ] Review dataset licensing, access requirements, and annotation format.
- [ ] Download the required images and detection annotations.
- [ ] Select a representative subset.
- [ ] Convert annotations to YOLO format.
- [ ] Validate labels and class mappings.
- [ ] Establish train, validation, and test splits.
- [ ] Prevent known recording-level leakage.
- [ ] Generate dataset statistics.
- [ ] Document preprocessing and dataset revision.

**Deliverable:** A validated dataset compatible with the training pipeline.

### Phase 3 — Baseline Training and Evaluation

**Scope:** Establish a quantitative reference for the automotive dataset.

- [ ] Configure dataset paths, class names, image size, batch size, epochs, and seed.
- [ ] Run a small CPU training smoke test.
- [ ] Execute a baseline experiment on suitable hardware.
- [ ] Evaluate the pretrained model on the selected evaluation data.
- [ ] Record mAP@50 and mAP@50–95.
- [ ] Record precision, recall, and per-class metrics.
- [ ] Measure inference latency and throughput.
- [ ] Save training curves, metrics, configurations, and checkpoints.
- [ ] Analyze representative false positives and false negatives.

**Deliverable:** A baseline report and a documented set of failure cases.

### Phase 4 — Fine-Tuning and Target-Specific Improvements

**Scope:** Investigate the effect of automotive-domain fine-tuning.

- [ ] Fine-tune YOLO11n using the prepared BDD100K subset.
- [ ] Compare pretrained and fine-tuned models.
- [ ] Analyze per-class performance.
- [ ] Investigate small objects, distant road users, occlusion, and lighting conditions.
- [ ] Experiment with training duration, image size, augmentation, and class balancing.
- [ ] Use validation results to select configurations.
- [ ] Evaluate the selected model on the fixed test set.
- [ ] Document improvements, regressions, and negative results.

**Deliverable:** A fine-tuned model and a comparative report supported by quantitative evaluation.

### Phase 5 — Model Benchmarking and Versioning

**Scope:** Establish a reproducible framework for comparing candidate models.

- [ ] Define a common evaluation interface.
- [ ] Record architecture, checkpoint source, dataset revision, hyperparameters, seed, software versions, and hardware.
- [ ] Compare YOLO11n against candidate architectures, potentially including YOLO26n.
- [ ] Use identical dataset splits and evaluation settings.
- [ ] Compare accuracy, per-class recall, latency, throughput, and resource requirements.
- [ ] Define model-selection criteria based on measured results.
- [ ] Consider MLflow for experiment tracking and DVC for dataset or artifact versioning.
- [ ] Keep large datasets and checkpoints outside Git.

**Deliverable:** A model comparison report and traceable experiment history.

### Phase 6 — Software Quality and ML Pipeline Automation

**Scope:** Improve maintainability, testability, and reproducibility.

- [ ] Add unit tests for annotation conversion and dataset validation.
- [ ] Add tests for metric aggregation and configuration validation.
- [ ] Add integration tests using a small fixture dataset.
- [ ] Standardize experiment outputs and metadata.
- [ ] Add linting and automated tests through continuous integration.
- [ ] Separate data preparation, training, inference, and evaluation into reusable components.
- [ ] Document how to reproduce an experiment from a clean environment.

**Deliverable:** A tested and maintainable ML development pipeline.

### Phase 7 — Inference Optimization and Deployment Prototype

**Scope:** Investigate deployment-oriented inference without claiming production readiness.

- [ ] Benchmark inference on the target CPU and, if available, a GPU.
- [ ] Investigate export to a supported format such as ONNX.
- [ ] Compare native and exported outputs for consistency.
- [ ] Measure latency, throughput, memory usage, and model size.
- [ ] Implement a simple interface that returns structured detections.
- [ ] Add input validation and inference tests.
- [ ] Document hardware requirements and limitations.

**Deliverable:** A reproducible inference prototype and deployment benchmark.

### Phase 8 — ADAS-Oriented Perception Analytics

**Scope:** Explore how object detections could support future ADAS analytics.

- [ ] Define a stable detection schema containing class, confidence, bounding box, and frame metadata.
- [ ] Investigate tracking across consecutive frames.
- [ ] Explore localization and temporal consistency when appropriate calibration and data are available.
- [ ] Define the additional inputs required for a future BSD or FCW prototype.
- [ ] Separate perception evaluation from warning-logic evaluation.
- [ ] Document limitations related to distance estimation, sensor calibration, temporal behavior, and safety validation.

**Deliverable:** An experimental perception interface for future ADAS analytics research.

This phase does not establish suitability for real-vehicle use or safety-critical warnings.

## Model Versioning Strategy

Project versions represent development milestones, not guaranteed performance improvements.

| Version | Scope | Main deliverable |
|---|---|---|
| `v0` | YOLO11n pretrained baseline | Initial checkpoint and inference pipeline |
| `v1` | Automotive dataset baseline | Prepared BDD100K subset and baseline evaluation |
| `v2` | Automotive-domain fine-tuning | Fine-tuned YOLO11n and comparative metrics |
| `v3` | Model benchmarking | Comparison with candidate models, potentially YOLO26n |
| `v4` | ML pipeline engineering | Automated tests and experiment tracking |
| `v5` | Inference optimization | Export and deployment-oriented benchmark |
| `v6` | ADAS analytics extension | Experimental temporal analysis and integration interface |

Each release should document its configuration, artifacts, dataset revision, and evaluation results.

## Repository Layout

```text
configs/                  Dataset and model training settings
data/raw/                 Original data (ignored by Git)
data/interim/             Temporary/preprocessed data (ignored by Git)
data/processed/           YOLO-format dataset (ignored by Git)
models/pretrained/        Downloaded starting weights (ignored by Git)
models/checkpoints/       Exported/final checkpoints (ignored by Git)
notebooks/                Exploratory analysis
reports/figures/          Generated plots (ignored by Git)
reports/metrics/          Generated metric summaries (ignored by Git)
runs/                     Training/evaluation/prediction outputs (ignored)
scripts/                  Download, train, evaluate, predict, and analyze CLIs
src/adas_perception/models/
                          Custom model implementations
tests/                    Unit and integration tests
pyenv/                    Environment setup and dependency configuration
```

## Commands

Run from the repository root with the virtual environment activated.

### Download Pretrained Weights

```bat
python scripts\download_weights.py
```

### Train

```bat
python scripts\train.py
```

### Evaluate

```bat
python scripts\evaluate.py --split test
```

### Predict

```bat
python scripts\predict.py --source path\to\image-or-video.mp4
```

### Analyze Training Results

```bat
python scripts\analyze.py
```

Training writes checkpoints and `results.csv` under `runs/train/`.

Evaluation artifacts are saved under `runs/eval/`. Prediction images and videos are saved under `runs/predict/`. Analysis summarizes training results under `reports/metrics/`.

Verify the actual script arguments and output paths against the current implementation; this README documents the intended interfaces.

## Evaluation Protocol

All model comparisons should use the same documented test set and consistent evaluation settings.

Report, where applicable:

- mAP@50.
- mAP@50–95.
- Precision and recall.
- Per-class precision, recall, and mAP.
- False positives and false negatives.
- Performance on small or difficult-to-detect objects.
- Inference latency and throughput.
- Memory usage and model size.

Use validation results to tune hyperparameters. Reserve the test split for final comparisons rather than repeatedly adjusting the model based on test results.

For driving footage, split by recording or drive where possible to reduce data leakage between adjacent frames.

## Model Versioning and Experiment Tracking

Treat each experiment as an immutable record. Store:

- Model architecture and checkpoint source.
- Dataset revision and split definitions.
- Training configuration and random seed.
- Software and library versions.
- Hardware and execution environment.
- Evaluation metrics and generated reports.
- Checkpoint or artifact references.

Keep large datasets and model checkpoints out of Git. Consider DVC or an artifact store as experiment history grows.

## Licensing

Ultralytics models and code are distributed under AGPL-3.0 by default, with alternative commercial licensing available. Review the terms applicable to the implementation, weights, dependencies, and intended distribution or deployment before using the project in a proprietary product.

Review BDD100K and COCO dataset terms separately before redistributing images, annotations, or derived artifacts.

## Out of Scope

The initial project does not aim to:

- Develop a complete autonomous-driving system.
- Implement production-ready BSD, FCW, or other safety-critical warning logic.
- Claim automotive functional-safety certification or regulatory compliance.
- Guarantee detection reliability under all environmental conditions.
- Estimate accurate object distance from a single image without additional methods and assumptions.
- Deploy the model directly into a production vehicle.

The focus is on reproducible perception experiments, software engineering quality, measurable model performance, and a foundation for future ADAS research.