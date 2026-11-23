# Age-Group Classification with a Vision Transformer

An image-classification notebook that uses a pretrained Vision Transformer (ViT) checkpoint to classify facial images into three broad age groups: `YOUNG`, `MIDDLE`, and `OLD`.

## Overview

The project uses the Faces Age Detection dataset, PyTorch, and Hugging Face Transformers. The notebook prepares image/label records, applies image transformations, addresses class imbalance with random oversampling, trains and evaluates a ViT image classifier, and demonstrates inference with a Transformers pipeline.

This predicts an age **category**, not a person's exact age.

## Workflow

1. Load the dataset metadata and corresponding training images.
2. Create image/label records and map the three categories to class IDs.
3. Apply image resizing, normalization, and training-time augmentation.
4. Use random oversampling to balance the classes.
5. Load the pretrained `dima806/faces_age_detection` checkpoint and configure the classifier labels.
6. Train with Hugging Face `Trainer` and log experiments through MLflow.
7. Evaluate with accuracy, macro F1, a confusion matrix, and a classification report.
8. Save the trained model and demonstrate image-classification inference.

## Technology

- Python
- PyTorch and Torchvision
- Hugging Face Transformers and Datasets
- scikit-learn and imbalanced-learn
- Pandas, NumPy, Matplotlib
- MLflow

## Repository Contents

```text
.
├── Age Detection.ipynb
└── README.md
```

## Dataset

The notebook expects the Kaggle Faces Age Detection dataset in a Kaggle-style input directory:

- Metadata: `/kaggle/input/faces-age-detection-dataset/train.csv`
- Images: `/kaggle/input/faces-age-detection-dataset/Train/`

The dataset is not stored in this repository. Obtain it from its original source and update the paths if running outside Kaggle.

## Run the Notebook

1. Open `Age Detection.ipynb` in Kaggle or a compatible Jupyter environment.
2. Install compatible versions of PyTorch, Torchvision, Transformers, Datasets, Evaluate, MLflow, and imbalanced-learn.
3. Download and mount the dataset, then verify the paths used in the notebook.
4. Run the notebook cells in order. GPU acceleration is recommended for training.

The notebook uses a pretrained Hugging Face checkpoint and currently specifies a GPU device for the inference pipeline; change the device setting when running on CPU.

## Important Evaluation Note

The current notebook applies random oversampling **before** the train/test split. This can allow duplicated examples to appear across both partitions and make evaluation overly optimistic. For a reliable result, split the original data first, then oversample only the training partition. Re-run evaluation with a held-out test set before reporting performance numbers.

## Limitations and Responsible Use

- The model predicts coarse dataset-defined age groups, not exact ages.
- Facial appearance is not a reliable basis for determining a person's age in every context.
- Performance can vary with image quality, demographic representation, dataset labeling, and capture conditions.
- Do not use this experimental model for identity, eligibility, access, employment, or other consequential decisions.
