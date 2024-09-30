```markdown
# Breast Cancer Classification Pipeline

## Overview

This repository contains a Kubeflow Pipeline for classifying breast cancer images using the curated_breast_imaging_ddsm dataset. The pipeline leverages Google Cloud Vertex AI for training and utilizes TensorFlow Datasets for data processing.

## Pipeline Stages

The pipeline consists of the following stages:

1. **Download Dataset:** Downloads the curated_breast_imaging_ddsm dataset from TensorFlow Datasets to Google Cloud Storage.
2. **Create TF Records:** Converts the downloaded dataset into TFRecord format for efficient training.
3. **Train Model:** Trains a Flax CNN model on the preprocessed TFRecord dataset using Vertex AI custom training.

## Components

### 1. `download_dataset`

* **Description:** Downloads the CBIS-DDSM dataset.
* **Implementation:** Uses a custom container image `davidnet/cbis_ddsm_dataloader:1.0.2` to download and extract the dataset.
* **Output:** An artifact containing the downloaded dataset.

### 2. `create_tf_records`

* **Description:** Converts the downloaded dataset into TFRecords.
* **Implementation:** Uses TensorFlow Datasets to load and process the CBIS-DDSM dataset, then saves it in TFRecord format.
* **Input:** The dataset artifact from the `download_dataset` component.
* **Output:** An artifact containing the TFRecords.

### 3. `train`

* **Description:** Trains a Flax CNN model on the TFRecords.
* **Implementation:** Uses a custom container image `davidnet/flax-cnn-model:1.0.0` to train the model.
* **Input:** The TFRecords artifact from the `create_tf_records` component and various hyperparameters.
* **Output:** An artifact containing the trained model checkpoints.

## Pipeline Definition

The `breast_cancer_classification` function defines the pipeline structure, including the order of execution and dependencies between components. It also defines pipeline parameters such as training steps, evaluation frequency, batch size, learning rate, and momentum.

## Running the Pipeline

1. **Prerequisites:**
    * Google Cloud Project with Vertex AI enabled.
    * Kubeflow Pipelines installed.
    * Google Cloud SDK configured and authenticated.
2. **Compilation:** Compile the pipeline definition into a YAML file:
   ```bash
   python pipeline.py
   ```
3. **Deployment:** Deploy the compiled pipeline to your Kubeflow Pipelines instance.
4. **Execution:** Run the pipeline with desired parameter values.

## Customization

The pipeline can be customized by:

* Modifying the hyperparameters in the `breast_cancer_classification` function.
* Changing the container images used for downloading, preprocessing, and training.
* Adding or removing pipeline stages to suit specific needs.

## Note

This pipeline assumes that the required container images are available in a container registry accessible by Vertex AI. You may need to build and push these images to your own registry before running the pipeline.
```