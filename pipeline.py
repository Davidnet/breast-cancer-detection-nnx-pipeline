from google_cloud_pipeline_components.v1.custom_job import (
    create_custom_training_job_from_component,
)
from kfp import dsl
from kfp.dsl import (
    Artifact,
    ContainerSpec,
    Input,
    Output,
    component,
    container_component,
)


@container_component
def download_dataset(
    dataset: Output[Artifact],
):
    return ContainerSpec(
        image="davidnet/cbis_ddsm_dataloader:1.0.2",
        command=["python", "setup.py"],
        args=[
            "--output",
            dataset.path,
        ],
    )


download_dataset_vertex = create_custom_training_job_from_component(
    download_dataset, display_name="Download Dataset", boot_disk_size_gb=600
)


@component(
    base_image="python:3.12",
    packages_to_install=[
        "tensorflow-datasets==4.9.6",
        "opencv-python-headless==4.10.0.84",
        "tensorflow==2.17.0",
    ],
)
def create_tf_records(
    dataset: Input[Artifact],
    tf_records: Output[Artifact],
):
    import os
    import tarfile

    import tensorflow_datasets as tfds

    with tarfile.open(dataset.path) as tar:
        # Extract all contents to the specified directory
        tar.extractall(path="./extracted_dataset")
    curated_breast_imaging_ddsm = tfds.builder("curated_breast_imaging_ddsm")
    curated_breast_imaging_ddsm.download_and_prepare(
        download_config=tfds.download.DownloadConfig(manual_dir="./extracted_dataset")
    )
    datasets = curated_breast_imaging_ddsm.as_dataset()
    # ~/tensorflow_datasets/curated_breast_imaging_ddsm/patches/3.0.0
    folder_path = os.path.join(
        os.path.expanduser("~"),
        "tensorflow_datasets",
        "curated_breast_imaging_ddsm",
        "patches",
        "3.0.0",
    )
    with tarfile.open(tf_records.path, "w") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))


create_tf_records_vertex = create_custom_training_job_from_component(
    create_tf_records, display_name="Create TF Records", boot_disk_size_gb=600
)


@container_component
def train(
    tf_records: Input[Artifact],
    train_steps: int,
    eval_every: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    checkpoints: Output[Artifact],
):
    return ContainerSpec(
        image="davidnet/flax-cnn-model:1.0.0",
        command=["python", "train.py"],
        args=[
            "--train-steps",
            str(train_steps),
            "--eval-every",
            str(eval_every),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--momentum",
            str(momentum),
            "--output",
            checkpoints.path,
            "--tfrecords",
            tf_records.path,
        ],
    )


train_vertex = create_custom_training_job_from_component(
    train,
    display_name="Train Model",
    boot_disk_size_gb=600,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)


@dsl.pipeline(
    name="Breast Cancer Classification",
    description="A pipeline that process and learns from images of the curated_breast_imaging_ddsm dataset",
    display_name="Breast Cancer Classification",
)
def breast_cancer_classification(
    train_steps: int = 1200,
    eval_every: int = 200,
    batch_size: int = 4,
    learning_rate: float = 0.005,
    momentum: float = 0.9,
):
    download_task = download_dataset_vertex()
    create_tf_records_task = create_tf_records_vertex(
        dataset=download_task.outputs["dataset"]
    )
    train_task = train_vertex(
        train_steps=train_steps,
        eval_every=eval_every,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
        tf_records=create_tf_records_task.outputs["tf_records"],
    )


if __name__ == "__main__":
    import kfp.compiler as compiler

    compiler.Compiler().compile(breast_cancer_classification, "pipeline.yaml")
