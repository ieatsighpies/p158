# Migrating a Local PyTorch Training Pipeline to AWS SageMaker

This guide explains how to take a **local MONAI deep learning training workflow** and run it on **AWS SageMaker** for scalable, GPU training.

## 1. Training in Sagemaker Studio Code Editor

1. #### **Add Sagemaker Pytorch Estimator** into training notebook with:

   1. **image uri** (found & updated here: https://github.com/aws/deep-learning-containers/blob/master/available_images.md)
   2. **Pytorch version** (framework_version param, for now it is 2.7.1)
   3. **Python version** (py_version param= ""py312")
   4. **entry point**, ie. the name of your training script. Ensure that it's in the same directory as your ipynb.
2. #### Edit the **training script's directories**.

   1. For inputs (data, model, config directories), they need to be in the format: /opt/ml/input/data/`<channel-name>`
3. #### **Edit the training notebook.**

   1. ARN role, sagemaker session and S3 bucket.
   2. Uploading of training, testing, validation datasets into S3 bucket.
   3. Declaring S3 bucket directories as inputs for Sagemaker Pytorch Estimator.
   4. Copying the model artifact into the same directory in S3 bucket as the pre-trained weights.

## 2. Prerequisites

Before starting:

- **AWS Account** with SageMaker, S3, and IAM permissions.
- **AWS CLI** installed and configured:

  ```bash
  aws configure
  ```
- Python Environment with:
- ```
  pip install sagemaker boto3
  ```
