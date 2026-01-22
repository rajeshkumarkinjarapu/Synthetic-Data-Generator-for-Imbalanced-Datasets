# Synthetic-Data-Generator-for-Imbalanced-Datasets
### This project was developed as a technical assessment for the Reaidy.io selection process.

## Overview
Imbalanced datasets are a common challenge in real-world machine learning problems. When one class significantly outnumbers another, standard models tend to perform poorly on the minority class, even if overall accuracy looks high.

This project focuses on data-level solutions to this problem. It implements and evaluates different oversampling techniques and demonstrates their impact on model performance using a complete, reproducible machine learning pipeline.

## Requirements
The project was developed and tested using the following environment:

Python ≥ 3.8

NumPy ≥ 1.21

Pandas ≥ 1.3

Matplotlib ≥ 3.5

scikit-learn ≥ 1.0

imbalanced-learn ≥ 0.10

## Notebook Descriptions
### 1. synthetic_generator.ipynb
What is implemented here:

Creation of a synthetic imbalanced dataset to simulate real-world scenarios

Analysis of class imbalance and its impact

Design of a custom oversampling strategy based on neighbor interpolation

Careful handling of edge cases (small minority class, reproducibility)

Robust implementation that avoids environment-specific instability

### Key outcome:
A reusable, stable oversampling function that increases minority samples while preserving local data structure

Clear documentation of design decisions and limitations

This notebook is intentionally focused only on data-level operations, keeping it modular and reusable.

### 2. demo_pipeline.ipynb

What is done here:

Creation of an imbalanced classification dataset

Train–test split with stratification

Baseline model training without oversampling

Oversampling using:

-->Standard SMOTE (from imbalanced-learn)

-->Custom oversampling method developed earlier

Integration with a scikit-learn Pipeline

Model evaluation using classification metrics

Visual comparison of minority class recall across approaches

Key outcome:

Clear, empirical comparison showing how oversampling improves minority class performance

Demonstration of proper ML workflow: preprocessing → training → evaluation

This notebook connects the data generation logic to real model performance.

## Observations and Results
The baseline model performs poorly on the minority class despite reasonable overall accuracy

Standard SMOTE improves minority recall but may introduce additional false positives

The custom oversampling method improves minority class recall while remaining stable and reproducible

Data-level techniques play a critical role in imbalanced learning problems and can significantly affect downstream model behavior

### Key Takeaways
Class imbalance should be addressed early in the ML pipeline

Oversampling techniques must balance effectiveness with stability

Clear evaluation is essential to justify data preprocessing decisions

Robust, well-documented implementations are as important as model performance

