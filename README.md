# TML26_Privacy_MIA_A1
A practical implementation of a LiRA-inspired Membership Inference Attack demonstrating how trained neural networks leak information about their training data through confidence and prediction behavior.
# Membership Inference Attack (MIA) – TML Task 1

This repository contains an implementation of a Membership Inference Attack (MIA) on a pretrained ResNet-18 image classification model. The goal is to predict whether a given sample was part of the model’s training dataset.

The attack is based on a LiRA-style likelihood ratio method with class-wise Gaussian modeling of confidence scores.

---

## 📌 Overview

Machine learning models often behave differently on training data compared to unseen data. This project exploits those differences to infer membership.

Given:
- A pretrained ResNet-18 model
- A public dataset with membership labels
- A private dataset without membership labels

We compute a score ∈ [0,1] for each sample indicating the likelihood of being a training member.

---

## ⚙️ Methodology

The implementation follows a likelihood-based approach:

1. Pass each sample through the model to obtain logits
2. Compute a **margin score**:
   - Difference between top-1 and top-2 logits
3. Separate scores into:
   - Members (training samples)
   - Non-members
4. Fit **Gaussian distributions per class** for both groups
5. Compute **log-likelihood ratio (LiRA-style)**:
   - Measures how likely a sample belongs to training data
6. Apply:
   - Percentile clipping (1%–99%)
   - Min-max normalization to [0,1]

This produces a robust ranking aligned with the **TPR@5%FPR** evaluation metric.

---

## 📦 Requirements

Install dependencies:

```bash
pip install torch torchvision numpy pandas scipy requests
