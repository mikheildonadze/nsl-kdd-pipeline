# nsl-kdd-pipeline
Evaluation of machine learning algorithms in intrusion detection systems: using the NSL-KDD database as an example

# AI-Driven Intrusion Detection System (IDS) on NSL-KDD Benchmark

An end-to-end, reproducible Python machine learning pipeline for Network Intrusion Detection using the standard NSL-KDD benchmark dataset.

---

## Project Overview

Intrusion Detection Systems (IDS) are critical for safeguarding modern cybersecurity infrastructures against an ever-growing array of threats. As cyber-attacks evolve in sophistication, traditional rule-based IDS struggle to detect novel or adaptive threats effectively. This article explores how machine learning (ML), particularly AI-driven techniques, enhances IDS performance by enabling adaptive, dynamic, and accurate threat detection. We review the current research landscape across advanced ML paradigms - deep learning, reinforcement learning, and unsupervised learning - and their application to IDS, and we present a Python-based baseline pipeline illustrating data preprocessing, model training, and evaluation on the NSL-KDD dataset. This work underscores the transformative potential of AI-driven IDS in addressing contemporary cybersecurity challenges and situates the proposed pipeline within the broader body of empirical research on the topic.

---

## Pipeline Architecture & Methodology

The pipeline implements an empirical baseline benchmarking three core machine learning models across both binary and multi-class classification tasks.

### Core Pipeline Phases

1. **Data Ingestion & Integrity Sanity Checks**
   * Imports network traffic logs using standard NSL-KDD feature schemas.
   * Drops non-predictive metadata fields (e.g., `difficulty`).
   * Analyzes attack category discrepancies between train and test distributions to quantify unseen/zero-day threat vectors.

2. **Data Preprocessing & Leakage Safeguards**
   * **Categorical Encoding:** Applies label encoding across categorical features (`protocol_type`, `service`, `flag`) to establish feature vocabulary alignment across training and test splits.
   * **Standardization:** Fits `StandardScaler` strictly on training feature arrays (`KDDTrain+.txt`) and transforms evaluation arrays (`KDDTest+.txt`) downstream to guarantee no data leakage occurs from test set distributions.

3. **Classification Tasks**
   * **Binary Classification:** Models distinguish between `Normal` and `Attack` network traffic.
   * **Multi-Class Classification:** Maps granular attack subtypes into five standardized categories:
     * **Normal:** Standard network traffic.
     * **DoS (Denial of Service):** Attacks flooding system resources (e.g., `neptune`, `smurf`, `back`).
     * **Probe:** Surveillance and port scanning activities (e.g., `satan`, `ipsweep`, `nmap`).
     * **R2L (Remote to Local):** Unauthorized remote access exploitation (e.g., `guess_passwd`, `warezmaster`).
     * **U2R (User to Root):** Local privilege escalation attempts (e.g., `buffer_overflow`, `rootkit`).

4. **Model Benchmarking**
   * **Random Forest Classifier:** 100 decision trees trained with full parallelization (`n_jobs=-1`).
   * **Support Vector Machine (SVC):** Radial Basis Function (RBF) kernel with scale gamma.
   * **Deep Neural Network (MLP):** Multi-Layer Perceptron architecture with hidden layer topology `(128, 64, 32)`, ReLU activation, Adam optimizer, and early stopping.

---

## Evaluation Protocol

This project strictly adheres to the standard NSL-KDD benchmark protocol:
* Models are trained **exclusively** on `KDDTrain+.txt` (125,973 records).
* Models are evaluated **exclusively** on `KDDTest+.txt` (22,544 records).

> **Evaluation Note:** The evaluation set contains novel attack subtypes absent from the training split by design. Evaluating models on `KDDTest+` measures true model generalization against unseen threats rather than simple same-distribution performance.

---

## Prerequisites & Installation

### Requirements
* Python 3.8+
* Required packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

### Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/nsl-kdd-ids-pipeline.git](https://github.com/your-username/nsl-kdd-ids-pipeline.git)
   cd nsl-kdd-ids-pipeline
