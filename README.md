# VITAL: Versatile Intelligent Task Allocation for Latency-sensitive Healthcare

#### A scalable, fault-tolerant, and resource-efficient framework for dynamic task scheduling and resource allocation in latency-sensitive healthcare systems.

---

## Abstract

###### This project presents a comprehensive implementation, critical analysis, and significant enhancement of the "Dynamic Priority-based Task Scheduling and Adaptive Resource Allocation (DPTARA)" framework proposed by J. Anand and B. Karthikeyan.

#### The project is twofold:
- **Baseline DPTARA:** A faithful implementation of the original paper's rule-based algorithm to establish a performance benchmark.
- **VITAL (Versatile Intelligent Task Allocation for Latency-sensitive Healthcare):** An advanced, re-engineered framework that addresses critical flaws of the original model. VITAL integrates data-driven intelligence, deep reinforcement learning, and a privacy-preserving architecture to create a robust, secure, and efficient system suitable for real-world healthcare IoT deployments.

*The goal is to empirically demonstrate the superiority of VITAL via comparative analysis of key performance metrics, including load utilization, critical task success rate, and power efficiency.*

---

## The Problem: Latency in Critical Healthcare

- In modern healthcare, IoT devices continuously monitor patients, generating vast data streams. While the cloud offers immense computational power, latency makes it unsuitable for time-sensitive alerts (e.g., detecting a cardiac arrhythmia).

- Edge computing brings computation closer to the data source, but edge servers have limited resources. The challenge: how to decide which tasks to process locally and which to offload, especially when multiple critical tasks compete for resources?

- The original DPTARA paper proposed a rule-based solution, but it suffers from limitations: simplistic priority modeling, lack of adaptability, and absence of security—unacceptable for sensitive patient data.

---

## Our Solution: DPTARA vs. VITAL

This project implements both the original concept and our enhanced solution to provide a clear comparison.

| Feature            | Baseline DPTARA (Original)                   | VITAL (Enhanced)                                              |
|--------------------|----------------------------------------------|----------------------------------------------------------------|
| Priority Model     | Static, rule-based (Normal/High)             | Dynamic, context-aware (AI-driven Urgency Score)              |
| Scheduling Logic   | Rigid, threshold-based offloading            | Adaptive (Deep Reinforcement Learning: DQN agent)             |
| Security & Privacy | Not addressed                                | Privacy-by-design (Federated Learning)                        |

----

### Key Enhancements in VITAL

- Context-Aware Prioritization with Anomaly Detection: An autoencoder analyzes task data (e.g., ECG signal coefficients). Reconstruction error generates a dynamic Urgency Score—higher error implies a more urgent medical event.
- Intelligent Scheduling with Deep Reinforcement Learning (DRL): A Deep Q-Network (DQN) agent learns optimal scheduling considering system state (task urgency, server loads, network latency) to maximize critical task completion and efficiency.
- Security & Privacy by Design with Federated Learning: Local training at the edge; only anonymous model updates (weights), not patient data, are aggregated centrally—supporting compliance with regulations such as HIPAA.

Architecture note: Local models are trained at the edge and only non-sensitive model weights are aggregated, ensuring patient data remains private.

---

## Project Structure

```
VITAL-Healthcare-Edge-Scheduler/
│
├── data/
│   ├── raw/
│   │   ├── DPTARA_dataset.csv
│   │   ├── DPTARA_Synthetic_Healthcare_Tasks.csv
│   │   └── healthcare_offloading_dataset.csv
│   └── processed/
│       └── (Processed or cleaned data will go here)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_dptara_simulation.ipynb
│   ├── 03_vital_anomaly_detection_model.ipynb
│   ├── 04_vital_drl_scheduler_training.ipynb
│   ├── 05_vital_federated_learning_poc.ipynb
│   └── 06_comparative_analysis_and_visualization.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── environment.py       # Orchestrator, main simulation loop
│   │   ├── entities.py          # Task, IoTDevice, EdgeServer, CloudServer
│   │   └── metrics.py           # Performance metrics
│   │
│   ├── schedulers/
│   │   ├── __init__.py
│   │   ├── base_scheduler.py    # Abstract base class
│   │   ├── dptara_baseline.py   # Original DPTARA scheduler
│   │   └── vital_drl.py         # VITAL DRL-based scheduler
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── anomaly_detector.py  # Autoencoder for Urgency Score
│   │   └── drl_agent.py         # DQN agent and network
│   │
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py       # CSV load/preprocess
│       ├── logger.py            # Logging config
│       └── plotter.py           # Comparison graphs
│
├── results/
│   ├── plots/
│   │   ├── load_utilization_comparison.png
│   │   └── critical_task_success_rate.png
│   └── logs/
│       ├── dptara_baseline_run.log
│       └── vital_run.log
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Datasets

This project uses eight datasets, grouped into primary sources for simulation and supplementary sources for clinical feature engineering.

### Primary Datasets (Simulation and Modeling)
- DPTARA_Synthetic_Healthcare_Tasks.csv: Used to replicate experiments and validate the baseline DPTARA implementation.
- healthcare_offloading_dataset.csv: Primary dataset for training/evaluating VITAL; includes features like DWT coefficients and heart rate for context-aware prioritization.
- DPTARA_dataset.csv: Reference dataset guiding simulation I/O structure.

### Supplementary Datasets (Feature Engineering)
These provide clinical and operational context to engineer data-driven features (e.g., Urgency Score).
- healthcare_dataset.csv: Patient demographics, admin, and high-level clinical info.
- healthcare_analysis.csv: Granular clinical data, vital signs, and unstructured clinical notes for NLP.
- HCP.csv: Patient-level administrative/clinical details (stays, severity, outcomes).
- medical_data.csv: Socio-demographics, lifestyle, and treatment outcomes.
- medical_resource_allocation_binary_dataset.csv: Links health status to resource utilization and network performance metrics (for the DRL agent).

----

## Getting Started:

### Prerequisites
```bash
Requirements:
- Python 3.9+
- pip
- Git
```
### Installation

Clone the repository:
```bash
git clone https://github.com/your-username/VITAL-Healthcare-Edge-Scheduler.git
cd VITAL-Healthcare-Edge-Scheduler
```

Create and activate a virtual environment:
```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
# venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```
---

## Running the Simulation

The simulation can be run from a main script (e.g., main.py). Select the scheduler via command-line arguments.

Run the Baseline DPTARA:
```bash
python main.py --scheduler=baseline
```

Run the Enhanced VITAL:
```bash
python main.py --scheduler=vital
```

NOTE: *Results (logs and plots) are saved in the results/ directory.

## Expected Results

- Higher critical task success rate (VITAL).
- More balanced load utilization across edge servers.
- Lower average task latency, especially for high-priority tasks.
- Improved power efficiency via intelligent resource allocation.

---

## Future Work

- Multi-Agent Reinforcement Learning (MARL): Extend to multi-agent where edge servers negotiate and cooperate.
- Physical Testbed Deployment: Validate on devices like Raspberry Pi or NVIDIA Jetson.
- Blockchain for Auditability: Permissioned blockchain for immutable audit trails of scheduling decisions.

---

## Citation

```
This project builds on the concepts introduced in:
- J. Anand and B. Karthikeyan, "Dynamic Priority-based Task Scheduling and Adaptive Resource Allocation (DPTARA)."
```
---
