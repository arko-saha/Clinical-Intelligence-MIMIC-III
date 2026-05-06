# Safe Offline Reinforcement Learning with Hierarchical and Multimodal Architectures for Personalized Sepsis Care

## Project Structure

src/
├── notebooks/          # Jupyter notebooks for analysis and experimentation
│   ├── module1.ipynb   # Data Pipeline
│   ├── module2.ipynb   # History-Aware State Encoder
│   ├── module3.ipynb   # [Description]
│   ├── module4.ipynb   # [Description]
│   ├── module5.ipynb   # [Description]
│   └── module6.ipynb   # [Description]
├── utils/              # Shared utility functions and classes
│   ├── __init__.py
│   └── data_pipeline.py # Data processing pipeline components
├── models/             # Saved model files
│   ├── acuity_conditioned_dt.pt
│   ├── multiobjective_reward_model.pt
│   ├── reward_model.pt
│   └── state_encoder.pt
└── charts/             # Generated plots and visualizations

## Shared Components

The utils/data_pipeline.py contains shared classes and functions used across multiple modules:

- CohortExtractor: Extracts sepsis cohort from MIMIC-III data
- FeatureExtractor: Extracts hourly clinical features
- FeatureImputer: Handles missing data imputation
- ActionDiscretizer: Discretizes fluid and vasopressor actions
- TrajectoryBuilder: Builds patient trajectories
- DatasetSplitter: Splits data into train/val/test
- FeatureNormalizer: Normalizes features for ML models

## Data Requirements

- MIMIC-III CSV files in ../../../data/
- Required files: ADMISSIONS.csv, PATIENTS.csv, ICUSTAYS.csv, DIAGNOSES_ICD.csv, CHARTEVENTS.csv, LABEVENTS.csv, OUTPUTEVENTS.csv, INPUTEVENTS_MV.csv

## Usage

Each notebook can be run independently by importing shared utilities.

## Development Notes

- All notebooks are designed to run independently
- Shared code is extracted to utils/ to avoid duplication
- Models are saved in models/ directory
- Charts and plots go in charts/ directory
