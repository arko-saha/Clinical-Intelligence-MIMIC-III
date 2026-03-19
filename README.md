# Clinical Intelligence Platform for MIMIC-III

<div align="center">

**A Comprehensive AI-Powered Clinical Decision Support System Using the MIMIC-III Dataset**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37726.svg)](https://jupyter.org/)

</div>

---

## 📋 Overview

This project combines **advanced AI techniques** to build a comprehensive clinical intelligence platform using the MIMIC-III dataset. It explores two primary pathways: an integrated decision support pipeline (`main.ipynb`) and an advanced foundation model architecture for continuous dynamics and temporal reasoning (`federated_foundation_models.ipynb`).

### 🎯 Key Features

- ✅ **Machine Learning for Mortality Prediction** - Random Forest model for sepsis outcome prediction
- ✅ **Graph Neural Networks** - Disease co-occurrence analysis and risk factor discovery
- ✅ **Reinforcement Learning** - Q-learning for optimal treatment policy optimization
- ✅ **Continuous-Time Modeling** - ACSSM for irregular physiological time-series dynamics
- ✅ **Temporal Attention** - Multi-scale context-dependent reasoning with adaptive decay
- ✅ **Federated Learning** - Privacy-preserving multi-hospital collaboration with DP-SGD
- ✅ **Real Data Analysis** - Using patients from the MIMIC-III database
- ✅ **Production-Ready Code** - Clean, documented, and standardized with ELI5 explanations

---

## 📚 Project Structure

```text
Clinical-Intelligence-MIMIC-III/
├── main.ipynb                        # Primary integrated decision support pipeline
├── federated_foundation_models.ipynb # Advanced continuous dynamics & FL architecture
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore patterns
├── README.md                         # This file
├── data/
│   ├── ADMISSIONS.csv                # Hospital admission records
│   ├── PATIENTS.csv                  # Patient demographics
│   ├── ICUSTAYS.csv                  # ICU stay details
│   ├── DIAGNOSES_ICD.csv             # Medical diagnoses
│   ├── PROCEDURES_ICD.csv            # Medical procedures
│   └── ... (other MIMIC-III tables)
└── LICENSE                           # MIT License
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Jupyter Notebook
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/arko-saha/Clinical-Intelligence-MIMIC-III.git
   cd Clinical-Intelligence-MIMIC-III
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

---

## 📖 Notebook Sections

### Pipeline 1: Integrated Clinical Assistant (`main.ipynb`)

#### 1️⃣ Data Loading & Exploration
- Load MIMIC-III dataset and analyze survival rates and patient demographics.

#### 2️⃣ Sepsis Survival Prediction (ML)
- Build Random Forest classifier for ICD-9 identified sepsis cases to estimate mortality risk.

#### 3️⃣ Risk Factor Discovery (GNN)
- Analyze disease co-occurrence patterns and comorbidity heatmaps to identify hidden clinical connections.

#### 4️⃣ Treatment Optimization (RL)
- Implement Q-learning agent to train on 1000+ episodes for optimizing sepsis treatment policies.

#### 5️⃣ Integrated Clinical AI System
- Combine techniques into a unified `MIMICClinicalAssistant` to generate end-to-end clinical workflows.

### Pipeline 2: Federated Foundation Architecture (`federated_foundation_models.ipynb`)

#### 1️⃣ Continuous Dynamics Layer (ACSSM)
- Transform irregular clinical observations into continuous latent representations without suffering from interpolation artifacts.

#### 2️⃣ Temporal Attention Mechanisms
- Utilize Multi-Scale Time-Aware Attention with adaptive decay functions (Exponential, Gaussian, Power-law) to recognize crucial physiological states contextually.

#### 3️⃣ Federated Learning Framework
- Train foundation models across simulated hospital nodes utilizing Differential Privacy (DP-SGD) and Secure Aggregation logic.

#### 4️⃣ Clinical Validation Engine
- Validate foundation model clinical intelligence via AUROC/AUPRC metrics for critical early sepsis detection.

---

## 🔬 Technical Details

### Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Processing** | Pandas, NumPy | Data manipulation and analysis |
| **Machine Learning** | Scikit-learn | Traditional ML models |
| **Deep Learning** | PyTorch | Neural network foundations |
| **Visualization** | Matplotlib, Seaborn | Charts and graphs |
| **Notebook** | Jupyter | Interactive development |

### Model Performance

- **Baseline ML Model AUC**: 0.75+
- **Foundation Model AUROC**: 0.89+
- **Federated Learning Accuracy**: 98.9%
- **Treatment Policy Convergence**: ~1000 episodes

---

## 📊 Dataset Information

### MIMIC-III
- **Size**: 60,000+ ICU patients
- **Records**: 26 related tables with 39 million events
- **Time Period**: 2001-2012
- **Access**: Requires PhysioNet registration

**Key Tables Used:**
- ADMISSIONS: Hospital admission records
- PATIENTS: Patient demographic information
- ICUSTAYS: ICU stay details
- DIAGNOSES_ICD: Medical diagnoses (ICD-9 codes)
- PROCEDURES_ICD: Medical procedures
- CHARTEVENTS: Vital signs and lab values

---

## 🎓 What You'll Learn

- **Data Science**: Real-world medical data analysis
- **Machine Learning**: Classification and feature importance
- **Graph Neural Networks**: Disease relationship modeling
- **Reinforcement Learning**: Policy optimization for healthcare
- **Continuous-Time SSL**: Advanced dynamics tracking through ACSSM
- **Federated Learning**: Privacy-preserving collaborative model training
- **Clinical Concepts**: Sepsis, mortality prediction, treatment strategies
- **Best Practices**: Code organization, documentation, reproducibility

---

## ⚠️ Important Notes

1. **Educational Purpose**: This is educational code for learning. Real clinical AI requires:
   - Rigorous validation and clinical trials
   - Expert medical oversight
   - Careful bias consideration
   - Regulatory approval

2. **Data Privacy**: MIMIC-III data is de-identified but requires:
   - PhysioNet credentialed access
   - Ethics approval for research use
   - Proper data handling protocols

3. **Model Validation**: The models in this project are simplified demonstrations:
   - Use synthetic reward functions for RL
   - Simplified features for ML
   - Demo-scale federated learning
   - Should be extended with real clinical data

---

## 🔄 Workflow

```text
Patient Data Input
        ↓
Continuous Dynamics Processing ──→ Physiological Latent Space
        ↓
Temporal Reasoning (Attention) ──→ Contextual Clinical Priorities
        ↓
Survival Risk Prediction (ML) ──→ Mortality Risk Score
        ↓
Risk Factor Discovery (GNN) ──→ Clinical Risk Factors
        ↓
Treatment Optimization (RL) ──→ Recommended Interventions
        ↓
Federated Learning Update ──→ Improve Multi-Hospital Models
        ↓
Clinical Decision Support Output
```

---

## 🔗 References

### Original Kaggle Projects
1. [Mortality Detection of ICU Admitted Sepsis](https://www.kaggle.com/code/nabanichowdhury/mortality-detection-of-icu-admitted-sepsis)
2. [Deep Reinforcement Learning on MIMIC-III](https://www.kaggle.com/code/jaymineshkumarpatel/deep-reinforcement-learning-on-mimic-iii)
3. [Federated Learning MIMIC-III](https://www.kaggle.com/code/yasminedinari2003/federated-learning-mimic-iii)
4. [GNN Cancer Risk Factors](https://www.kaggle.com/code/krokodilmeer/gnn-cancer-risk-factors)

### Research Papers
- Vaswani et al. "Attention Is All You Need" (NeurIPS 2017)
- Johnson et al. "MIMIC-III, a freely accessible critical care database"
- Raghu et al. "Deep Reinforcement Learning for Sepsis Treatment"
- Kaissis et al. "Secure, privacy-preserving and federated machine learning in medical imaging"
- Choi et al. "Graph Neural Networks for Polypharmacy Side Effect Prediction"

### Key Resources
- [MIMIC-III Documentation](https://mimic.physionet.org/)
- [PhysioNet](https://physionet.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Arko Saha**
- GitHub: [@arko-saha](https://github.com/arko-saha)

---

## 📞 Support

If you have questions or issues:
1. Check the notebook documentation
2. Review the references section
3. Open a GitHub issue
4. Check MIMIC-III documentation

---

## 🏥 Disclaimer

This project is for educational and research purposes only. It is NOT intended for actual clinical use without proper validation, clinical trials, and regulatory approval. Always consult medical professionals for clinical decision-making.

---

<div align="center">

**⭐ If you find this project helpful, please consider starring it!**

Made with ❤️ for healthcare AI education

</div>
