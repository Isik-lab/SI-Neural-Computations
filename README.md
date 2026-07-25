# SI-Neural-Computations

Data and analysis code for **“Bottom-up and generative computations uniquely explain neural responses across the social brain”**

📄 Preprint: https://doi.org/10.64898/2026.02.20.707082  
📌 Preregistration: https://osf.io/hq3r7  

#### (_❗️Repository is under construction ⚒️_)
---
## Data
- 🧠 **fMRI**
  (orig files, and Steps and where to find preprocessed outputs from fMRIPrep and GLMsingle)
- 👩🏻‍🤝‍👩🏽 **Behavioral**

---
## 🤖 Computational Models
- SocialGNN  
- SIMPLE   
- Motion Energy (ME)  
- VisualRNN (referred to as ControlRNN in the paper)

The representations used in the RSA analyses can be found under:
```derivatives/model_&_behavioral_representations/```
Note that the motion energy representation are stored as a zip. Uncompress it after cloning the repo!

---
## ⚙️ Environment Setup
All analyses were run using the Conda environment defined in:
```env_macOS_fMRI_analysis.yml```

Create the environment:
```
conda env create -f env_macOS_fMRI_analysis.yml
conda activate fMRI_analysis
```

---
## 📊 Analysis Pipeline

All scripts should be run from the `code/` directory.

### 1️⃣ Beta Processing
Process single-trial beta estimates for each participant:
```
python process_betas.py
```
### 2️⃣ ROI Creation
Generate subject-specific ROIs:
```
python createROIs.py
```
### 3️⃣ Representational Similarity Analysis (RSA)

#### 3.1 ROI-based RSA
```
python standardRSA_analysis.py \
--mode ROI \
--features SocialGNN10s_trained10s SIMPLE10s HR ME10s VisualRNN10s
--roi  ('sipsts', 'psts_r') ('tom', 'tpj_r')
```
#### 3.2 ROI-based RSA — Unique Variance
```
python standardRSA_analysis.py \
--mode ROIuniqvar \
--features SocialGNN10s_trained10s SIMPLE10s HR ME10s VisualRNN10s
--roi  ('sipsts', 'psts_r') ('tom', 'tpj_r')
```

#### 3.3 Whole-Brain Searchlight RSA (computationally intensive)
```
python standardRSA_analysis.py \
--mode wholebrain \
--features SocialGNN10s_trained10s SIMPLE10s HR ME10s_reduced VisualRNN10s
```

#### 3.4 Whole-Brain Searchlight — Unique Variance
```
python standardRSA_analysis.py --mode wholebrain_uniqvar
```

#### 3.5 Group-Level Whole-Brain Plots (computationally intensive)
```
python standardRSA_plotWholebrainGroupMaps.py \
--features2test HR SocialGNN10s_trained10s SIMPLE10s ME10s_reduced VisualRNN10s\
--sr_comparisons SocialGNN10s_trained10s,SIMPLE10s
```

### 4️⃣ Time-Resolved Analyses
- GLM fitting
- Time-resolved RSA

## 📚 Supplementary Analyses
Supplementary analyses in one Jupyter notebook?

---
### For questions/issues: 👩‍💻 mmalik16@jhu.edu
