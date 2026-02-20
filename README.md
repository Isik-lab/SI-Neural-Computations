# SI–Neural–Computations

Data and analysis code for **“The neural computations underlying human social evaluations from visual input.”**

📄 Preprint: *[link]*  
📌 Preregistration: https://osf.io/hq3r7  

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

---
## ⚙️ Environment Setup
All analyses were run using the Conda environment defined in:
```fMRI_analysis_env.yml```

Create the environment:
```
conda env create -f fMRI_analysis_env.yml
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
```
#### 3.2 ROI-based RSA — Unique Variance
```
python standardRSA_analysis.py \
--mode ROIuniqvar \
--features SocialGNN10s_trained10s SIMPLE10s HR ME10s VisualRNN10s
```

#### 3.3 Whole-Brain Searchlight RSA (computationally intensive)
```
python standardRSA_analysis.py \
--mode wholebrain \
--features SocialGNN10s_trained10s SIMPLE10s HR ME10s VisualRNN10s
```

#### 3.4 Whole-Brain Searchlight — Unique Variance
```
python standardRSA_analysis.py --mode wholebrain_uniqvar
```

#### 3.5 Group-Level Whole-Brain Plots (computationally intensive)
```
python standardRSA_plotWholebrainGroupMaps.py
```

### 4️⃣ Time-Resolved Analyses
- GLM fitting
- Time-resolved RSA

## 📚 Supplementary Analyses
Supplementary analyses in one Jupyter notebook?

---
### For questions/issues: 👩‍💻 mmalik16@jhu.edu
