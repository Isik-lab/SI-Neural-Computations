# SI-Neural-Computations

Data and analysis code for **“Bottom-up and generative computations uniquely explain neural responses across distinct social brain regions”** *(in press, PNAS)*

📄 Preprint: https://doi.org/10.64898/2026.02.20.707082  
📌 Preregistration: https://osf.io/hq3r7  


---
## 🧠 fMRI Data
The neuroimaging data associated with this study are publicly available on **OpenNeuro**: https://openneuro.org/datasets/ds008495

The OpenNeuro dataset contains:

- BIDS-formatted fMRI data from the main task and localizer scans
- fMRIPrep preprocessing outputs
- processed single-trial beta estimates generated using GLMsingle
- post-scan behavioural ratings

To reproduce the analyses in this repository, clone the GitHub repository and download the complete OpenNeuro dataset **into the repository root**.

The resulting project directory should look approximately like:

```text
SI-Neural-Computations/
├── code/
├── CONDA_ENVS/
├── model_&_behavioural_representations/
├── derivatives/
├── stimuli/
├── sub-M02/
├── sub-M03/
├── ...
├── dataset_description.json
├── participants.tsv
├── participants.json
├── README.md
└── LICENSE
```

## 🤖 Computational Models & Human Behavioural Ratings
//need to describe each better and add link to orig repos and papers for these...

- SocialGNN  
- SIMPLE   
- Motion Energy (ME)  
- VisualRNN (referred to as ControlRNN in the paper)

The model and behavioural representations used in the representational similarity analyses are provided in: `model_&_behavioural_representations/`

---
## ⚙️ Environment Setup
All analyses were run using the Conda environment defined in:
```env_macOS_fMRI_analysis.yml```

Create the environment:
```
conda env create -f CONDA_ENVS/env_macOS_fMRI_analysis.yml
conda activate fMRI_analysis
```

---
## 📊 Analysis Pipeline

All scripts should be run from the `code/` directory. ```cd code```

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

ROI creation requires third-party parcel maps that are not included in this repository. Download them from the original sources and place them in:

`derivatives/localizer_parcelmaps/`

The expected filenames and folder structure are defined in `createROIs.py`.

### 3️⃣ Representational Similarity Analysis (RSA)

#### 3.1 ROI-based RSA
```
python standardRSA_analysis.py \
	--mode ROI \
	--features SocialGNN10s_trained10s SIMPLE10s HR ME10s_reduced VisualRNN10s \
	--rois psts_r tpj_r
```

#### 3.2 ROI-based RSA — Unique Variance
```
python standardRSA_analysis.py \
	--mode ROIuniqvar \
	--features SocialGNN10s_trained10s SIMPLE10s HR ME10s VisualRNN10s \
	--rois psts_r tpj_r
```

#### 3.3 Whole-Brain Searchlight RSA (computationally intensive)
```
python standardRSA_analysis.py \
	--mode wholebrain \
	--features SocialGNN10s_trained10s SIMPLE10s HR ME10s_reduced VisualRNN10s
```

#### 3.4 Whole-Brain Searchlight — Unique Variance
```
python standardRSA_analysis.py \
	  --mode wholebrain_uniqvar \
	  --features SocialGNN10s_trained10s SIMPLE10s \
	  --sr_comparisons SocialGNN10s_trained10s,SIMPLE10s
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
python standardRSA_analysis_2schunks.py \
    --mode ROIuniqvar \
    --features SocialGNN10s_trained10s SIMPLE10s

## 📚 Supplementary Analyses
Supplementary analyses in one Jupyter notebook?

---
### For questions/issues: 👩‍💻 mmalik16@jhu.edu
