# SI-Neural-Computations

Data and analysis code for **“Bottom-up and generative computations uniquely explain neural responses across distinct social brain regions”** *(in press, PNAS)*

📄 Preprint: https://doi.org/10.64898/2026.02.20.707082  
📌 Preregistration: https://osf.io/hq3r7  

---
## 🧠 fMRI Data
The neuroimaging data are publicly available on **OpenNeuro**: https://openneuro.org/datasets/ds008495

The OpenNeuro dataset contains:

- BIDS-formatted fMRI data from the main task and localizers
- fMRIPrep outputs
- GLMsingle beta estimates
- post-scan behavioural ratings

---
To reproduce the analyses in this repository, clone the GitHub repository and download the complete OpenNeuro dataset **into the repository root**.

The resulting project directory should look approximately like:

```text
SI-Neural-Computations/
├── code/
│   └── preprocessing/
├── CONDA_ENVS/
├── derivatives/
│   ├── fmriprep/
│   ├── GLMsingle/
│   ├── model_and_behavioural_representations/
│   ├── analyses/
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

---

## 📦 What this repository contains:

- analysis code used in the paper
- processed intermediate derivatives needed for the reported analyses
- model and behavioural representations used for RSA
- supplementary-analysis code

The main analyses can therefore be run directly from the provided derivatives, without rerunning beta processing or ROI creation.

The computational model and behavioural representations used in the representational similarity analyses are provided in: `model_and_behavioural_representations/`

These include **SocialGNN, SIMPLE, Motion Energy, VisualRNN/ControlRNN, and Human Ratings**.

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
## 📊 Main Analyses

All scripts should be run from the `code/` directory. ```cd code```

### ROI-based RSA
```
python standardRSA_analysis.py \
    --mode ROI \
    --features SocialGNN10s_trained10s SIMPLE10s HR ME10s_reduced VisualRNN10s \
    --rois psts_r tpj_r
````

### ROI-based unique variance
```
python standardRSA_analysis.py \
    --mode ROIuniqvar \
    --features SocialGNN10s_trained10s SIMPLE10s \
    --sr_comparisons SocialGNN10s_trained10s,SIMPLE10s \
    --rois psts_r tpj_r
```

### Whole-brain RSA
```
python standardRSA_analysis.py \
    --mode wholebrain \
    --features SocialGNN10s_trained10s SIMPLE10s HR ME10s_reduced VisualRNN10s
```

### Whole-brain unique variance
```
python standardRSA_analysis.py \
    --mode wholebrain_uniqvar \
    --features SocialGNN10s_trained10s SIMPLE10s \
    --sr_comparisons SocialGNN10s_trained10s,SIMPLE10s
```

### Group-level whole-brain plots
```
python standardRSA_plotWholebrainGroupMaps.py \
    --features2test HR SocialGNN10s_trained10s SIMPLE10s ME10s_reduced VisualRNN10s \
    --sr_comparisons SocialGNN10s_trained10s,SIMPLE10s
```
---
## ⏱️ Time-resolved analyses

Processed 2-second-chunk derivatives are provided, so the time-resolved analysis can be run directly:
```
python standardRSA_analysis_2schunks.py \
    --mode ROIuniqvar \
    --features SocialGNN10s_trained10s SIMPLE10s
```

To regenerate the 2-second-chunk derivatives, use:
```
python preprocessing/glm_2schunks.py
python process_betas_2schunks.py
```
---
## 🔄 Optional: regenerate intermediate derivatives

To regenerate the processed betas and ROI/reliability derivatives from the released GLMsingle outputs:
```
python process_betas.py
python createROIs.py
```
ROI creation requires third-party parcel maps that are not included in this repository. Place them in: ```derivatives/localizer_parcelmaps/```

The expected filenames and directory structure are defined in ```createROIs.py.```

---
## 📚 Supplementary Analyses
Additional analyses reported in the Supplementary Information are in:

```code/supplementary_analyses.ipynb```

---
### For questions/issues: 👩‍💻 mmalik16@jhu.edu
