# SI-Neural-Computations
Data and code for paper "The neural computations underlying human social evaluations from visual input"
Paper link:
Preregistration link: https://osf.io/hq3r7

## Data
#### fMRI (also here: <TBA: fMRIprep, GLMsingle>?)
#### Behavioural

## Models

## ⚙️ Environment
The Conda environment used for the analyses is defined in: fMRI_analysis_env.yml

## 🔄 Analysis Pipeline
Run all steps from the `code/` folder.

### Step 1: Process betas for each participant 
``` python process_betas.py ```

### Step 2: Create ROIs for each participant
```python createROIs.py```

### Step 3: Standard RSA Analysis & Variance partitioning

#### 3.1 ROI-based RSA

python standardRSA_analysis.py --mode ROI --features SocialGNN10s_trained10s SIMPLE10s HR ME10s VisualRNN10s

#### 3.2 ROI-based RSA: Unique Variance

python standardRSA_analysis.py --mode ROIuniqvar --features SocialGNN10s_trained10s SIMPLE10s HR ME10s VisualRNN10s

#### 3.3 Whole-brain Searchlight RSA (Takes time!)

python standardRSA_analysis.py --mode wholebrain --features SocialGNN10s_trained10s SIMPLE10s HR ME10s VisualRNN10s

#### 3.4 Whole-brain Searchlight RSA: Unique Variance

python standardRSA_analysis.py --mode wholebrain_uniqvar

#### 3.5 Group-level Whole-brain RSA Plots (Also takes time)

python standardRSA_plotWholebrainGroupMaps.py

### Step 4: Time-resolved analysis
glm fitting, rsa

### Misc/Supplementary analyses
<python notebook?>
