import os
import pandas as pd
import numpy as np
import nibabel as nib
import h5py
import pickle
import glob
from nilearn import plotting
from scipy.stats import pearsonr
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning) #for warnings during nan slices subtraction in betas

class SubTrialsProcessor:
    """
    Class to process trial-wise beta values from GLMsingle outputs,
    merge them with behavioral data (for task 'main'),
    compute contrasts (for localizer tasks),
    and compute voxelwise split-half reliability maps.
    """
    def __init__(self, sub_id, task, space="MNI152NLin2009cAsym", beta_type="fracridge", betas_normalize=True):
        self.sub_id = sub_id
        self.task = task
        self.space = space
        self.beta_type = beta_type
        self.betas_normalize = betas_normalize

        self.output_dir = f"../derivatives/nilearn_analysis/glmsingle_betas/sub-{sub_id}/"
        os.makedirs(self.output_dir, exist_ok=True)

        self.beta_string = "TYPED_FITHRF_GLMDENOISE_RR"
        if task == "physics":
            self.beta_type = "fittedhrf"  # If task is "physics", use "fittedhrf" as the default beta_type
            self.beta_string = 'TYPEB_FITHRF'
        self.runs = self._get_valid_runs()
        if self.task not in self.runs: # Check if the task is valid (i.e., not empty)
            print(f"Warning: Task '{self.task}' is not available for participant {self.sub_id}. Skipping this subject.")
            return 

        # Load event and beta data
        self.trials_df = self._load_event_files()
        self._process_event_data()
        self.beta_values = self._load_betas()

    def _get_valid_runs(self):
        """Define valid runs for each participant."""
        runs = {
            "main": ["01", "02", "03", "04", "05", "06", "07", "08"],
            "sipsts": ["01", "02"],
            "tom": ["01", "02"]
        }
        if self.sub_id not in ["M15", "M19", "M22"]:
            runs["main"].extend(["09", "10"])
        if self.sub_id not in ["M23"]:
            runs["physics"] = ["01"]

        return runs

    def _load_event_files(self):
        """Load and concatenate event files for all runs of a task."""
        event_files = [
            f"../sub-{self.sub_id}/func/sub-{self.sub_id}_task-{self.task}_run-{run}_events.tsv"
            for run in self.runs[self.task]
        ]
        return pd.concat([pd.read_csv(f, sep="\t") for f in event_files], ignore_index=True)

    def _process_event_data(self):
        """Standardize event file columns and add filename mapping if task=='main'."""
        if self.task != "main":
            if self.task == "physics":
                self.trials_df.drop(columns=["identifier"], inplace=True)
            self.trials_df.rename(columns={"trial_type": "identifier"}, inplace=True)
        else:
            with open("../derivatives/model_&_behavioural_representations/fname_i_dict", "rb") as f:
                fname_i_dict = pickle.load(f)
            self.trials_df["filename"] = self.trials_df["identifier"].map({v: k for k, v in fname_i_dict.items()})

    def _load_betas(self):
        """Load and optionally z-normalize GLMsingle beta values."""
        beta_file = f"../derivatives/GLMsingle/sub-{self.sub_id}/task-{self.task}_space-{self.space}/files/{self.beta_string}.hdf5"
        with h5py.File(beta_file, "r") as file:
            beta_values = file["betasmd"][()]
        if self.betas_normalize:
            beta_values = (beta_values - np.nanmean(beta_values, axis=3, keepdims=True)) / np.nanstd(beta_values, axis=3, keepdims=True)
        self.trials_df["betas"] = [beta_values[..., i] for i in range(beta_values.shape[3])]
        return beta_values

    def compute_condition_wise_betas(self):
        """Average trial-level betas per condition."""
        averaged_betas = self.trials_df.groupby("identifier")["betas"].apply(
            lambda arrays: np.mean(arrays) if len(arrays) > 1 else arrays.iloc[0]
        ).reset_index()
        return averaged_betas

    def merge_behavioral_responses(self, averaged_betas):
        """For 'main' task, merge averaged betas with behavioral ratings."""
        if self.task != "main":
            return
        behavior_file = glob.glob(f"../derivatives/model_&_behavioural_representations/subj_ratings/subj1{self.sub_id[1:]}/behaviouralfiles/*.csv")[0]
        behavioral_response = pd.read_csv(behavior_file)
        merged_df = averaged_betas.merge(
            behavioral_response[["video_name", "response", "movie_path"]],
            left_on="identifier",
            right_on="video_name",
            how="inner",
        )

        # Save results
        norm_suffix = "" if self.betas_normalize else "_betasnormalize-False"
        merged_filename = f"{self.output_dir}sub-{self.sub_id}_task-{self.task}_space-{self.space}_stat-conditionwise{norm_suffix}"
        trials_filename = f"{self.output_dir}sub-{self.sub_id}_task-{self.task}_space-{self.space}_stat-alltrialsdf{norm_suffix}"

        with open(merged_filename, "wb") as f:
            pickle.dump(merged_df, f)
            print(f"File written: {merged_filename}")

        with open(trials_filename, "wb") as f:
            pickle.dump(self.trials_df, f)
            print(f"File written: {trials_filename}")


    def compute_contrasts(self, averaged_betas):
        """Compute contrast maps for localizer tasks."""
        if self.task not in ["sipsts", "tom", "physics"]:
            return
        contrast = averaged_betas.iloc[0]["betas"] - averaged_betas.iloc[1]["betas"]

        # Load appropriate mask
        mask_template = f"../derivatives/fmriprep/sub-{self.sub_id}/func/sub-{self.sub_id}_task-{self.task}_space-{self.space}_desc-brain_mask.nii.gz"
        mask_file = mask_template if self.task == "physics" else mask_template.replace("_space-", "_run-1_space-")

        mask_image = nib.load(mask_file)
        contrast_image = nib.Nifti1Image(contrast, affine=mask_image.affine)
        contrast_filename = f"{self.output_dir}sub-{self.sub_id}_task-{self.task}_space-{self.space}_stat-{self.task}contrast.nii.gz"
        nib.save(contrast_image, contrast_filename)
        print(f"File written: {contrast_filename}")

        if self.task == "sipsts":
            no_interact_filename = f"{self.output_dir}sub-{self.sub_id}_task-{self.task}_space-{self.space}_stat-{self.task}nointeract.nii.gz"
            no_interact_image = nib.Nifti1Image(averaged_betas.iloc[1]["betas"], affine=mask_image.affine)
            nib.save(no_interact_image, no_interact_filename)
            print(f"File written: {no_interact_filename}")

    def compute_and_save_wholebrain_split_half_reliability(self, thres_clip_negatives=True):
        """
        Compute and save whole-brain voxelwise split-half reliability and p-values as NIfTI images.
        Optionally clips negative reliability values to zero.
        """

        even, odd = [], []

        # Split trials into even and odd for each video (identifier)
        for video, group in self.trials_df.groupby('identifier'):
            group = group.reset_index(drop=True)
            if len(group) > 1:
                odd_rows = group[group.index % 2 != 0]
                even_rows = group[group.index % 2 == 0]
                # Average the beta estimates across odd and even trials, respectively. (note that makes odd 2 rows, and even 3)
                odd.append(odd_rows['betas'].mean(axis=0).flatten())
                even.append(even_rows['betas'].mean(axis=0).flatten())

        # Convert lists to arrays; shape will be (voxels, number_of_splits)
        even = np.array(even).T
        odd = np.array(odd).T

        n_voxels = even.shape[0]
        reliability = np.full(n_voxels, np.nan)
        p_value = np.full(n_voxels, np.nan)

        # Compute voxelwise Pearson correlation between even and odd splits.
        for voxel_id in range(n_voxels):
            # Skip voxels with non-finite data or with no variance.
            if (not np.all(np.isfinite(even[voxel_id])) or
                not np.all(np.isfinite(odd[voxel_id])) or
                np.all(even[voxel_id] == even[voxel_id, 0]) or
                np.all(odd[voxel_id] == odd[voxel_id, 0])):
                continue
            r, p = pearsonr(even[voxel_id], odd[voxel_id])
            reliability[voxel_id] = r
            p_value[voxel_id] = p

        # set negative r values to 0
        if thres_clip_negatives:
            reliability[reliability < 0] = 0

        # --- Convert the Reliability Vector to a 3D Map ---
        # Load reference image to shape the reliability map
        mask_template = f"../derivatives/fmriprep/sub-{self.sub_id}/func/sub-{self.sub_id}_task-{self.task}_space-{self.space}_desc-brain_mask.nii.gz"
        mask_path = mask_template if self.task == "physics" else mask_template.replace("_space-", "_run-1_space-")

        ref_img = nib.load(mask_path)
        shape = ref_img.get_fdata().shape

        reliability_3d = reliability.reshape(shape)
        pval_3d = p_value.reshape(shape)

        reliability_img = nib.Nifti1Image(reliability_3d, affine=ref_img.affine, header=ref_img.header)
        pval_img = nib.Nifti1Image(pval_3d, affine=ref_img.affine, header=ref_img.header)

        reliability_dir = f"../derivatives/nilearn_analysis/reliability"
        os.makedirs(reliability_dir, exist_ok=True)

        suffix = "_betasnormalize-True" if self.betas_normalize else "_betasnormalize-False"
        r_outfile = f"{reliability_dir}/sub-{self.sub_id}_task-{self.task}_space-{self.space}_desc-betas-{self.beta_type}{suffix}_stat-r_statmap.nii.gz"
        p_outfile = f"{reliability_dir}/sub-{self.sub_id}_task-{self.task}_space-{self.space}_desc-betas-{self.beta_type}{suffix}_stat-p_statmap.nii.gz"
        nib.save(reliability_img, r_outfile)
        nib.save(pval_img, p_outfile)
        print(f"Saved r map: {r_outfile}")
        print(f"Saved p map: {p_outfile}")


    def process(self):
        """Main method to run full analysis pipeline for a given subject/task."""
        if self.task not in self.runs: # Check if the task is valid (i.e., not empty)
            return 
        averaged_betas = self.compute_condition_wise_betas()
        self.merge_behavioral_responses(averaged_betas) #relevant only for main task
        self.compute_contrasts(averaged_betas) #relevant only for localizers
        self.compute_and_save_wholebrain_split_half_reliability()


if __name__ == "__main__":
    sub_ids = ["M03", "M04", "M05", "M06", "M08", "M09", "M10", 
           "M11", "M12", "M13", "M15", "M17", "M18", "M19", "M20",
           "M21", "M22", "M23", "M24", "M25","M26", "M27", "M28", "M29", "M30"]

    for sub_id in sub_ids:
        # Use a dummy instance just to get valid tasks for that subject
        valid_tasks = SubTrialsProcessor(sub_id=sub_id, task="main")._get_valid_runs().keys()
        
        for task in valid_tasks:
            print(f"\nProcessing sub-{sub_id}, task-{task}")
            processor = SubTrialsProcessor(sub_id=sub_id, task=task)
            processor.process()
    
