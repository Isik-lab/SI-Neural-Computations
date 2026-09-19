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
    def __init__(self, sub_id, task, space="MNI152NLin2009cAsym", beta_type="fracridge", betas_normalize=True, chunk="0-2s"):
        self.sub_id = sub_id
        self.task = task
        self.space = space
        self.beta_type = beta_type
        self.betas_normalize = betas_normalize
        self.chunk = chunk

        self.output_dir = f"../derivatives/analyses/processed_betas_chunks/sub-{sub_id}/"
        os.makedirs(self.output_dir, exist_ok=True)

        self.beta_string = "TYPED_FITHRF_GLMDENOISE_RR"
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
        }
        if self.sub_id not in ["M15", "M19", "M22"]:
            runs["main"].extend(["09", "10"])

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
        with open("../derivatives/model_and_behavioural_representations/fname_i_dict", "rb") as f:
            fname_i_dict = pickle.load(f)
        self.trials_df["filename"] = self.trials_df["identifier"].map({v: k for k, v in fname_i_dict.items()})

    def _load_betas(self):
        """Load and optionally z-normalize GLMsingle beta values."""
        beta_file = f"../derivatives/GLMsingle_2schunks/sub-{self.sub_id}/task-{self.task}_space-{self.space}/files/{self.beta_string}.hdf5"
        with h5py.File(beta_file, "r") as file:
            beta_values_allchunks = file["betasmd"][()]

        betas_chunked_reshaped = beta_values_allchunks.reshape(71, 84, 56, len(self.runs["main"]), 28, 5)
        map_chunks = {"0-2s":0, "2-4s":1, "4-6s":2, "6-8s":3, "8-10s":4}
        beta_values = betas_chunked_reshaped[..., map_chunks[self.chunk]].reshape(71, 84, 56, 28*len(self.runs["main"]))

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

        fname_map = self.trials_df[["identifier", "filename"]].drop_duplicates()
        averaged_betas = averaged_betas.merge(fname_map, on="identifier", how="left")

        behavior_file = f"../sub-{self.sub_id}/beh/sub-{self.sub_id}_task-postscan_events.tsv"
        behavioral_response = pd.read_csv(behavior_file, sep="\t")

        merged_df = averaged_betas.merge(
            behavioral_response[["stimulus_id", "response"]],
            left_on="identifier",
            right_on="stimulus_id",
            how="inner",
        ).drop(columns="stimulus_id")

        # Save results
        norm_suffix = "" if self.betas_normalize else "_betasnormalize-False"
        merged_filename = f"{self.output_dir}sub-{self.sub_id}_task-{self.task}_chunk-{self.chunk}_space-{self.space}_stat-conditionwise{norm_suffix}"
        trials_filename = f"{self.output_dir}sub-{self.sub_id}_task-{self.task}_chunk-{self.chunk}_space-{self.space}_stat-alltrialsdf{norm_suffix}"

        with open(merged_filename, "wb") as f:
            pickle.dump(merged_df, f)
            print(f"File written: {merged_filename}")


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
        mask_path = f'../derivatives/fmriprep/sub-{self.sub_id}/func/sub-{self.sub_id}_task-{self.task}_run-1_space-{self.space}_desc-brain_mask.nii.gz'
        ref_img = nib.load(mask_path)
        shape = ref_img.get_fdata().shape

        reliability_3d = reliability.reshape(shape)
        pval_3d = p_value.reshape(shape)

        reliability_img = nib.Nifti1Image(reliability_3d, affine=ref_img.affine, header=ref_img.header)
        pval_img = nib.Nifti1Image(pval_3d, affine=ref_img.affine, header=ref_img.header)

        reliability_dir = f"../derivatives/nilearn_analysis/reliability/chunks"
        os.makedirs(reliability_dir, exist_ok=True)

        suffix = "_betasnormalize-True" if self.betas_normalize else "_betasnormalize-False"
        r_outfile = f"{reliability_dir}/sub-{self.sub_id}_task-{self.task}_chunk-{self.chunk}_space-{self.space}_desc-betas-{self.beta_type}{suffix}_stat-r_statmap.nii.gz"
        p_outfile = f"{reliability_dir}/sub-{self.sub_id}_task-{self.task}_chunk-{self.chunk}_space-{self.space}_desc-betas-{self.beta_type}{suffix}_stat-p_statmap.nii.gz"
        nib.save(reliability_img, r_outfile)
        nib.save(pval_img, p_outfile)
        print(f"Saved r map: {r_outfile}")
        print(f"Saved p map: {p_outfile}")


    def process(self):
        """Main method to run full analysis pipeline for a given subject/task."""
        if self.task not in self.runs:  # Check if the task is valid (i.e., not empty)
            return 
        """Run the full pipeline."""
        averaged_betas = self.compute_condition_wise_betas()
        self.merge_behavioral_responses(averaged_betas)
        self.compute_and_save_wholebrain_split_half_reliability()


if __name__ == "__main__":
    sub_ids = ["M03", "M04", "M05", "M06", "M08", "M09", "M10", 
           "M11", "M12", "M13", "M15", "M17", "M18", "M19", "M20",
           "M21", "M22", "M23", "M24", "M25","M26", "M27", "M28", "M29", "M30"]

    chunks = ["0-2s", "2-4s", "4-6s", "6-8s", "8-10s"]

    for chunk in chunks:
        print(f"\n==== Processing chunk: {chunk} ====")
        for sub_id in sub_ids:
            # Use a dummy instance just to get valid tasks for that subject
            valid_tasks = SubTrialsProcessor(sub_id=sub_id, task="main", chunk=chunk)._get_valid_runs().keys()

            for task in valid_tasks:
                print(f"\nProcessing sub-{sub_id}, task-{task}")
                processor = SubTrialsProcessor(sub_id=sub_id, task="main", chunk=chunk)
                processor.process()

    
