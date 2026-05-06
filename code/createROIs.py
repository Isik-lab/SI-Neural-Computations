# ROI and Subject ROI Analysis Pipeline
# -----------------------------------
# This script defines classes and methods to generate ROI masks from localizer+parcel data,
# compute split-half reliability and split-half RSA, optionally using reliability-masked voxels.

import nibabel as nib
from nilearn import plotting
from nilearn.image import new_img_like, resample_to_img
import numpy as np
from general_utils import mask_img, plot_on_surf, get_rdm, load_reliability_mask
import warnings
import pickle
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from scipy.stats import pearsonr, spearmanr
import copy
from roi_utils import make_roiwise_boxplot
import seaborn as sns

class ROI:
    """
    Class to create subject-specific ROI mask by combining a group-level parcel mask
    with individual subject's localizer activation map. Includes logic to handle
    parcel-specific preprocessing (e.g. splitting STS into asts and psts).
    """
    def __init__(self, sub_id, parcel_name, task, top_perc, space):
        self.parcel_map_dir = '../derivatives/localizer_parcelmaps/'
        self.parcel_name = parcel_name
        self.task = task
        self.sub_id = sub_id
        self.space = space

        self.top_perc = top_perc
        
        self.sub_localizer = self._load_sublocalizer()
        self.parcel = self._get_parcel()
        self.roi = self.create_ROI()

    def _get_parcel(self):
        
        # Define base paths for each parcel, using placeholders for hemisphere-specific files
        base_paths = {
            "tpj": 'Saxe_tom/ToM_thresholded/{}TPJ_xyz.img',
            "mmpfc": 'Saxe_tom/ToM_thresholded/MMPFC_xyz.img',
            "dmpfc": 'Saxe_tom/ToM_thresholded/DMPFC_xyz.img',
            "vmpfc": 'Saxe_tom/ToM_thresholded/VMPFC_xyz.img',
            "sts": 'BenDeen_SocialPerception/{}STS.nii.gz',
            "mt": 'SabineKastner/subj_vol_all/perc_VTPM_vol_roi13_{}h.nii.gz',
            "v1v": 'SabineKastner/subj_vol_all/perc_VTPM_vol_roi1_{}h.nii.gz',
            "v1d": 'SabineKastner/subj_vol_all/perc_VTPM_vol_roi2_{}h.nii.gz',
            "v2v": 'SabineKastner/subj_vol_all/perc_VTPM_vol_roi3_{}h.nii.gz',
            "v2d": 'SabineKastner/subj_vol_all/perc_VTPM_vol_roi4_{}h.nii.gz',
            "v3v": 'SabineKastner/subj_vol_all/perc_VTPM_vol_roi5_{}h.nii.gz',
            "v3d": 'SabineKastner/subj_vol_all/perc_VTPM_vol_roi6_{}h.nii.gz',
            "physics": 'Pramod_Sharishared_Physics/combphysicsparcel_1_5.nii.gz'
        }

        # Helper function to load and resample the parcel data
        def load_and_resample(path, hemi_suffix=''):
            # Create the full file path with hemisphere suffix
            full_path = self.parcel_map_dir + path.format(hemi_suffix)
            # Load the NIfTI image and resample to match the subject localizer dimensions
            return resample_to_img(nib.load(full_path), self.sub_localizer, interpolation='nearest')

        # Function to split STS into anterior (asts) and posterior (psts) parts
        def split_parcel(parcel, split_type):
            # Get the parcel data as a numpy array
            parcel_data = parcel.get_fdata()
            # Find the voxel indices for the parcel
            parcel_voxels = np.where(parcel_data == 1)
            # Compute the center of the parcel in the anterior-posterior axis
            center_y = int((np.min(parcel_voxels[1]) + np.max(parcel_voxels[1])) / 2)
            # If the parcel is posterior (psts), zero out the anterior half, and vice-versa
            if split_type == 'psts':
                parcel_data[:, center_y:, :] = 0
            elif split_type == 'asts':
                parcel_data[:, :center_y, :] = 0
            # Return the split parcel as a new NIfTI image
            return nib.Nifti1Image(parcel_data, parcel.affine, parcel.header)

        def split_lr_by_x0(parcel):
            """Assumes img is already resampled to RAS target. Splits at world x=0."""
            data = parcel.get_fdata()
            if data.ndim == 4 and data.shape[3] == 1:
                data = data[..., 0]
            I, J, K = np.indices(data.shape[:3])
            A = parcel.affine
            xworld = A[0,0]*I + A[0,1]*J + A[0,2]*K + A[0,3]
            mask = data > 0
            Lmask = (xworld <  0) & mask
            Rmask = (xworld >= 0) & mask  # ties go Right; flip if you prefer
            L = np.zeros_like(data); L[Lmask] = 1
            R = np.zeros_like(data); R[Rmask] = 1
            return {'l': nib.Nifti1Image(L, parcel.affine, parcel.header),'r': nib.Nifti1Image(R, parcel.affine, parcel.header)}

        # Initialize an empty dictionary to hold the parcel data
        parcel = {}
        # sts parcel
        if self.parcel_name in ["asts", "psts"]:
            for hemi in ['l', 'r']:
                # Load and resample the original parcel
                original_parcel = load_and_resample(base_paths["sts"], hemi)
                # Split the parcel into anterior or posterior sections
                parcel[hemi] = split_parcel(original_parcel, self.parcel_name)
        # tpj or mt parcels
        elif self.parcel_name in ["tpj", "mt"]:
            for hemi in ['l', 'r']:
                # Load and resample the parcel
                parcel[hemi] = load_and_resample(base_paths[self.parcel_name], hemi)
                # Binarize the mt parcel by thresholding at >0
                if self.parcel_name in ["mt"]:
                    parcel_data = parcel[hemi].get_fdata()
                    binary_data = np.where(parcel_data > 0, 1, 0)
                    parcel[hemi] = nib.Nifti1Image(binary_data, parcel[hemi].affine, parcel[hemi].header)
        elif self.parcel_name == "v1":
            for hemi in ['l', 'r']:
                # Load and resample the parcels
                v1v = load_and_resample(base_paths['v1v'], hemi)
                v1d = load_and_resample(base_paths['v1d'], hemi)
                v2v = load_and_resample(base_paths['v2v'], hemi)
                v2d = load_and_resample(base_paths['v2d'], hemi)
                v3v = load_and_resample(base_paths['v3v'], hemi)
                v3d = load_and_resample(base_paths['v3d'], hemi)
                
                parcel_data = v1v.get_fdata() + v1d.get_fdata() + v2v.get_fdata() + v2d.get_fdata() + v3v.get_fdata() + v3d.get_fdata() 
                #top_x_percent_threshold = np.nanpercentile(parcel_data[parcel_data > 0], 100 - self.top_perc)
                top_x_percent_threshold = np.nanpercentile(parcel_data[parcel_data > 0], 100-100) #keep all voxels
                binary_data = np.where(parcel_data >= top_x_percent_threshold, 1, 0).astype(np.int32)
                parcel[hemi] = nib.Nifti1Image(binary_data, v1v.affine, v1v.header)
        elif self.parcel_name == "physics_pramod": 
            physics_img = load_and_resample(base_paths['physics'], hemi_suffix='')
            parcel = split_lr_by_x0(physics_img)  # returns {'l': imgL, 'r': imgR}         
        else:
            # For parcels that don't require special handling, load and resample directly
            parcel_path = base_paths.get(self.parcel_name, '')
            if parcel_path:
                parcel = load_and_resample(parcel_path)

        return parcel

    def _load_sublocalizer(self):
        # Load the subject's localizer file based on the parcel and task
        if self.parcel_name == "mt":
            return nib.load(f'../derivatives/nilearn_analysis/glmsingle_betas/sub-{self.sub_id}/sub-{self.sub_id}_task-{self.task}_space-{self.space}_stat-{self.task}nointeract.nii.gz')
        elif self.task is None:
            return nib.load(f'../derivatives/nilearn_analysis/glmsingle_betas/sub-{self.sub_id}/sub-{self.sub_id}_task-sipsts_space-{self.space}_stat-sipstscontrast.nii.gz')
        else:
            return nib.load(f'../derivatives/nilearn_analysis/glmsingle_betas/sub-{self.sub_id}/sub-{self.sub_id}_task-{self.task}_space-{self.space}_stat-{self.task}contrast.nii.gz')

    def _combine_parcel_localizer(self):
        # If the parcel is split into left and right hemispheres
        if isinstance(self.parcel, dict):
            # Combine left and right hemisphere parcels by adding their data
            combined_parcel = self.parcel['l'].get_fdata() + self.parcel['r'].get_fdata()
            
            # Extract sub_localizer values within the combined parcel
            sub_localizer_data = self.sub_localizer.get_fdata()
            sub_localizer_within_parcel = sub_localizer_data[combined_parcel == 1]
            
            # Calculate the threshold for the top percentile of sub_localizer values within the parcel
            top_x_percent_threshold = np.nanpercentile(sub_localizer_within_parcel, 100 - self.top_perc)

            # Create boolean masks for left and right hemisphere parcels
            mask_l = self.parcel['l'].get_fdata() == 1
            mask_r = self.parcel['r'].get_fdata() == 1

            # Initialize empty arrays for storing binary masks
            roi = {'l': np.zeros_like(sub_localizer_data), 'r': np.zeros_like(sub_localizer_data)}

            # Apply the threshold to create binary masks for each hemisphere
            roi['l'][mask_l] = (sub_localizer_data[mask_l] >= top_x_percent_threshold).astype(int)
            roi['r'][mask_r] = (sub_localizer_data[mask_r] >= top_x_percent_threshold).astype(int)
            
            # Convert the binary masks to NIfTI images
            roi['l'] = nib.Nifti1Image(roi['l'], self.sub_localizer.affine, self.sub_localizer.header)
            roi['r'] = nib.Nifti1Image(roi['r'], self.sub_localizer.affine, self.sub_localizer.header)
        
        else:
            # For non-split parcels, apply the mask and threshold
            sub_localizer_data = self.sub_localizer.get_fdata()
            sub_localizer_within_parcel = sub_localizer_data[self.parcel.get_fdata() == 1]
            
            # Calculate the threshold for the top percentile of sub_localizer values within the parcel
            top_x_percent_threshold = np.nanpercentile(sub_localizer_within_parcel, 100 - self.top_perc)
            
            # Create a boolean mask for the parcel
            mask = self.parcel.get_fdata() == 1
            roi = np.zeros_like(sub_localizer_data)

            # Apply the threshold to create a binary mask
            roi[mask] = (sub_localizer_data[mask] >= top_x_percent_threshold).astype(int)
            
            # Convert the binary mask to a NIfTI image
            roi = nib.Nifti1Image(roi, self.sub_localizer.affine, self.sub_localizer.header)

        return roi

    def create_ROI(self):
        # Combine the parcel and localizer information to create an ROI mask using the top percentage threshold
        if self.parcel_name == 'v1':
            roi = self.parcel
        else:
            roi = self._combine_parcel_localizer()
        
        return roi

    def _get_roi_size(self):
        if isinstance(self.roi, dict):
            return np.where(self.roi['l'].get_fdata()==1)[0].shape, np.where(self.roi['r'].get_fdata()==1)[0].shape
        else:
            return np.where(self.roi.get_fdata()==1)[0].shape
    
    def plot_roi(self, plot="glass"):
        if plot == "glass":
            plotting.plot_glass_brain(self.sub_localizer, colorbar=True, plot_abs=False, display_mode='lyrz', title="sub_localizer", threshold=1.3)
            if isinstance(self.roi, dict):
                plotting.plot_glass_brain(self.parcel['l'], colorbar=True, plot_abs=False, display_mode='lyrz', title="parcel")
                plotting.plot_glass_brain(self.parcel['r'], colorbar=True, plot_abs=False, display_mode='lyrz', title="parcel")
                plotting.plot_glass_brain(self.roi['l'], colorbar=True, plot_abs=False, display_mode='lyrz', title="roi")
                plotting.plot_glass_brain(self.roi['r'], colorbar=True, plot_abs=False, display_mode='lyrz', title="roi")
            else:
                plotting.plot_glass_brain(self.parcel, colorbar=True, plot_abs=False, display_mode='lyrz', title="parcel")
                plotting.plot_glass_brain(self.roi, colorbar=True, plot_abs=False, display_mode='lyrz', title="roi")
            plt.show()
        elif plot == "surf":
                plot_on_surf(self.sub_localizer, thres=0.01)
                if isinstance(self.roi, dict):
                    plot_on_surf(self.roi['l'], thres=0.01)
                    plot_on_surf(self.roi['r'], thres=0.01)
                else:
                    plot_on_surf(self.roi, thres=0.01)

                plt.show()


class SubjROIs:
    """
    Class to manage all ROIs for a single subject: generates masks, computes reliability,
    supports saving/loading, and runs ROI-based split-half RSA analyses.
    """
    def __init__(self, sub_id, roi_names, space='MNI152NLin2009cAsym', top_perc=10, overwrite=False):
        self.sub_id = sub_id
        self.roi_names = roi_names
        self.space = space
        self.top_perc = top_perc
        self.overwrite = overwrite
        self.subjROIs = {}  # To hold all ROIs for this subject

        self.roi_sizes = {}

        if overwrite:
            self.create_rois()
            self.save_subjROIs()
            print(f"{self.sub_id}: Created and saved ROIs (overwrite=True)")
        else:
            try:
                self.load_saved_rois()
                print(f"{self.sub_id}: Loaded saved ROIs")
            except FileNotFoundError:
                self.create_rois()
                self.save_subjROIs()
                print(f"{self.sub_id}: Created and saved ROIs (fallback)")

    def create_rois(self, plot_mode=None):
        print(f"\n{self.sub_id}")
        for task, roi_name in self.roi_names:
            if self.sub_id == "M23" and task == "physics":
                continue

            if task not in self.subjROIs:
                self.subjROIs[task] = {}
                self.roi_sizes[task] = {}

            # Create ROI object for the current task and roi_name
            self.roi_obj = ROI(self.sub_id, roi_name, task, self.top_perc, self.space)


            if isinstance(self.roi_obj.roi, dict):
                self.subjROIs[task][f'{roi_name}_l'], self.subjROIs[task][f'{roi_name}_r'] = self.roi_obj.roi['l'], self.roi_obj.roi['r']
                self.roi_sizes[task][f'{roi_name}_l'], self.roi_sizes[task][f'{roi_name}_r'] = self.roi_obj._get_roi_size()
            else:
                # If it's a single parcel, store it directly
                self.subjROIs[task][roi_name] = self.roi_obj.roi
                self.roi_sizes[task][roi_name] = self.roi_obj._get_roi_size()

            # Plot the ROI
            self.roi_obj.plot_roi(plot_mode)

    def save_subjROIs(self):
        # Save subject masks for this subject
        out_f = f'../derivatives/nilearn_analysis/glmsingle_betas/sub-{self.sub_id}/sub-{self.sub_id}_space-{self.space}_topperc-{self.top_perc}_desc-subjectrois'

        if not os.path.isfile(out_f) or self.overwrite:
            with open(out_f, 'wb') as f:
                pickle.dump(self.subjROIs, f)
            print(f"File written: {out_f}")

    def load_saved_rois(self):
        in_f = f'../derivatives/nilearn_analysis/glmsingle_betas/sub-{self.sub_id}/sub-{self.sub_id}_space-{self.space}_topperc-{self.top_perc}_desc-subjectrois'
        with open(in_f, 'rb') as f:
            self.subjROIs = pickle.load(f)

    def get_roi_sizes(self):
        if not self.roi_sizes:
            self.roi_sizes = {
                task: {
                    roi_name: np.where(mask.get_fdata()==1)[0].shape
                    for roi_name, mask in roi_dict.items()
                }
                for task, roi_dict in self.subjROIs.items()
            }
        return self.roi_sizes


    def make_dice_coeff_table(self, roi_list=None, ax=None, plot=True): 

        def dice_coefficient(roi1, roi2):
            """
            Calculate Dice coefficient between two binary ROIs.
            
            Parameters:
            roi1, roi2 : nibabel Nifti1Image
                The two ROI images to compare, should be binary (0, 1).
            
            Returns:
            dice : float
                Dice coefficient, a value between 0 and 1.
            """
            # Load ROI data
            data1 = roi1.get_fdata().astype(bool)
            data2 = roi2.get_fdata().astype(bool)

            # Compute intersection and Dice coefficient
            intersection = np.logical_and(data1, data2).sum()
            denominator = data1.sum() + data2.sum()
            if denominator == 0:
                return np.nan

            dice = (2. * intersection) / denominator
            return np.round(dice,2)
        
        # Collect ROI masks using ONLY roi names
        if roi_list is not None:
            roi_source = {}
            for task, name in roi_list:
                roi_source.setdefault(task, {})
                roi_source[task][name] = self.subjROIs[task][name]
        else:
            roi_source = self.subjROIs

        roi_imgs = {}
        for task, roi_dict in roi_source.items():
            for roi_name, roi_img in roi_dict.items():
                roi_imgs[roi_name] = roi_img  # overwrite is fine if duplicated

        if roi_list is not None:
            roi_names = [name for _, name in roi_list]
        else:
            roi_names = list(roi_imgs.keys())
        dice = pd.DataFrame(np.nan, index=roi_names, columns=roi_names)

        # Fill diagonal
        for r in roi_names:
            dice.loc[r, r] = np.nan

        # Compute Dice
        for r1, r2 in itertools.combinations(roi_names, 2):
            d = dice_coefficient(roi_imgs[r1], roi_imgs[r2])
            dice.loc[r1, r2] = d
            dice.loc[r2, r1] = d

        # Plot lower triangle only
        if not plot:
            return dice
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones(dice.shape, dtype=bool), k=1)
        sns.heatmap(dice,mask=mask,cmap="Blues",annot=True,vmin=0,vmax=1,square=True,
            cbar_kws={"label": "Dice coefficient"},ax=ax)
        ax.set_title(f"ROI overlap (Dice) — {self.sub_id}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        return dice

    def compute_split_half_reliability(self):
        # Load trial data for each subject
        with open(f'../derivatives/nilearn_analysis/glmsingle_betas/sub-{self.sub_id}/sub-{self.sub_id}_task-main_space-{self.space}_stat-alltrialsdf', 'rb') as f:
            trials_df = pickle.load(f) 
            trials_df = trials_df[trials_df.trial_type == "experimental_trial"]

        # Initialize lists to store even and odd split data
        even = []
        odd = []

        for video, group in trials_df.groupby('identifier'):
            group = group.reset_index(drop=True)
            if len(group) > 1:
                odd_rows = group[group.index % 2 != 0]
                even_rows = group[group.index % 2 == 0]
                odd.append(odd_rows['betas'].mean(axis=0).flatten())
                even.append(even_rows['betas'].mean(axis=0).flatten())

        even = np.array(even).T
        odd = np.array(odd).T

        # Initialize dictionary to store split-half Pearson correlation for each ROI
        self.split_half_r = {}

        for task, roi_dict in self.subjROIs.items():
            for roi_name, roi_img in roi_dict.items():
                # Retrieve voxel mask for the current ROI and flatten to get relevant voxel indices
                mask_data = roi_img.get_fdata().flatten()
                roi_voxel_indices = np.where(mask_data > 0)[0] # Indices where mask is non-zero
                
                # Calculate split-half Pearson correlation for each voxel in the ROI
                r_values = []
                for voxel_id in roi_voxel_indices: #vmpfc voxels sometimes give warning..
                    if not np.isfinite(even[voxel_id, :]).all() or not np.isfinite(odd[voxel_id, :]).all():
                        continue
                    if np.all(even[voxel_id, :] == even[voxel_id, 0]) or np.all(odd[voxel_id, :] == odd[voxel_id, 0]):
                        continue
                    r, _ = pearsonr(even[voxel_id, :], odd[voxel_id, :])
                    r_values.append(max(r, 0))  # Clip negatives to zero

                self.split_half_r[roi_name] = np.nanmean(r_values) if len(r_values) else np.nan

        return self.split_half_r

    def generate_reliable_rois(self, p_thres=1, r_thres=0):
        reliability_mask_im = load_reliability_mask(self.sub_id, p_thres=p_thres, r_thres = r_thres)
        self.reliable_rois = copy.deepcopy(self.subjROIs)
        for task, areas in self.subjROIs.items():
            for roi_name in areas:
                self.reliable_rois[task][roi_name] = mask_img(areas[roi_name], reliability_mask_im)

    def compute_split_half_RSA(self, within_reliable = None):
        # Load trial data
        with open(f'../derivatives/nilearn_analysis/glmsingle_betas/sub-{self.sub_id}/sub-{self.sub_id}_task-main_space-{self.space}_stat-alltrialsdf', 'rb') as f:
            trials_df = pickle.load(f) 
            trials_df = trials_df[trials_df.trial_type == "experimental_trial"]

        # Load an example affine image for the subject
        affine_image = nib.load(f'../derivatives/fmriprep/sub-{self.sub_id}/func/sub-{self.sub_id}_task-main_run-1_space-{self.space}_desc-preproc_bold.nii.gz')
    
        # Split trials into even and odd groups
        even, odd = [], []
        for _, group in trials_df.groupby('identifier'):
            group = group.reset_index(drop=True)
            if len(group) > 1:
                odd_rows = group[group.index % 2 != 0]
                even_rows = group[group.index % 2 == 0]
                odd.append(odd_rows['betas'].mean())
                even.append(even_rows['betas'].mean())
        
        # Convert even and odd lists to DataFrames with a single column 'betas'
        even_df = pd.DataFrame({'betas': pd.Series(list(even))})
        odd_df = pd.DataFrame({'betas': pd.Series(list(odd))})

        # Optionally restrict analysis to reliable voxels
        if within_reliable:
            self.generate_reliable_rois(p_thres=within_reliable)
            rois_to_use = self.reliable_rois
        else:
            rois_to_use = self.subjROIs

        # Get ROI representations from even and odd splits
        even_repr_df = even_df['betas'].apply(
            lambda x: pd.Series(self.get_roi_representation(
                nib.Nifti1Image(x, affine_image.affine), roi_masks = rois_to_use
            ))
        )
        odd_repr_df = odd_df['betas'].apply(
            lambda x: pd.Series(self.get_roi_representation(
                nib.Nifti1Image(x, affine_image.affine), roi_masks = rois_to_use
            ))
        )

        # Initialize dictionary to store split-half RDM correlations for each ROI
        self.rsa_r = {}
        for task, roi_dict in rois_to_use.items():
            for roi_name in roi_dict.keys():
                roi_mask = roi_dict[roi_name]
                roi_size = np.sum(roi_mask.get_fdata() == 1) #sometimes after reliabilitiy mask it becomes too small to do RSA
                if roi_size > 2:
                    # Generate RDMs for the ROI in even and odd splits
                    even_rdm = get_rdm(np.vstack(even_repr_df[roi_name].to_numpy()), plot=False, method="pearsonr")
                    odd_rdm = get_rdm(np.vstack(odd_repr_df[roi_name].to_numpy()), plot=False, method="pearsonr")
                    # Compute Spearman correlation between the RDMs as the split-half RSA noise ceiling
                    r, _ = spearmanr(even_rdm, odd_rdm)
                    self.rsa_r[roi_name] = r
                else:
                    self.rsa_r[roi_name] = np.nan

        return self.rsa_r

    def get_roi_representation(self, condition_i_image, valid_voxel_mask=None, roi_masks=None):
        """
        Extracts the ROI-wise representation for a single condition image using subject-specific ROIs.
        
        Parameters:
        - condition_i_image (Nifti1Image): The fMRI image for the condition of interest.
        - valid_voxel_mask (3D array or None): Optional binary mask to further restrict voxels.
        - roi_masks: eg. self.reliable_rois instead of self.subjROIs.
        
        Returns:
        - roi_repr (dict): Dictionary where keys are ROI names and values are extracted voxel data arrays.
        """
        roi_repr = {}

        if roi_masks == None:
            roi_masks = self.subjROIs

        for task, roi_dict in roi_masks.items():
            for roi_name, roi_img in roi_dict.items():
                roi_data = roi_img.get_fdata().astype(bool)

                if valid_voxel_mask is not None:
                    combined_mask = roi_data & valid_voxel_mask
                else:
                    combined_mask = roi_data

                if not np.any(combined_mask):
                    print(f"Warning: No valid voxels found for ROI {roi_name}")
                    continue

                combined_mask_img = nib.Nifti1Image(combined_mask.astype(np.int32), roi_img.affine, roi_img.header)
                masked_data = mask_img(condition_i_image, combined_mask_img)
                roi_repr[roi_name] = masked_data.get_fdata()[combined_mask]

        return roi_repr



import warnings
warnings.simplefilter("ignore", category=RuntimeWarning) #for warnings during nan slices subtraction in betas

if __name__ == "__main__":

    subj_group = "M"
    if subj_group == "M":
        sub_ids = ["M03", "M04", "M05", "M06", "M08", "M09", "M10", "M11", "M12", "M13", "M15", "M17", "M18", "M19", "M20","M21", "M22", "M23", "M24", "M25","M26", "M27", "M28", "M29", "M30"]
    else:
        sub_ids = ["P01", "P02", "P04", "P07", "M01"]

    roi_names = [('sipsts','asts'), ('sipsts','psts'), ('tom','tpj'), ("sipsts",'mt'), 
    ('tom','dmpfc'), ('tom','mmpfc'),('tom','vmpfc'), (None, 'v1'), ('physics', 'physics_pramod')] #note that after running create_rois, or if loading existing rois, roi names become name_l/r

    overwrite = False

    split_half_r, split_half_rsa, split_half_rsa_withinreliable = {}, {}, {}
    for sub_id in sub_ids:
        subjROIs = SubjROIs(sub_id, roi_names, overwrite=overwrite)
        
        print(subjROIs.get_roi_sizes())

        # Split-half reliabilitiy
        split_half_r[sub_id] = subjROIs.compute_split_half_reliability()

        # Split-half reliabilitiy
        split_half_rsa_withinreliable[sub_id] = subjROIs.compute_split_half_RSA(within_reliable=1)
        split_half_rsa[sub_id] = subjROIs.compute_split_half_RSA()
        

    with open(f'../derivatives/nilearn_analysis/reliability/{subj_group}set_roiwise_splithalfreliability', 'wb') as f:
        pickle.dump(split_half_r, f)
    with open(f'../derivatives/nilearn_analysis/reliability/{subj_group}set_roiwise_splithalfrsa_withinreliable', 'wb') as f:
        pickle.dump(split_half_rsa_withinreliable, f)
    with open(f'../derivatives/nilearn_analysis/reliability/{subj_group}set_roiwise_splithalfrsa', 'wb') as f:
        pickle.dump(split_half_rsa, f)


    # Make reliability plots
    outdir = '../derivatives/plots/reliability/group'
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    make_roiwise_boxplot(split_half_r, sub_ids, plot_title="Split-Half Reliability Noise Ceiling Across ROIs", ax=ax)
    plt.ylabel('Pearson Correlation')
    ax.legend(bbox_to_anchor=(1.1, 1.05), loc="upper center")
    plt.savefig(f'{outdir}/roiwise_splithalfr_group{subj_group}.png', bbox_inches="tight", dpi=300)
    plt.show()


    fig, ax = plt.subplots(figsize=(10, 6))
    make_roiwise_boxplot(split_half_rsa, sub_ids, plot_title=f"Split-Half RSA Noise Ceiling Across ROIs", ax=ax)
    plt.ylabel('Spearman Correlation (b/w) RDMs')
    ax.legend(bbox_to_anchor=(1.1, 1.05), loc="upper center")
    plt.savefig(f'{outdir}/roiwise_splithalfrsa_group{subj_group}.png', bbox_inches="tight", dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    make_roiwise_boxplot(split_half_rsa_withinreliable, sub_ids, plot_title=f"Split-Half RSA Noise Ceiling Across ROIs within Reliable", ax=ax)
    plt.ylabel('Spearman Correlation (b/w) RDMs')
    ax.legend(bbox_to_anchor=(1.1, 1.05), loc="upper center")
    plt.savefig(f'{outdir}/roiwise_splithalfrsa_withinreliable_group{subj_group}.png', bbox_inches="tight", dpi=300)
    plt.show()


### Optional usage later
#subj = SubjROIs("M13", roi_names)
#subj.load_saved_rois()
#subj.compute_split_half_reliability()
#self.generate_reliable_rois(p_thres=within_reliable)
#rois_to_use = self.reliable_rois
#roi_reprs = subj.get_roi_representation(betas_image, rois_to_use)
