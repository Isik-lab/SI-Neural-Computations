import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from multiprocessing import Pool
import os
import glob
from statsmodels.stats.multitest import multipletests
import argparse

from createROIs import SubjROIs
from roi_utils import ROI_RSA_roiwise, make_roiwise_boxplot, ROI_RSAuniqvar_roiwise, signed_permutation_test_with_fdr
from general_utils import mask_img, get_rdm
from searchlight_utils import process_voxel, process_voxel_uniqvar

def load_features(features2test = [], sub_ids = None):
	""" Load feature representations and align them to the video order
		Handles both subject-specific and global features """

	# Load video names order to order the features
	with open('../derivatives/model_&_behavioural_representations/fname_i_dict', "rb") as f:
		fname_i_dict = pickle.load(f)
		i_fname_dict = {v: k for k, v in fname_i_dict.items()}
	with open(f'../derivatives/nilearn_analysis/glmsingle_betas/sub-M30/sub-M30_task-main_space-MNI152NLin2009cAsym_stat-conditionwise', "rb") as f:
		df = pickle.load(f)
	video_names_inorder = df['identifier'].map(i_fname_dict).to_list()

	# Each block below loads and organizes features in the order of video_names_inorder
	features = {}


	if 'ME10s_reduced' in features2test:
	    from sklearn.decomposition import PCA
	    from sklearn.preprocessing import StandardScaler
	    with open("../derivatives/model_&_behavioural_representations/ME/motion_energies_test_middle10s", "rb") as f:
	            motion_energies = pickle.load(f)

	    motion_energies = {k[:23]:v for k,v in motion_energies.items()}

	    motion_energies_inorder = []
	    for i,x in enumerate(video_names_inorder):
	        motion_energies_inorder.append( motion_energies[x[:23]])

	    X_all = np.vstack(motion_energies_inorder)
	    scaler = StandardScaler()
	    X_all_std = scaler.fit_transform(X_all)

	    # fit PCA
	    n_components = 128
	    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=False)
	    X_all_pca = pca.fit_transform(X_all_std)

	    print("Explained variance ratio (cumulative):", np.cumsum(pca.explained_variance_ratio_)[-1])

	    motion_energies_inorder_reduced = X_all_pca.reshape(50, 200, n_components)
	    
	    features['ME10s_reduced'] = motion_energies_inorder_reduced.mean(axis=1)


	if 'subj_ratings' in features2test:
	    features['subj_ratings'] = {}
	    mapping = {'Friendly': [1, 0, 0], 'Neutral': [0, 1, 0], 'Adversarial': [0, 0, 1], 'InvalidResponse': [0, 0, 0], 'NoResponse': [0, 0, 0]}

	    for sub_id in sub_ids:     
	        with open(f'../derivatives/nilearn_analysis/glmsingle_betas/sub-{sub_id}/sub-{sub_id}_task-main_space-MNI152NLin2009cAsym_stat-conditionwise', "rb") as f:
	            df = pickle.load(f)

	        features['subj_ratings'][sub_id] = np.array(df['response'].map(mapping).to_list())


	if 'HR' in features2test:
	    with open('../derivatives/model_&_behavioural_representations/HR/human_rating_labels_all_genset', 'rb') as file:
	        human_ratings_all = pd.read_pickle(file)
	        human_ratings_all_dict = human_ratings_all.to_dict()

	    # HR
	    HR_wAmb = {}
	    for k,v in human_ratings_all_dict['relationship'].items():
	        temp = {'friendly':0, 'neutral':0, 'adversarial': 0}
	        divisor = float(sum(v.values()))
	        for label, count in v.items():
	            temp[label] = count/divisor

	        id = k.split("/")[-1][:23]
	        HR_wAmb[id] = temp

	    HR = []
	    names = []
	    for i,x in enumerate(video_names_inorder):
	        temp = HR_wAmb[x[:23]]
	        HR.append(list(temp.values()))

	    features['HR'] = np.array(HR)

	if 'HR_PHASE_pg' in features2test: #FIX THIS: create it here instead of using presaved?
		with open('../derivatives/model_&_behavioural_representations/HR_PHASE_pg_features', 'rb') as f:
			features['HR_PHASE_pg'] = np.array(pickle.load(f))

	if 'SocialGNN10s_trained10s' in features2test:
	    with open('../derivatives/model_&_behavioural_representations/SocialGNN/RNN_activations_PHASE_originalsplit_middle10s_contextTrue_20240220_SocialGNN_E_originalsplit_middle10s_21-02-2024', "rb") as f:
	        SocialGNN_act = pickle.load(f)
	    
	    # Order correct and keep only chosen 50 videos
	    SocialGNN_repr = []
	    for i,x in enumerate(video_names_inorder):
	        SocialGNN_repr.append(list(SocialGNN_act[x]))
	        
	    features['SocialGNN10s_trained10s'] = np.array(SocialGNN_repr)

	    
	if 'SIMPLE10s' in features2test:
	    with open('../derivatives/model_&_behavioural_representations/SIMPLE/SIMPLE_probabilities_test10s_origbeliefs_MMcode_run3_9Sept24', "rb") as f:
	        probs_relations_all = pickle.load(f)

	    SIMPLE_genset10s = {}
	    for k,v in probs_relations_all.items():
	        temp = {'friendly':0, 'neutral':0, 'adversarial': 0}
	        for r in v:
	            temp[r[0]] = round(r[1],3)

	        id = k.split("/")[-1][:23]
	        SIMPLE_genset10s[id] = temp

	    # Order correct and keep only chosen 50 videos
	    SIMPLE10s_repr = []
	    for i,x in enumerate(video_names_inorder):
	        temp = SIMPLE_genset10s[x[:23]]
	        SIMPLE10s_repr.append(list(temp.values()))

	    features['SIMPLE10s'] = np.array(SIMPLE10s_repr)

	if 'SIMPLE10s_goals' in features2test:
		with open('../derivatives/model_&_behavioural_representations/SIMPLE/SIMPLE_probabilities_test10s_origbeliefs_MMcode_run3_9Sept24', "rb") as f:
		    _ = pickle.load(f)
		    probs_goals_all = pickle.load(f)


		from general_utils import parse_video_to_abstractgoal_prob
		ABSTRACT_GOAL_LABELS = {
		    1: [
		        'go to landmark',
		        'take object to landmark',
		        'help green agent',
		        'hinder green agent',
		        'get to green agent',
		        'get away from green agent'
		    ],
		    2: [
		        'go to landmark',
		        'take object to landmark',
		        'help red agent',
		        'hinder red agent',
		        'get to red agent',
		        'get away from red agent'
		    ]
		}

		SIMPLE_goals_pred = {}
		for k,v in probs_goals_all.items():
		    id = k.split("/")[-1][:23]
		    SIMPLE_goals_pred[id] = parse_video_to_abstractgoal_prob(v, ABSTRACT_GOAL_LABELS)

		SIMPLE_goals_repr = []
		for i,x in enumerate(video_names_inorder):
		    temp = SIMPLE_goals_pred[x[:23]]
		    SIMPLE_goals_repr.append(temp)

		features['SIMPLE10s_goals'] = np.array(SIMPLE_goals_repr)

	if 'SIMPLE10s_proposalsdists' in features2test:
		from general_utils import parse_video_to_propoposal_dists

		folder = "/Users/mmalik16/Downloads/SocialGNN/SIMPLE-new-main/record/test10s_origbeliefs_MMcode_run3/" 
		pattern = os.path.join(folder, "*sim*.pik")

		ABSTRACT_GOAL_LABELS = {
		    1: [
		        'go to landmark',
		        'take object to landmark',
		        'help green agent',
		        'hinder green agent',
		        'get to green agent',
		        'get away from green agent'
		    ],
		    2: [
		        'go to landmark',
		        'take object to landmark',
		        'help red agent',
		        'hinder red agent',
		        'get to red agent',
		        'get away from red agent'
		    ]
		}

		SIMPLE_dist_all = {}
		for file_path in glob.glob(pattern):
		    with open(file_path, "rb") as f:
		        t = pickle.load(f)
		        id = file_path.split("/")[-1][:23]
		        SIMPLE_dist_all[id] = parse_video_to_propoposal_dists(t['dist_all'], ABSTRACT_GOAL_LABELS)

		SIMPLE_dists_repr = []
		for i,x in enumerate(video_names_inorder):
		    temp = SIMPLE_dist_all[x[:23]]
		    SIMPLE_dists_repr.append(temp)

		features['SIMPLE10s_proposalsdists'] = np.array(SIMPLE_dists_repr)

	if 'VisualRNN10s' in features2test:
	    with open('../derivatives/model_&_behavioural_representations/VisualRNN/RNN_activations_PHASE_originalsplit_middle10s_contextTrue_20250407_CueBasedLSTM_originalsplit_middle10s_07-04-2025', "rb") as f:
	        VisualRNN_act = pickle.load(f)
	    
	    # Order correct and keep only chosen 50 videos
	    VisualRNN_repr = []
	    for i,x in enumerate(video_names_inorder):
	        VisualRNN_repr.append(list(VisualRNN_act[x]))

	    features['VisualRNN10s'] = np.array(VisualRNN_repr)


	if 'SocialGNN10s_trained10s_classifier' in features2test:
	    with open('../derivatives/model_&_behavioural_representations/SocialGNN/classifier_activations_PHASE_originalsplit_middle10s_contextTrue_20240220_SocialGNN_E_originalsplit_middle10s_08-04-2025', "rb") as f:
	        SocialGNN_act = pickle.load(f)
	    
	    # Order correct and keep only chosen 50 videos
	    SocialGNN_repr = []
	    for i,x in enumerate(video_names_inorder):
	        SocialGNN_repr.append(list(SocialGNN_act[x]))
	        
	    features['SocialGNN10s_trained10s_classifier'] = np.array(SocialGNN_repr)

	return features

def get_feature_rdms(features, sub_ids = None):
	""" Convert loaded feature arrays into RDMs for RSA comparison """
	comparison_rdms = {}
	for f_name, feature in features.items():
		if f_name == 'random':
			comparison_rdms[f_name] = np.random.rand(int(feature*(feature-1)/2))
		elif isinstance(feature, dict):
		    comparison_rdms[f_name] = {sub_id: np.array(get_rdm(feature[sub_id])) for sub_id in sub_ids}
		elif f_name == 'SIMPLE10s_proposalsdists': #has nans which pandas can handle better
			correlation_matrix = pd.DataFrame(feature).T.corr(method='pearson')  # Handles NaNs with pairwise deletion
			dist_matrix = 1 - correlation_matrix.to_numpy()
			comparison_rdms[f_name] = [dist_matrix[i, j] for i in range(dist_matrix.shape[0]) for j in range(i)]
		
		else:
		    comparison_rdms[f_name] = np.array(get_rdm(feature))
	return comparison_rdms

def load_conditionwise_betas(sub_ids, mode = None):
	""" Load conditionwise GLMsingle beta maps and valid voxel masks """
	betas_4d_dict = {}
	valid_voxels_mask = {}
	for sub_id in sub_ids:
	    with open(f'../derivatives/nilearn_analysis/glmsingle_betas/sub-{sub_id}/sub-{sub_id}_task-main_space-MNI152NLin2009cAsym_stat-conditionwise', "rb") as f:
	        df = pickle.load(f)

	    mask_image = nib.load(f'../derivatives/fmriprep/sub-{sub_id}/func/sub-{sub_id}_task-main_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
	    mask_data = mask_image.get_fdata() == 1  # This creates a boolean mask where mask equals 1

	    betas = df['betas'].tolist()
	    betas_4d = np.stack(betas, axis=-1)

	    if mode == "images":
	    	betas_4d_dict[sub_id] = nib.Nifti1Image(betas_4d, affine=mask_image.affine)
	    	valid_voxels_mask[sub_id] = mask_image #nans not dealt with here, dealt with inside searchlight analysis
	    else:
	    	valid_voxels_mask[sub_id] = mask_data & ~np.isnan(betas_4d).any(axis=-1) # there could be nans within the brain mask due to signal dropouts?
	    	betas_4d_dict[sub_id] = betas_4d

	return betas_4d_dict, valid_voxels_mask

def get_wholebrain_neural_data(sub_ids, within_reliable=None):
	""" Load neural data and optionally mask with group reliability map """
	betas_images, mask_images = load_conditionwise_betas(sub_ids, mode = "images")

	if within_reliable:
		# Load all reliability maps
		reliability_data = []
		for sub_id in sub_ids:
		    img = nib.load(f"../derivatives/nilearn_analysis/reliability/sub-{sub_id}_task-main_space-MNI152NLin2009cAsym_desc-betas-fracridge_betasnormalize-True_stat-r_statmap.nii.gz")
		    reliability_data.append(img.get_fdata())

		# Stack into 4D array: (n_subjects, x, y, z) & Average across subjects & keep voxels where group mean r > 0
		reliability_array = np.stack(reliability_data, axis=0)
		mean_reliability = np.nanmean(reliability_array, axis=0)
		group_mask_data = (mean_reliability > 0).astype(int)
		group_mask_img = nib.Nifti1Image(group_mask_data, affine=img.affine, header=img.header)

		for sub_id in sub_ids:
			mask_images[sub_id] = mask_img(mask_images[sub_id], group_mask_img)

	return betas_images, mask_images

def get_roiwise_neural_rdms(sub_ids, roi_names, reliable_only=False):
	""" Extract neural RDMs for ROIs, optionally within reliable voxels """
	betas_4d_dict, valid_voxels_mask = load_conditionwise_betas(sub_ids)

	# ROI-RSA
	roiwise_neural_rdms = {}
	for sub_id in sub_ids:
	    ref_img = nib.load(f'../derivatives/fmriprep/sub-{sub_id}/func/sub-{sub_id}_task-main_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
	    condition_imgs = nib.Nifti1Image(betas_4d_dict[sub_id], affine=ref_img.affine)

	    roi_names_subj = [r for r in roi_names if not (sub_id == "M23" and r[0] == "physics")]
	    subj = SubjROIs(sub_id, roi_names_subj)
	    if reliable_only:
	        subj.generate_reliable_rois()
	        roi_masks = subj.reliable_rois
	    else:
	        roi_masks = subj.subjROIs

	    roi_reprs = subj.get_roi_representation(condition_imgs, valid_voxel_mask=valid_voxels_mask[sub_id],roi_masks=roi_masks)
	    
	    # Generate RDM for the ROI and store it in the dictionary
	    roiwise_neural_rdms[sub_id] = {roi_name[1]: np.array(get_rdm(roi_reprs[roi_name[1]].T)) for roi_name in roi_names_subj}

	return roiwise_neural_rdms

def run_searchlight(fmri_data, mask, radius, comparison_rdm):
    """ Searchlight RSA across the whole brain """
    indices = list(itertools.product(range(fmri_data.shape[0]),range(fmri_data.shape[1]),range(fmri_data.shape[2])))
    args = [(x, y, z, fmri_data, mask, radius, comparison_rdm) for x, y, z in indices]

    with Pool(processes=8) as pool: 
        results = pool.map(process_voxel, args)
    
    return results

def run_searchlight_uniqvar(fmri_data, mask, radius, model1_rdm, model2_rdm):
    """ Unique variance RSA searchlight comparing two models """
    indices = list(itertools.product(range(fmri_data.shape[0]),range(fmri_data.shape[1]),range(fmri_data.shape[2])))
    
    args = [(x, y, z, fmri_data, mask, radius, model1_rdm, model2_rdm) for x, y, z in indices]

    with Pool(processes=8) as pool: 
        results = pool.map(process_voxel_uniqvar, args)
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Run RSA analyses with flexible modes and settings.")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--within_reliable', dest='within_reliable', action='store_true',
                       help='Use reliability-masked voxels (default)')
    group.add_argument('--no-within_reliable', dest='within_reliable', action='store_false',
                       help='Do NOT use reliability-masked voxels')
    parser.set_defaults(within_reliable=True)

    parser.add_argument('--mode', type=str, default="ROI", choices=["ROI", "ROIuniqvar", "wholebrain", "wholebrain_uniqvar", "ROImask_on_wholebrain"],
                        help="Choose which analysis mode to run.")
    parser.add_argument('--features', nargs='*', default=[
        'SocialGNN10s_trained10s', 'SIMPLE10s', 'SIMPLE10s_goals', 'HR', 'VisualRNN10s', 'subj_ratings', 
        'SocialGNN10s_trained10s_classifier', 'ME10s_reduced', 'SIMPLE10s_proposalsdists'
    ], help="List of features to test.")
    parser.add_argument('--rois', nargs='*', default=roi_names = [(None, "v1_l"), (None, "v1_r"), ("sipsts", 'mt_l'), ("sipsts", 'mt_r'),
					('sipsts', 'psts_l'), ('sipsts', 'psts_r'), ('tom', 'tpj_l'), ('tom', 'tpj_r'), 
    ], help="List of ROIs to test.")
    return parser.parse_args()


if __name__ == "__main__":
	# Define subject group
	subj_group = "M"
	if subj_group == "M":
		sub_ids = ["M03", "M04", "M05", "M06", "M08", "M09", "M10", "M11", "M12", "M13", "M15", "M17", "M18", "M19", "M20","M21", "M22", "M23", "M24", "M25","M26", "M27", "M28", "M29", "M30"]
	else:
		sub_ids = ["P01", "P02", "P04", "P07", "M01"]

	# Parse command-line arguments
	args = parse_args()
	features2test = args.features
	within_reliable = args.within_reliable
	mode = args.mode
	roi_names = args.rois

	# Load feature RDMs and subject neural data
	features = load_features(features2test, sub_ids)
	comparison_rdms = get_feature_rdms(features, sub_ids)

	os.makedirs('../derivatives/plots/rsa/group', exist_ok=True)

	# ROI-wise standard RSA mode
	if mode == "ROI":

		plot_style = "bar"

		# Load ROI neural RDMs
		roiwise_neural_rdms = get_roiwise_neural_rdms(sub_ids, roi_names, reliable_only=within_reliable)

		# Set up subplots for one plot per ROI
		n = int(np.ceil(len(roi_names) / 2)) #/4
		fig, axs = plt.subplots(n, 2, figsize=(14, n*5)) 
		axs = axs.ravel()

		diff_p_table = {}

		# Loop through ROIs
		for i, roi_name in enumerate(roi_names):
			roi_name = roi_name[1]

			# Skip physics ROIs for M23
			if roi_name in ["physics_pramod_l", "physics_pramod_r"]:
				sub_ids_rel = [s for s in sub_ids if s != "M23"]
			else:
				sub_ids_rel = sub_ids

			# Compute RSA values
			r_values, p_values = ROI_RSA_roiwise(roi_name, sub_ids_rel, roiwise_neural_rdms, comparison_rdms)

			# Prepare data for FDR testing
			reorganized_r = {feature: [r_values[sub][feature] for sub in r_values] for feature in next(iter(r_values.values()))}
			p_uncorrected, p_fdr_corrected = signed_permutation_test_with_fdr(reorganized_r)
			print(roi_name, p_fdr_corrected)

			# Compute pairwise differences for statistical testing
			diff_r = {}
			pairs = itertools.combinations(list(reorganized_r.keys()), 2)
			for f1, f2 in pairs:
				r1 = np.array(reorganized_r[f1])
				r2 = np.array(reorganized_r[f2])
				mask = ~np.isnan(r1) & ~np.isnan(r2)
				diff_r[(f1, f2)] = (r1 - r2)[mask]
			diff_p_uncorrected, diff_p_fdr_corrected = signed_permutation_test_with_fdr(diff_r, two_tailed=True)
			diff_p_table[roi_name] = diff_p_fdr_corrected # Store significant differences

			# Load noise ceiling (split-half RSA values)
			reliability_path = '../derivatives/nilearn_analysis/reliability/Mset_roiwise_splithalfrsa_withinreliable' if within_reliable else '../derivatives/nilearn_analysis/reliability/Mset_roiwise_splithalfrsa'
			with open(reliability_path, 'rb') as f:
				split_half_rsa_values = pickle.load(f)

			# Plot RSA boxplot
			make_roiwise_boxplot(r_values, sub_ids_rel, plot_title=roi_name, ax=axs[i], add_noiseceiling=split_half_rsa_values,
								fdr_pvals=p_fdr_corrected, diff_fdr_pvals=None, visualtype=plot_style)

		# Final layout and save
		handles, labels = axs[0].get_legend_handles_labels()
		plt.tight_layout()
		fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=9)
		plt.suptitle("Standard RSA Scores", fontsize=16, y=1.03)
		reliability_suffix = "_withinreliablevoxels" if within_reliable else ""
		plt.savefig(f'../derivatives/plots/rsa/group/roiRSA_group{subj_group}{reliability_suffix}_{plot_style}.png',
					bbox_inches="tight", dpi=300)
		plt.show()

		# Show pairwise p value table
		df_pvals = pd.DataFrame.from_dict(diff_p_table, orient='index')
		df_pvals.columns = [f"{a}_vs_{b}" for (a, b) in df_pvals.columns]
		df_pvals = df_pvals.sort_index(axis=1)
		df_pvals = df_pvals.round(4) 
		pd.set_option('display.max_columns', None)
		pd.set_option('display.width', None)
		print(df_pvals)

	# ROI-wise unique variance RSA mode
	elif mode == "ROIuniqvar":

		plot_style = "bar"

		roiwise_neural_rdms = get_roiwise_neural_rdms(sub_ids, roi_names, reliable_only=within_reliable)
		n = int(np.ceil(len(roi_names) / 2))
		fig, axs = plt.subplots(n, 2, figsize=(10, n * 3))
		axs = axs.ravel()

		for i, roi_name in enumerate(roi_names):
			roi_name = roi_name[1]

			# Skip physics ROIs for M23
			if roi_name in ["physics_pramod_l", "physics_pramod_r"]:
				sub_ids_rel = [s for s in sub_ids if s != "M23"]
			else:
				sub_ids_rel = sub_ids

			# Define model comparisons
			sr_comparisons_rdms = {
				pair: (comparison_rdms[pair[0]], comparison_rdms[pair[1]])
				for pair in [('SocialGNN10s_trained10s', 'SIMPLE10s')]
			}

			# Run unique variance RSA
			sr_values = ROI_RSAuniqvar_roiwise(roi_name, sub_ids_rel, roiwise_neural_rdms, sr_comparisons_rdms)

			# FDR tests
			reorganized_sr = {feature: [sr_values[sub][feature] for sub in sr_values] for feature in next(iter(sr_values.values()))}
			p_uncorrected, p_fdr_corrected = signed_permutation_test_with_fdr(reorganized_sr)

			diff_sr = {}
			pairs = itertools.combinations(list(reorganized_sr.keys()), 2)
			for f1, f2 in pairs:
				r1 = np.array(reorganized_sr[f1])
				r2 = np.array(reorganized_sr[f2])
				mask = ~np.isnan(r1) & ~np.isnan(r2)
				diff_sr[(f1, f2)] = (r1 - r2)[mask]
			diff_p_uncorrected, diff_p_fdr_corrected = signed_permutation_test_with_fdr(diff_sr, two_tailed=True)

			# Report significant differences
			print(f"\n{roi_name}")
			for pair, p in diff_p_fdr_corrected.items():
				if p < 0.05:
					print(f"{pair}: {'***' if p < 0.001 else '**' if p < 0.01 else '*' }")
				else:
					print(p)

			# Load noise ceiling if applicable
			reliability_path = '../derivatives/nilearn_analysis/reliability/Mset_roiwise_splithalfrsa_withinreliable' if within_reliable else '../derivatives/nilearn_analysis/reliability/Mset_roiwise_splithalfrsa'
			with open(reliability_path, 'rb') as f:
				split_half_rsa_values = pickle.load(f)

			# Plot unique variance RSA boxplot
			make_roiwise_boxplot(sr_values, sub_ids_rel, plot_title=roi_name, ax=axs[i], add_noiseceiling=None,
								fdr_pvals=p_fdr_corrected, diff_fdr_pvals=None, visualtype=plot_style, ylabel = 'Semipartial Correlation (sr)')

		# Final layout and save
		handles, labels = axs[0].get_legend_handles_labels()
		plt.tight_layout()
		fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=9)
		plt.suptitle("Standard RSA UniqVar Scores", fontsize=16, y=1.03)
		reliability_suffix = "_withinreliablevoxels" if within_reliable else ""
		plt.savefig(f'../derivatives/plots/rsa/group/roiRSAuniqvar_group{subj_group}{reliability_suffix}_{plot_style}.png',
					bbox_inches="tight", dpi=300)
		plt.show()

	# Whole-brain RSA searchlight mode
	elif mode == "wholebrain":
		radius = 3
		n_permutations = 0  # Currently unused but placeholder for future tests

		# Load 4D beta images and mask for each subject
		beta_images, mask_images = get_wholebrain_neural_data(sub_ids, within_reliable)

		# Run searchlight RSA per feature and subject
		for f_name, feature in comparison_rdms.items():
			for sub_id in tqdm(sub_ids):

				comparison_rdm_vector = feature[sub_id] if isinstance(feature, dict) else feature
				fmri_img = beta_images[sub_id]
				mask = mask_images[sub_id]

				# Run voxelwise RSA
				results = run_searchlight(fmri_img, mask, radius, comparison_rdm_vector)

				# Fill output volume
				correlation_map = np.zeros(fmri_img.shape[:3])
				for x, y, z, corr in results:
					correlation_map[x, y, z] = corr
				correlation_img = nib.Nifti1Image(correlation_map, affine=mask.affine)

				# Save result NIfTI image
				output_dir = f'../derivatives/nilearn_analysis/rsa/sub-{sub_id}/'
				os.makedirs(output_dir, exist_ok=True)
				reliable_suffix = '_withinreliablevoxels' if within_reliable else ''
				outfile = output_dir + f'sub-{sub_id}_searchlightRSA-{f_name}_radius-{radius}_stat-rmap{reliable_suffix}.nii.gz'
				nib.save(correlation_img, outfile)

	# Whole-brain unique variance RSA
	elif mode == "wholebrain_uniqvar":
		radius = 3
		# Model comparisons for unique variance RSA
		comparisons = [('HR', 'ME10s_reduced'),]

		# Load data
		beta_images, mask_images = get_wholebrain_neural_data(sub_ids, within_reliable)

		for feature1, feature2 in comparisons:
			for sub_id in tqdm(sub_ids):
				# Extract feature vectors for subject
				f1_vec = comparison_rdms[feature1][sub_id] if isinstance(comparison_rdms[feature1], dict) else comparison_rdms[feature1]
				f2_vec = comparison_rdms[feature2][sub_id] if isinstance(comparison_rdms[feature2], dict) else comparison_rdms[feature2]
				fmri_img = beta_images[sub_id]
				mask = mask_images[sub_id]

				# Compute unique variance maps
				results = run_searchlight_uniqvar(fmri_img, mask, radius, f1_vec, f2_vec)
				sr1_map = np.zeros(fmri_img.shape[:3])
				sr2_map = np.zeros(fmri_img.shape[:3])
				for x, y, z, sr1, sr2 in results:
					sr1_map[x, y, z] = sr1
					sr2_map[x, y, z] = sr2

				# Save NIfTI images
				output_dir = f'../derivatives/nilearn_analysis/rsa/sub-{sub_id}/'
				os.makedirs(output_dir, exist_ok=True)
				reliable_suffix = '_withinreliablevoxels' if within_reliable else ''
				sr1_outfile = output_dir + f'sub-{sub_id}_searchlightRSA-{feature1}{feature2}_radius-{radius}_stat-srmap{reliable_suffix}.nii.gz'
				sr2_outfile = output_dir + f'sub-{sub_id}_searchlightRSA-{feature2}{feature1}_radius-{radius}_stat-srmap{reliable_suffix}.nii.gz'
				nib.save(nib.Nifti1Image(sr1_map, affine=mask.affine), sr1_outfile)
				nib.save(nib.Nifti1Image(sr2_map, affine=mask.affine), sr2_outfile)



	    

