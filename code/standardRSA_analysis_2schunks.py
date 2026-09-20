from standardRSA_analysis import load_features, get_feature_rdms
import numpy as np
import matplotlib.pyplot as plt
from createROIs import SubjROIs
from roi_utils import ROI_RSA_roiwise, ROI_RSAuniqvar_roiwise, signed_permutation_test_with_fdr
from general_utils import get_rdm
import pickle
import nibabel as nib
import pandas as pd
import seaborn as sns
from scipy.stats import sem, t
import argparse
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

def load_conditionwise_betas_chunks(sub_ids, mode = None, chunk="0-2s"):
	betas_4d_dict = {}
	valid_voxels_mask = {}
	for sub_id in sub_ids:
	    with open(f'../derivatives/analyses/processed_betas_chunks/sub-{sub_id}/sub-{sub_id}_task-main_chunk-{chunk}_space-MNI152NLin2009cAsym_stat-conditionwise', "rb") as f:
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


def get_roiwise_neural_rdms_chunks(sub_ids, roi_names, reliable_only=False, chunk="0-2s"):
	betas_4d_dict, valid_voxels_mask = load_conditionwise_betas_chunks(sub_ids, chunk=chunk)

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


def plot_roi_barplot_with_significance(ax, df, roi_name):
    """
    Plot RSA barplot with chunk-level grouping and FDR-corrected significance stars.
    - Reds for SIMPLE10s, Blues for SocialGNN10s_trained10s
    - No legend; chunk labels shown under each bar
    """

    sns.set(style="whitegrid")

    # Orders
    features = list(pd.unique(df["Feature"]))
    chunks   = list(pd.unique(df["Chunk"]))

    # Per-bar hue key
    df = df.copy()
    df["HueKey"] = df["Feature"].astype(str) + "|" + df["Chunk"].astype(str)

    # Present (Feature, Chunk) pairs in seaborn draw order: Feature-major, Chunk-minor
    f_idx = {f: i for i, f in enumerate(features)}
    c_idx = {c: i for i, c in enumerate(chunks)}
    pairs = (df[["Feature", "Chunk"]]
             .drop_duplicates()
             .assign(_f=lambda x: x["Feature"].map(f_idx),
                     _c=lambda x: x["Chunk"].map(c_idx))
             .sort_values(["_f", "_c"]))
    hue_order = (pairs["Feature"] + "|" + pairs["Chunk"]).tolist()

    # Palette map: Reds for SIMPLE10s, Blues for SocialGNN..., shades by chunk index
    palette_map = {}
    # Precompute color ramps per feature
    base_by_feat = {
        "SIMPLE10s": "Reds",
        "SocialGNN10s_trained10s": "Blues",
        "SocialGNN10s_trained10s-SIMPLE10s": "Blues",
        "SocialGNN10s_trained10s-SIMPLE10s_goals": "Blues",
        "SIMPLE10s-SocialGNN10s_trained10s": "Reds",
        "SIMPLE10s_goals-SocialGNN10s_trained10s": "Reds",
    }
    ramp_by_feat = {}
    for feat in features:
        base = base_by_feat.get(feat, "Greys")
        ramp_by_feat[feat] = sns.color_palette(base, n_colors=max(1, len(chunks)))

    for key in hue_order:
        feat, ch = key.split("|", 1)
        palette_map[key] = ramp_by_feat[feat][c_idx[ch]]

    # ---- Plot: each bar as its own category (tight spacing) ----
    sns.barplot(
        data=df,
        x="HueKey",
        y="r",
        order=hue_order,
        palette=palette_map,
        errorbar=("ci", 95),
        width=1,
        ax=ax
    )

    # Title/axes basics
    ax.set_title(f"{roi_name}", fontsize=14)
    ax.set_ylabel("RSA r-value")
    ax.set_xlabel("")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x")

    # Remove legend
    leg = ax.get_legend()
    if leg:
        leg.remove()

    # Centers of drawn bars (same order as hue_order)
    bars = [p for p in ax.patches if p.get_height() is not None]
    bar_centers = [p.get_x() + p.get_width() / 2 for p in bars]

    # Chunk labels under each bar
    labels_for_bars = [k.split("|", 1)[1] for k in hue_order[:len(bar_centers)]]
    try:
        ax.set_xticks(bar_centers, labels_for_bars, fontsize=16)  # mpl>=3.5
    except TypeError:
        ax.set_xticks(bar_centers)
        ax.set_xticklabels(labels_for_bars, fontsize=14)


    # Tighten side margins a touch (optional)
    if len(bar_centers) >= 2:
        dx = bar_centers[1] - bar_centers[0]
        ax.set_xlim(bar_centers[0] - 0.8 * dx, bar_centers[-1] + 0.8 * dx)

    # Make space for stars
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.025)

    # --- Significance stars (use actual bar x positions) ---
    grouped = df.groupby(["Feature", "Chunk"])
    bar_means = grouped["r"].mean()
    bar_sizes = grouped.size()
    # handle small n: if n<=1, t-interval is degenerate; fall back to mean
    bar_sems = grouped["r"].apply(lambda s: sem(s, nan_policy="omit") if len(s) > 1 else 0.0)
    tcrit = bar_sizes.apply(lambda n: t.ppf(0.975, n - 1) if n > 1 else 0.0)
    bar_upper = bar_means + bar_sems * tcrit

    # map key -> x
    xpos_by_key = {k: x for k, x in zip(hue_order[:len(bar_centers)], bar_centers)}

    for feat in features:
        for ch in chunks:
            key = f"{feat}|{ch}"
            sub = df[(df["Feature"] == feat) & (df["Chunk"] == ch)]
            if sub.empty:
                continue
            p = sub["p_value"].iloc[0]
            label = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            x = xpos_by_key.get(key)
            if x is None:
                continue
            y = float(bar_upper.get((feat, ch), 0)) + 0.01
            ax.text(x, y, label, ha="center", va="bottom", fontsize=24)

    # Aesthetics
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="y", labelsize=14, width=1.5)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")




def parse_args():
    parser = argparse.ArgumentParser(description="Run RSA analyses with flexible modes and settings.")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--within_reliable', dest='within_reliable', action='store_true',
                       help='Use reliability-masked voxels (default)')
    group.add_argument('--no-within_reliable', dest='within_reliable', action='store_false',
                       help='Do NOT use reliability-masked voxels')
    parser.set_defaults(within_reliable=True)

    parser.add_argument('--mode', type=str, default="ROIuniqvar", choices=["ROI", "ROIuniqvar"],
                        help="Choose which analysis mode to run.")
    parser.add_argument('--features', nargs='*', default=['SocialGNN10s_trained10s', 'SIMPLE10s'], help="List of features to test.")
    return parser.parse_args()


import pingouin as pg
import statsmodels.formula.api as smf

def get_interactioneffect(ax, df, roi_name="ROI"):
	# Convert chunk to categorical and numeric
	df["Chunk"] = df["Chunk"].astype("category")
	df['Chunk_num'] = df['Chunk'].apply(lambda x: int(x.split('-')[0])).astype(float)

	# Repeated measures ANOVA (Chunk x Feature interaction)
	print(f"=== Repeated Measures ANOVA: {roi_name} ===")
	aov = pg.rm_anova(dv="r", within=["Chunk", "Feature"], subject="Sub_id", data=df, detailed=True)
	print(aov.loc[aov['Source'] == "Chunk * Feature", ["Source", "ddof1", "ddof2", "F", "p-unc", "p-GG-corr"]])

	df["t"] = df["Chunk_num"] - df["Chunk_num"].mean()  

	print(f"\n=== MixedLM: do trends/shapes differ? {roi_name} ===")
	fit = smf.mixedlm(
		"r ~ Feature * t + Feature * I(t**2)",  
		data=df,
		groups=df["Sub_id"],
		re_formula="~t"
	).fit(reml=False, method="lbfgs")

	print("\nTrend-difference terms to look at:")
	for term in fit.pvalues.index:
	    if ":t" in term or "I(t ** 2)" in term:
	        print(f"{term}: coef={fit.params[term]:.4f}, p={fit.pvalues[term]:.4g}")

	# Plotting model R-values over chunks
	sns.pointplot(data=df, x="Chunk", y="r", hue="Feature", errorbar='se', dodge=True, ax=ax)
	ax.set_title(f"Feature r-values over time – {roi_name}")
	plt.ylabel("RSA (Spearman r)")
	plt.xlabel("Time Chunk")
	plt.legend(title="Model")


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

	# Load feature RDMs and subject neural data
	features = load_features(features2test, sub_ids)
	comparison_rdms = get_feature_rdms(features, sub_ids)

	chunks = ["0-2s", "2-4s", "4-6s", "6-8s", "8-10s"]

	# ROI-wise standard RSA mode
	if mode == "ROI":
		roi_names = [(None, "evc_l"), (None, "evc_r"), ("sipsts", 'mt_l'), ("sipsts", 'mt_r'),
					('sipsts', 'psts_l'), ('sipsts', 'psts_r'),
					('sipsts', 'asts_l'), ('sipsts', 'asts_r'),
					('tom', 'tpj_l'), ('tom', 'tpj_r'), 
					('physics', 'physics_pramod_l'), ('physics', 'physics_pramod_r'),
					('tom', 'dmpfc'), ('tom', 'mmpfc'), ('tom', 'vmpfc'),]

		plot_style = "bar"

		all_roi_dfs = {}

		# Load ROI neural RDMs
		roiwise_neural_rdms_allchunks = {chunk: get_roiwise_neural_rdms_chunks(
			        sub_ids, roi_names, reliable_only=within_reliable, chunk=chunk) for chunk in chunks}

		# Set up subplots for one plot per ROI
		n = int(np.ceil(len(roi_names)/2))
		fig, axs = plt.subplots(n, 2, figsize=(20, n*5))
		axs = axs.ravel()  # Flatten the 6x2 array for easy iteration

		fig2, axs2 = plt.subplots(n, 2, figsize=(20, n*5))
		axs2 = axs2.ravel()  # Flatten the 6x2 array for easy iteration

		# Loop through ROIs
		for idx, (_, roi_name) in enumerate(roi_names):
			print(f"\nProcessing ROI: {roi_name}")
			records = []

			# Skip physics ROIs for M23
			if roi_name in ["physics_pramod_l", "physics_pramod_r"]:
				sub_ids_rel = [s for s in sub_ids if s != "M23"]
			else:
				sub_ids_rel = sub_ids

			for chunk in chunks:
				# Compute RSA values
				roiwise_neural_rdms = roiwise_neural_rdms_allchunks[chunk]
				r_dict, _ = ROI_RSA_roiwise(roi_name, sub_ids_rel, roiwise_neural_rdms, comparison_rdms)

				# Prepare data for FDR testing
				reorganized_r = {feature: [r_dict[sub][feature] for sub in r_dict] for feature in next(iter(r_dict.values()))}
				p_uncorrected, p_fdr_corrected = signed_permutation_test_with_fdr(reorganized_r)

				# Organize data for plotting
				for sub_id, f_dict in r_dict.items():
					for feature, r_value in f_dict.items():
						records.append({
						    "Sub_id": sub_id,
						    "Feature": feature,
						    "Chunk": chunk,
						    "r": r_value,
						    "p_value": p_fdr_corrected[feature]
						})

			# Plot RSA values across chunks for each model
			df = pd.DataFrame(records)
			all_roi_dfs[roi_name] = df
		
			plot_roi_barplot_with_significance(axs[idx], df, roi_name)

			get_interactioneffect(axs2[idx], df, roi_name)
		
		handles, labels = axs[0].get_legend_handles_labels()
		fig.tight_layout()
		fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=9)
		fig.suptitle(f"Standard RSA Scores", fontsize=16, y=1.03)
		reliability_suffix = "_withinreliablevoxels" if within_reliable else ""
		fig.savefig(f'../derivatives/analyses/plots/rsa/group/roiRSA_chunks_group{subj_group}{reliability_suffix}_{plot_style}.png', bbox_inches="tight", dpi=300)

		handles, labels = axs2[0].get_legend_handles_labels()
		fig2.tight_layout()
		fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=9)
		fig2.suptitle(f"Standard RSA Scores", fontsize=16, y=1.03)
		reliability_suffix = "_withinreliablevoxels" if within_reliable else ""
		fig2.savefig(f'../derivatives/analyses/plots/rsa/group/roiRSA_chunks_group{subj_group}{reliability_suffix}_fittedcurves.png', bbox_inches="tight", dpi=300)
		plt.show()
		

	elif mode == "ROIuniqvar":

		roi_names = [('sipsts', 'psts_r'), ('tom', 'tpj_r')]
		plot_style = "bar"

		all_roi_dfs = {}

		# Load ROI neural RDMs
		roiwise_neural_rdms_allchunks = {chunk: get_roiwise_neural_rdms_chunks(
			        sub_ids, roi_names, reliable_only=within_reliable, chunk=chunk) for chunk in chunks}

		# Set up subplots for one plot per ROI
		n = int(np.ceil(len(roi_names)/2))
		fig, axs = plt.subplots(n, 2, figsize=(20, n*5))
		axs = axs.ravel()  # Flatten the 6x2 array for easy iteration

		fig2, axs2 = plt.subplots(n, 2, figsize=(20, n*5))
		axs2 = axs2.ravel()  # Flatten the 6x2 array for easy iteration

		# Loop through ROIs
		for idx, (_, roi_name) in enumerate(roi_names):
			print(f"Processing ROI: {roi_name}")
			records = []

			# Skip physics ROIs for M23
			if roi_name in ["physics_pramod_l", "physics_pramod_r"]:
				sub_ids_rel = [s for s in sub_ids if s != "M23"]
			else:
				sub_ids_rel = sub_ids

			for chunk in chunks:
				# Compute semipartial RSA values
				roiwise_neural_rdms = roiwise_neural_rdms_allchunks[chunk]
				sr_comparisons_rdms = {pair: (comparison_rdms[pair[0]],comparison_rdms[pair[1]]) for pair in [('SocialGNN10s_trained10s', 'SIMPLE10s')]} #, ('SocialGNN10s_trained10s', 'SIMPLE10s_goals')]} #, ('SocialGNN10s_trained10s', 'ME10s'),  ('SIMPLE10s', 'ME10s')]}
				sr_dict = ROI_RSAuniqvar_roiwise(roi_name, sub_ids_rel, roiwise_neural_rdms, sr_comparisons_rdms)

				# Prepare data for FDR testing
				reorganized_sr = {feature: [sr_dict[sub][feature] for sub in sr_dict] for feature in next(iter(sr_dict.values()))}
				p_uncorrected, p_fdr_corrected = signed_permutation_test_with_fdr(reorganized_sr)

				# Organize data for plotting
				for sub_id, f_dict in sr_dict.items():
					for feature, r_value in f_dict.items():
						records.append({
						    "Sub_id": sub_id,
						    "Feature": feature,
						    "Chunk": chunk,
						    "r": r_value,
						    "p_value": p_fdr_corrected[feature]
						})

			# Plot semipartial RSA values across chunks for each model
			df = pd.DataFrame(records)
			all_roi_dfs[roi_name] = df
			plot_roi_barplot_with_significance(axs[idx], df, roi_name)

			get_interactioneffect(axs2[idx], df, roi_name)
			
		handles, labels = axs[0].get_legend_handles_labels()
		fig.tight_layout()
		fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=9)
		fig.suptitle(f"Standard RSA UniqVar Scores", fontsize=16, y=1.03)
		reliability_suffix = "_withinreliablevoxels" if within_reliable else ""
		fig.savefig(f'../derivatives/analyses/plots/rsa/group/roiRSAuniqvar_chunks_group{subj_group}{reliability_suffix}_{plot_style}.png', bbox_inches="tight", dpi=300)

		handles, labels = axs2[0].get_legend_handles_labels()
		fig2.tight_layout()
		fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=9)
		fig2.suptitle(f"Standard RSA UniqVar Scores", fontsize=16, y=1.03)
		reliability_suffix = "_withinreliablevoxels" if within_reliable else ""
		fig2.savefig(f'../derivatives/analyses/plots/rsa/group/roiRSAuniqvar_chunks_group{subj_group}{reliability_suffix}_fittedcurves.png', bbox_inches="tight", dpi=300)
		plt.show()



