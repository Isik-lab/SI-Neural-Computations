import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import nibabel as nib
from general_utils import mask_img, spearmanr_permutation_test
import scipy
import pingouin as pg
from itertools import combinations
from statsmodels.stats.multitest import multipletests

def make_roiwise_boxplot(r_values, sub_ids, plot_title=None, ax=None, ylabel = 'Spearman Correlation (r)', add_noiseceiling = None, fdr_pvals=None, diff_fdr_pvals = None, visualtype="violin"):
    suppress_plot = False

    # Create axis if none provided
    if ax is None:
        fig = plt.figure(figsize=(6,4))
        ax = plt.gca()

    # --- Compute noise ceiling & early exit ---
    if add_noiseceiling is not None:
        reliability_per_roi = [(2*add_noiseceiling[sub_id][plot_title])/(1+add_noiseceiling[sub_id][plot_title]) for sub_id in sub_ids]
        #reliability_per_roi = [np.sqrt(abs(r_sb))*np.sign(r_sb) for r_sb in reliability_per_roi]
        mean_nc = np.nanmean(reliability_per_roi)

        ax.axhline(mean_nc, color="gray", linestyle='-.', linewidth=2, alpha=0.6, label='Split-Half RSA (mean)')

        min_nc = np.nanpercentile(reliability_per_roi, 25)
        max_nc = np.nanpercentile(reliability_per_roi, 75)
        # Add shaded region for the noise ceiling
        #ax.set_xlim(-0.75, len(r_values_df.columns) - 0.75)
        #ax.fill_between(ax.get_xlim(), min_nc, max_nc, color='gray', alpha=0.3, label='Split-Half RSA 25–75%ile')

        if mean_nc < 0:
            suppress_plot = True

    # Plotting
    r_values_df = pd.DataFrame(r_values).T  # Transpose to have subjects as rows and comparisons as columns
    r_values_df = r_values_df.fillna(np.nan)  # Explicitly keep NaNs
    r_values_melted = r_values_df.reset_index().melt(id_vars='index', var_name='Comparison RDM', value_name='r')

    model2colors = {
        'SocialGNN10s_trained10s': (0.2980392156862745, 0.4470588235294118, 0.6901960784313725), #blue
        'SIMPLE10s': (0.7686274509803922, 0.3058823529411765, 0.3215686274509804), #red
        'SIMPLE10s_goals': (0.90, 0.58, 0.59), #(0.7686274509803922, 0.3058823529411765, 0.3215686274509804), #red
        'SIMPLE10s_proposalsdists': (1, 0.52, 0.55),
        'HR': (0.8666666666666667, 0.5176470588235295, 0.3215686274509804), #orange
        'VisualRNN10s': (0.3333333333333333, 0.6588235294117647, 0.40784313725490196), #green
        'SocialGNN10s_trained10s_classifier': (0.39215686274509803, 0.7098039215686275, 0.803921568627451), #lightblue
        'ME10s': (0.5058823529411764, 0.4470588235294118, 0.7019607843137254), #(0.8, 0.7254901960784313, 0.4549019607843137), #yellowgreen
        'subj_ratings': (0.8549019607843137, 0.5450980392156862, 0.7647058823529411), #pink
        'shuffledHR': (0.5490196078431373, 0.5490196078431373, 0.5490196078431373), #gray
        'SocialGNN10s_trained10s-SIMPLE10s': (0.2980392156862745, 0.4470588235294118, 0.6901960784313725), 
        'SIMPLE10s-SocialGNN10s_trained10s': (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
        'SocialGNN10s_trained10s-SIMPLE10s_goals': (0.2980392156862745, 0.4470588235294118, 0.6901960784313725), 
        'SIMPLE10s_goals-SocialGNN10s_trained10s': (0.90, 0.58, 0.59),
        'ME10s_reduced': (0.5058823529411764, 0.4470588235294118, 0.7019607843137254), 
        'SocialGNN10s_trained10s-ME10s_reduced': (0.2980392156862745, 0.4470588235294118, 0.6901960784313725), 
        'ME10s_reduced-SocialGNN10s_trained10s': (0.5058823529411764, 0.4470588235294118, 0.7019607843137254), 
        'SIMPLE10s-ME10s_reduced': (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
        'ME10s_reduced-SIMPLE10s': (0.5058823529411764, 0.4470588235294118, 0.7019607843137254), 
        'HR-ME10s_reduced': (0.8666666666666667, 0.5176470588235295, 0.3215686274509804), #orange
        'ME10s_reduced-HR': (0.5058823529411764, 0.4470588235294118, 0.7019607843137254), 
        'SocialGNN10s_trained10s-v1_r': (0.2980392156862745, 0.4470588235294118, 0.6901960784313725), 
        'SIMPLE10s-v1_r': (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
        'HR-v1_r': (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
        'SocialGNN10s_trained10s_classifier-SIMPLE10s': (0.39215686274509803, 0.7098039215686275, 0.803921568627451), 
        'SIMPLE10s-SocialGNN10s_trained10s_classifier': (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
        'SocialGNN10s_trained10s-SIMPLE10s_proposalsdists': (0.2980392156862745, 0.4470588235294118, 0.6901960784313725), 
        'SIMPLE10s_proposalsdists-SocialGNN10s_trained10s': (1, 0.52, 0.55),
        }

    # Dealing with keys not in dict above
    np.random.seed(13)
    unique_keys = r_values_melted['Comparison RDM'].unique()
    for key in unique_keys:
        if key not in model2colors:
            model2colors[key] = np.random.rand(3)  # Returns an RGB color with values between 0 and 1
    np.random.seed(None)

    if not suppress_plot:
        if visualtype == "box":
            sns.boxplot(x='Comparison RDM',y='r', hue='Comparison RDM', data=r_values_melted,
                palette="muted", boxprops=dict(alpha=0.8, linewidth=2),whiskerprops={'linewidth': 2},
                capprops={'linewidth': 2}, medianprops={'color': 'gray', 'linewidth': 2},
                ax=ax,legend=False)
        elif visualtype == "violin":
            palette = sns.color_palette("muted", n_colors=len(r_values_df.columns))
            vp = sns.violinplot(x='Comparison RDM', y='r', hue='Comparison RDM', data=r_values_melted,
                inner=None,  # We'll add scatter manually
                linewidth=1.2, cut=0, density_norm="width",
                palette=palette, ax=ax, legend=False)

            for violin in vp.collections:
                violin.set_alpha(0.5)

            group_means = r_values_df.mean(skipna=True)
            violin_colors = [c.get_facecolor()[0] for c in vp.collections[:len(group_means)]]
            for i, (x, y) in enumerate(group_means.items()):
                ax.plot([i - 0.2, i + 0.2], [y, y], color=palette[i], linewidth=2)
        elif visualtype == "bar":
            sns.barplot(x='Comparison RDM', y='r', hue='Comparison RDM', data=r_values_melted,
            estimator=np.mean, errorbar=('ci', 95), ax=ax, palette=model2colors) #, edgecolor='black', linewidth=1.5)

    '''
    # Add individual data points for each subject
    for i, sub_id in enumerate(sub_ids):
        r_values_per_sub = [r_values[sub_id][comp_name] for comp_name in r_values_df.columns]
        r_values_per_sub = np.array(r_values_per_sub, dtype=np.float64)
        mask = ~np.isnan(r_values_per_sub)  # Exclude NaNs
        #ax.scatter(np.arange(len(r_values_df.columns))[mask] + np.random.uniform(-0.15, 0.15, mask.sum()), 
        #    r_values_per_sub[mask], alpha=0.6, label=f'Subject {sub_id}')
        ax.scatter(np.arange(len(r_values_df.columns))[mask] + np.random.uniform(-0.15, 0.15, mask.sum()),
            r_values_per_sub[mask], alpha=0.6, color='slategray')
    '''

    sns.despine(ax=ax, top=True, right=True)
    # Set title, clear x-axis label, and set y-axis label
    ax.set_title(plot_title)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)

    # Add grid and horizontal reference line
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.4)

    # Thicken bottom and left spines
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Bold y-axis tick labels
    ax.tick_params(axis='y', labelsize=12, width=1.5)
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    # Annotate FDR-corrected significance
    if fdr_pvals and not suppress_plot:
        existing_ylim = ax.get_ylim()
        new_top = existing_ylim[1] + 0.02 #max(existing_ylim[1], np.nanmax(r_values_melted["r"]) + 0.05)
        ax.set_ylim(existing_ylim[0], new_top)

        #means = r_values_df.mean(skipna=True)
        # Compute upper 95% CI bound per model
        ci_upper = {model: r_values_df[model].mean() + 
        scipy.stats.sem(r_values_df[model].dropna()) * scipy.stats.t.ppf(0.975, df=r_values_df[model].count() - 1)
            for model in r_values_df.columns}
        for i, model in enumerate(r_values_df.columns):
            p = fdr_pvals.get(model)
            if p is not None:
                y = max(0,ci_upper[model]) + 0.001 #np.nanmax(r_values_df[model]) + 0.005
                label = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                ax.text(i, y, label, ha='center', va='bottom', fontsize=12, fontweight="bold")

    # Annotate pairwise feature differences if diff_pvals are passed
    if diff_fdr_pvals and not suppress_plot:
        means = r_values_df.mean(skipna=True)
        bar_height_base = means.max() + 0.1
        spacing = 0.06  # vertical space between levels
        existing_ylim = ax.get_ylim()

        # Group comparisons by level (distance between bars)
        columns = list(r_values_df.columns)
        level_dict = {}
        for i in range(1, len(columns)):
            level_dict[i] = [(columns[j], columns[j+i]) for j in range(len(columns) - i)]

        max_level = 0
        for level, pairs in level_dict.items():
            plotted = False
            for idx, (f1, f2) in enumerate(pairs):
                p = diff_fdr_pvals.get((f1, f2))
                if p is None or p >= 0.05:
                    continue

                x1, x2 = columns.index(f1)+0.025, columns.index(f2)-0.025
                x_center = (x1 + x2) / 2
                jitter = (idx - len(pairs)/2) * 0.05  if level>1 else 0 # Centered jitter per level
                height = bar_height_base + spacing * (level - 1) + jitter

                if p < 0.001:
                    label = "***"
                elif p < 0.01:
                    label = "**"
                else:
                    label = "*"

                ax.plot([x1, x1, x2, x2],
                        [height - 0.001, height, height, height - 0.001],
                        lw=1.5, c='darkred')
                ax.text(x_center, height, label,
                        ha='center', va='bottom', fontsize=10,
                        fontweight="bold", color="darkred")
                plotted = True
            if plotted:
                max_level = level

        # Adjust y-limits to fit annotations
        ax.set_ylim(existing_ylim[0], max(existing_ylim[1], bar_height_base + spacing * max_level))


    # Adjust x-tick labels to split long labels into two lines
    new_labels = [label.get_text()[:13] + "\n" + label.get_text()[13:] if len(label.get_text()) > 10 else label.get_text()
                  for label in ax.get_xticklabels()]
    ax.set_xticks(range(len(new_labels)))  # Set tick positions based on the number of labels
    ax.set_xticklabels(new_labels, rotation=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    # Only add legend if ax is None to avoid duplicate legends
    if ax is None:
        ax.legend(bbox_to_anchor=(1.1, 1.05), loc="upper center")
        plt.tight_layout()
        plt.show()

def ROI_RSA_roiwise(roi_name, sub_ids, ROIbetas_rdm_vector, comparison_rdms):
    # ROIbetas_rdm_vector is the allsubj_allROIs_rdms
    r_values = {sub_id: {} for sub_id in sub_ids}
    p_values = {sub_id: {} for sub_id in sub_ids}
    
    for sub_id in sub_ids:
        for name, feature_rdm in comparison_rdms.items():
            if np.isnan(ROIbetas_rdm_vector[sub_id][roi_name]).all():
                print(sub_id, name, roi_name, "Warning: ROI has only one voxel, returning NaN RDM.")
                r_values[sub_id][name] = np.nan
                p_values[sub_id][name] = np.nan
                continue
            if isinstance(feature_rdm,dict):
                r, p = scipy.stats.spearmanr(ROIbetas_rdm_vector[sub_id][roi_name], feature_rdm[sub_id])
            else:
                r, p = scipy.stats.spearmanr(ROIbetas_rdm_vector[sub_id][roi_name], feature_rdm)
            r_values[sub_id][name] = r
            p_values[sub_id][name] = p
    
    # Pass the plot_title and ax arguments provided to the function
    #make_roiwise_boxplot(r_values, sub_ids, plot_title=plot_title if plot_title else roi_name, ax=ax,add_noiseceiling = False)

    return r_values, p_values

def ROI_RSAuniqvar_roiwise(roi_name, sub_ids, ROIbetas_rdm_vector, comparison_rdms):
    # ROIbetas_rdm_vector is the allsubj_allROIs_rdms
    sr_values = {sub_id: {} for sub_id in sub_ids}
    
    for sub_id in sub_ids:
        for name, (model1_rdm,model2_rdm) in comparison_rdms.items():
            if np.isnan(ROIbetas_rdm_vector[sub_id][roi_name]).all():
                print(sub_id, name, roi_name, "Warning: ROI has only one voxel, returning NaN RDM.")
                sr_values[sub_id][f"{name[0]}-{name[1]}"] = np.nan
                sr_values[sub_id][f"{name[1]}-{name[0]}"] = np.nan
                continue

            # Compute semi-partial correlations
            data = pd.DataFrame({'X': ROIbetas_rdm_vector[sub_id][roi_name], 'Y1': model1_rdm, 'Y2': model2_rdm})
            sr1 = pg.partial_corr(data=data, x='X', y='Y1', y_covar=['Y2'], method='spearman')['r'].iloc[0]
            sr2 = pg.partial_corr(data=data, x='X', y='Y2', y_covar=['Y1'], method='spearman')['r'].iloc[0]

            sr_values[sub_id][f"{name[0]}-{name[1]}"] = sr1
            sr_values[sub_id][f"{name[1]}-{name[0]}"] = sr2

    return sr_values

def signed_permutation_test_with_fdr(r_values_dict, n_perm=5000, alpha=0.05, two_tailed=False):
    """
    Performs a signed permutation test (mean r > 0) across subjects for each feature.
    Applies FDR correction across features.

    Parameters:
        r_values_dict (dict): Dictionary of {feature_name: list/array of r values across subjects}
        n_perm (int): Number of permutations for null distribution
        alpha (float): Significance level for FDR correction

    Returns:
        p_uncorrected (dict): {feature: uncorrected p-value}
        p_fdr_corrected (dict): {feature: FDR-corrected p-value}
        is_significant (dict): {feature: True/False after FDR correction}
    """
    feature_names = list(r_values_dict.keys())
    p_vals = []

    for feature in feature_names:
        r = np.array(r_values_dict[feature])
        r = r[~np.isnan(r)]  # Remove NaNs
        if len(r) == 0: #basically no subj without nans
            p_vals.append(np.nan)
            continue
        observed_mean = np.mean(r)

        # Generate null distribution: flip signs randomly
        signs = np.random.choice([-1, 1], size=(n_perm, len(r)))
        null_distribution = (signs * r).mean(axis=1)

        if two_tailed:
            # Two-tailed: test if abs(observed) is !=0
            p = (np.sum(np.abs(null_distribution) >= np.abs(observed_mean)) + 1) / (n_perm + 1)
        else:
            # One-tailed: test if observed r is significantly > 0
            p = (np.sum(null_distribution >= observed_mean) + 1) / (n_perm + 1)
        p_vals.append(p)

    # FDR correction
    _, p_fdr_vals, _, _ = multipletests(p_vals, alpha=alpha, method="fdr_bh")

    # Package into dictionaries
    p_uncorrected = dict(zip(feature_names, p_vals))
    p_fdr_corrected = dict(zip(feature_names, p_fdr_vals))

    return p_uncorrected, p_fdr_corrected
