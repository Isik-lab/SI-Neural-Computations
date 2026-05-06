import numpy as np
from scipy.stats import spearmanr
import pingouin as pg
import pandas as pd
import nibabel as nib
from nilearn import plotting, datasets, surface
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map
from nilearn.image import load_img, concat_imgs, math_img, new_img_like
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Apply a binary mask to an image; zero out voxels outside the mask
def mask_img(img, mask):
    if isinstance(img, nib.nifti1.Nifti1Image):
        masked_img = np.array(img.dataobj)
        mask = np.array(mask.dataobj)
    else:
        masked_img = img.copy()

    if isinstance(mask, nib.nifti1.Nifti1Image):
        mask = np.array(mask.dataobj)
    mask = np.invert(mask.astype('bool'))

    i, j, k = np.where(mask)
    masked_img[i, j, k] = 0.
    if isinstance(img, nib.nifti1.Nifti1Image):
        masked_img = nib.Nifti1Image(masked_img, img.affine, img.header)
    return masked_img


# Plot a 3-view surface projection (medial, lateral, ventral)
# Uses intermediate PNGs to combine into a single matplotlib figure
def plot_on_surf(im, thres=None, vmax_inp=None, dpi=300):
    medial_fig, _ = plotting.plot_img_on_surf(stat_map=im, surf_mesh='fsaverage', inflate=True, threshold=thres, views=['medial'], vmax=vmax_inp)
    medial_fig.savefig('medial.png', dpi=dpi, bbox_inches="tight")
    plt.close(medial_fig)

    lateral_fig, _ = plotting.plot_img_on_surf(stat_map=im, surf_mesh='fsaverage', inflate=True, threshold=thres, views=['lateral'], vmax=vmax_inp)
    lateral_fig.savefig('lateral.png', dpi=dpi, bbox_inches="tight")
    plt.close(lateral_fig)

    '''
    ventral_fig, _ = plotting.plot_img_on_surf(stat_map=im, surf_mesh='fsaverage', inflate=True, threshold=thres, views=['ventral'], vmax=vmax_inp)
    ventral_fig.savefig('ventral.png', dpi=dpi, bbox_inches="tight")
    plt.close(ventral_fig)
    '''

    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].imshow(mpimg.imread('medial.png'))
    ax[0].set_title('Medial View')
    ax[1].imshow(mpimg.imread('lateral.png'))
    ax[1].set_title('Lateral View')
    #ax[2].imshow(mpimg.imread('ventral.png'))
    #ax[2].set_title('Ventral View')

    for a in ax:
        a.axis('off')

    os.remove("medial.png")
    os.remove("lateral.png")
    #os.remove("ventral.png")
    return fig, ax

def plot_on_surf_z(im, thres=None, vmax_inp=None, mesh="fsaverage",
                  views=("lateral", "medial"), 
                  symmetric_cbar=True, cmap="cold_hot", colorbar=True, figsize=(14, 4),):
    """
    Plot a continuous stat map (e.g., z-map) on inflated surface with sulcal shading.
    Returns a single matplotlib figure with the requested views side-by-side.
    """

    fsavg = datasets.fetch_surf_fsaverage(mesh=mesh)
    surf = {"left": fsavg.infl_left, "right": fsavg.infl_right}
    bg   = {"left": fsavg.sulc_left, "right": fsavg.sulc_right}

    # Project once per hemisphere (continuous map: linear interpolation is fine)
    tex = {"left": surface.vol_to_surf(im, fsavg.pial_left, interpolation="linear"),
        "right": surface.vol_to_surf(im, fsavg.pial_right, interpolation="linear"),}

    fig, axes = plt.subplots(1, 2 * len(views),figsize=figsize,subplot_kw={"projection": "3d"},constrained_layout=True)

    i = 0
    for view in views:
        for hemi in ("left", "right"):
            plotting.plot_surf_stat_map(
                surf_mesh=surf[hemi], stat_map=tex[hemi], hemi=hemi, view=view,
                bg_map=bg[hemi],bg_on_data=True,
                cmap=cmap, symmetric_cbar=symmetric_cbar,
                threshold=thres, vmax=vmax_inp, colorbar=colorbar, axes=axes[i],)
            axes[i].set_title(f"{hemi[0].upper()} – {view[:3]}", fontsize=11)
            i += 1

    return fig, axes

# Compute RDM using Pearson or Euclidean distances
# Optionally plot the full matrix
def get_rdm(repr_list, method="pearsonr", plot=False, plot_title=None):
    repr_list = np.array(repr_list, dtype=np.float64)
    if repr_list.shape[1] == 1:
        return np.nan

    if method == "pearsonr":
        repr_list += 1e-10 * np.random.randn(*repr_list.shape)  # Avoid zero variance
        correlation_distances = np.round(1 - np.corrcoef(repr_list), 6)
        dist_matrix = correlation_distances
    elif method == "euclidean":
        n, p = repr_array.shape
        euclidean_distances = np.sqrt(((repr_array[:, np.newaxis, :] - repr_array[np.newaxis, :, :]) ** 2).sum(axis=2)) / np.sqrt(p)
        dist_matrix = euclidean_distances

    rdm_vector = [dist_matrix[i, j] for i in range(dist_matrix.shape[0]) for j in range(dist_matrix.shape[1]) if i > j]

    if plot:
        plt.figure()
        plt.imshow(np.tril(dist_matrix), cmap='viridis', origin='upper', interpolation='none')
        plt.colorbar(label='Dissimilarity')
        plt.title(f'{plot_title}')
        plt.xlabel('Stimulus')
        plt.ylabel('Stimulus')

    return rdm_vector

# Load a subject's reliability mask based on p and r thresholds
def load_reliability_mask(sub_id, space='MNI152NLin2009cAsym', p_thres=1, r_thres=0):
    ref_img = nib.load(f'../derivatives/fmriprep/sub-{sub_id}/func/sub-{sub_id}_task-main_run-1_space-{space}_desc-preproc_bold.nii.gz')
    affine, header = ref_img.affine, ref_img.header

    p_vals = nib.load(f'../derivatives/nilearn_analysis/reliability/sub-{sub_id}_task-main_space-{space}_desc-betas-fracridge_betasnormalize-True_stat-p_statmap.nii.gz').get_fdata()
    r_vals = nib.load(f'../derivatives/nilearn_analysis/reliability/sub-{sub_id}_task-main_space-{space}_desc-betas-fracridge_betasnormalize-True_stat-r_statmap.nii.gz').get_fdata()

    reliability_mask = np.zeros_like(r_vals, dtype='int')
    reliability_mask[(r_vals > r_thres) & (~np.isnan(r_vals)) & (p_vals < p_thres)] = 1
    return nib.Nifti1Image(reliability_mask, affine, header)


def get_grouped_wholebrainmap(images, nonparametric=True, correction="fdr"):
    # Step 1: Load individual RSA maps
    images = [load_img(img) for img in images]
    img_4d = concat_imgs(images)
    data = img_4d.get_fdata()  # shape: (x, y, z, n_subjects)

    if nonparametric:
        np.random.seed(42)

        # Step 2: Compute group mean (effect size)
        valid_counts = np.sum(~np.isnan(data), axis=3)
        effect_map = np.nanmean(data, axis=3)

        # Step 3: Signed permutation test
        n_permutations = 5000
        n_subjects = data.shape[3]
        permuted_means = np.zeros((n_permutations, *effect_map.shape))
        for i in range(n_permutations):
            signs = np.random.choice([-1, 1], size=n_subjects)
            flipped = data * signs[np.newaxis, np.newaxis, np.newaxis, :]
            permuted_means[i] = np.nanmean(flipped, axis=3)

        # Step 4: Mask voxels that are NaN in >20% of subjects
        mask = valid_counts >= int(0.8 * n_subjects)
        effect_map[~mask] = np.nan
        permuted_means[:, ~mask] = np.nan

        # Step 5: Compute uncorrected two-sided p-values
        p_map_uncorrected = (np.sum(np.abs(permuted_means) >= np.abs(effect_map), axis=0) + 1) / (n_permutations + 1)

        # Step 6: Convert to uncorrected z-map
        z_map_uncorrected = norm.isf(p_map_uncorrected / 2) * np.sign(effect_map)

    else:
        # Parametric test using GLM (fallback)
        design_matrix = pd.DataFrame([1] * len(images), columns=['intercept'])
        model = SecondLevelModel(smoothing_fwhm=None).fit(images, design_matrix=design_matrix)
        out_map = model.compute_contrast(output_type='all')

        effect_map = out_map['effect_size'].get_fdata()
        z_map_uncorrected = out_map['z_score'].get_fdata()
        p_map_uncorrected = out_map['p_value'].get_fdata()

    # Step 7: Multiple comparisons correction
    if correction == 'fdr':
        _, pvals_fdr = fdrcorrection(p_map_uncorrected.ravel(), alpha=0.05)
        p_map_corrected = pvals_fdr.reshape(effect_map.shape)

    elif correction == 'fwer':
        if not nonparametric:
            raise ValueError("FWER correction requires nonparametric=True")
        max_null = np.max(np.abs(permuted_means), axis=(1, 2, 3))
        p_map_corrected = np.array([
            (np.sum(max_null >= np.abs(v)) + 1) / (n_permutations + 1)
            for v in effect_map.ravel()
        ]).reshape(effect_map.shape)

    else:
        raise ValueError("correction must be either 'fdr' or 'fwer'")

    # Step 8: Compute corrected z-map
    z_map_corrected = norm.isf(p_map_corrected / 2) * np.sign(effect_map)

    # Step 9: Wrap into nibabel images
    sample_img = images[0]
    return {
        'effect_size': new_img_like(sample_img, effect_map),
        'z_score_uncorrected': new_img_like(sample_img, z_map_uncorrected),
        'p_value_uncorrected': new_img_like(sample_img, p_map_uncorrected),
        'z_score_corrected': new_img_like(sample_img, z_map_corrected),
        'p_value_corrected': new_img_like(sample_img, p_map_corrected)
    }



def spearmanr_permutation_test(x, y, n_permutations=2000):
    # Step 1: Compute the observed Spearman correlation
    observed_r, _ = spearmanr(x, y)
    permuted_rs = np.zeros(n_permutations)

    # Step 2: Build the null distribution by permuting y
    for i in range(n_permutations):
        shuffled_y = np.random.permutation(y)
        permuted_rs[i], _ = spearmanr(x, shuffled_y)

    # Step 3: Compute two-sided p-value
    p_value = (np.sum(np.abs(permuted_rs) >= np.abs(observed_r)) + 1) / (n_permutations + 1)

    return observed_r, p_value


# SIMPLE representation helper functions
def parse_predgoal_to_label(g, char_num):
    g1,g2,g3,g4 = g.split('_')
    
    lms = ['blue','green','red','yellow']
    items = ['blue','pink']
    agents = ['red','green']
    
    if char_num == 1:
        if int(g4) == -1:
            if g1 != "TE":
                return 'hinder green agent'
            else:
                return 'get away from green agent'
        if int(g4) == 1:
            if g1 == "LMO":
                return 'Get '+items[int(g2)]+' item to '+lms[int(g3)]+' lm'
            if g1 == "LMA":
                return 'Get red agent to '+lms[int(g3)]+' lm'
            if g1 == "TE":
                if int(g3) in [0,1]:
                    return 'get to green agent'
                else:
                    print("in here")
                    return 'Get red agent to '+items[int(g3)-2]+' item' # never occurs so not worrying about abstract mapping
    if char_num == 2:
        if int(g4) == -1:
            if g1 != "TE":
                return 'hinder red agent'
            else:
                return 'get away from red agent'
        if int(g4) == 1:
            if g1 == "LMO":
                return 'Get '+items[int(g2)]+' item to '+lms[int(g3)]+' lm'
            if g1 == "LMA":
                return 'Get red agent to '+lms[int(g3)]+' lm'
            if g1 == "TE":
                if int(g3) in [0,1]:
                    return 'get to red agent'
                else:
                    print("in here")
                    return 'Get green agent to '+items[int(g3)-2]+' item'    # never occurs so not worrying about abstract mapping

def parse_video_to_abstractgoal_prob(goal_probs_dict, ABSTRACT_GOAL_LABELS):
    """
    goal_probs_dict: dict with keys 0 and 1, each mapping to a list of (goal_str, prob)
    e.g. {0: [('LMO_0_0_1', 0.65), ...], 1: [...]}
    """
    vec1 = [0.0] * len(ABSTRACT_GOAL_LABELS[1])
    vec2 = [0.0] * len(ABSTRACT_GOAL_LABELS[2])
    
    # Agent 0
    for goal_str, prob in goal_probs_dict[0]:
        goal_str = parse_predgoal_to_label(goal_str,1).strip().lower()
        if "get red agent to" in goal_str or "get green agent to" in goal_str:
            label = 'go to landmark'
        elif "get blue item to" in goal_str or "get pink item to" in goal_str:
            label = 'take object to landmark'
        else:
            label = goal_str  # e.g., 'help green agent'
        
        if label in ABSTRACT_GOAL_LABELS[1]:
            vec1[ABSTRACT_GOAL_LABELS[1].index(label)] += prob
        else:
            print(f"agent0 label not recognized: {label}")
            
    # Agent 1
    for goal_str, prob in goal_probs_dict[1]:
        goal_str = parse_predgoal_to_label(goal_str,2).strip().lower()
        if "get red agent to" in goal_str or "get green agent to" in goal_str:
            label = 'go to landmark'
        elif "get blue item to" in goal_str or "get pink item to" in goal_str:
            label = 'take object to landmark'
        else:
            label = goal_str  # e.g., 'help red agent'
        
        if label in ABSTRACT_GOAL_LABELS[2]:
            vec2[ABSTRACT_GOAL_LABELS[2].index(label)] += prob
        else:
            print(f"agent1 label not recognized: {label}")
    
    return vec1 + vec2  # soft 12D vector


def parse_video_to_propoposal_dists(proposal_dists, ABSTRACT_GOAL_LABELS):
    max_time = max(proposal_dists.keys(), key=lambda k: k[0])[0]

    agent1 = {label: np.full(4, np.inf) for label in ABSTRACT_GOAL_LABELS[1]}
    agent2 = {label: np.full(4, np.inf) for label in ABSTRACT_GOAL_LABELS[2]}

    for proposal, v in proposal_dists.items():
        if proposal[0]==max_time:
            # Agent 1
            if proposal[2] == 'help':
                proposal_str = 'help green agent'
            elif proposal[2] == 'hinder':
                proposal_str = 'hinder green agent'
            else:
                proposal_str = parse_predgoal_to_label(proposal[2],1).strip().lower()

            if "get red agent to" in proposal_str or "get green agent to" in proposal_str:
                label = 'go to landmark'
            elif "get blue item to" in proposal_str or "get pink item to" in proposal_str:
                label = 'take object to landmark'
            else:
                label = proposal_str  # e.g., 'help green agent'

            if label in ABSTRACT_GOAL_LABELS[1]:
                agent1[label] = np.minimum(agent1[label], v['dists'])
            else:
                print(f"agent0 label not recognized: {label}")

            # Agent 2 
            if proposal[3] == 'help':
                proposal_str = 'help red agent'
            elif proposal[3] == 'hinder':
                proposal_str = 'hinder red agent'
            else:
                proposal_str = parse_predgoal_to_label(proposal[3],2).strip().lower()

            if "get red agent to" in proposal_str or "get green agent to" in proposal_str:
                label = 'go to landmark'
            elif "get blue item to" in proposal_str or "get pink item to" in proposal_str:
                label = 'take object to landmark'
            else:
                label = proposal_str  # e.g., 'help green agent'

            if label in ABSTRACT_GOAL_LABELS[2]:
                agent2[label] = np.minimum(agent2[label], v['dists'])
            else:
                print(f"agent1 label not recognized: {label}")

    agent1 = np.concatenate([agent1[label] for label in ABSTRACT_GOAL_LABELS[1]])
    agent2 = np.concatenate([agent2[label] for label in ABSTRACT_GOAL_LABELS[2]])
    agent_dists = np.concatenate([agent1, agent2])
    agent_dists[np.isinf(agent_dists)] = np.nan
    return agent_dists



