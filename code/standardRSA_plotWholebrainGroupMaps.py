# Group-level Whole-Brain RSA Map Analysis and Plotting
# ------------------------------------------------------
# This script loads subject-level searchlight RSA results, computes group-level
# statistics (including semi-partial unique variance maps), and visualizes them
# as surface plots using Nilearn.

from general_utils import plot_on_surf, get_grouped_wholebrainmap, plot_on_surf_z
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# ==== CONFIGURATION ====
parser = argparse.ArgumentParser()

parser.add_argument("--thres", type=float, default=1.96)
parser.add_argument("--out_type", default="z_score_corrected")
parser.add_argument("--plot_styles", nargs="+", default=["inflated", "projectedsurf"])

parser.add_argument("--features2test", nargs="*", default=[])
parser.add_argument("--diff_pairs", nargs="*", default=[],
                    help="Format: feature1,feature2")
parser.add_argument("--sr_comparisons", nargs="*",
                    default=["SocialGNN10s_trained10s,SIMPLE10s"],
                    help="Format: feature1,feature2")

args = parser.parse_args()

subj_group = "M"
within_reliable = True
radius = 3
plot_dir = f'../derivatives/plots/rsa/group/'

thres = args.thres
out_type = args.out_type
plot_styles = args.plot_styles
features2test = args.features2test
diff_pairs = [tuple(x.split(",")) for x in args.diff_pairs]
sr_comparisons = [tuple(x.split(",")) for x in args.sr_comparisons]

# Subject list
sub_ids = ["M03", "M04", "M05", "M06", "M08", "M09", "M10", "M11", "M12", "M13", 
        "M15", "M17", "M18", "M19", "M20", "M21", "M22", "M23", "M24", "M25",
        "M26", "M27", "M28", "M29", "M30"]

reliable_suffix = f'_withinreliablevoxels' if within_reliable else ''


# ==== STEP 1: Standard RSA Group Maps ====
group_maps = {}
image_vectors = {}


for f_name in tqdm(features2test):
    images = []
    for sub_id in sub_ids:
        rsa_img = f'../derivatives/nilearn_analysis/rsa/sub-{sub_id}/sub-{sub_id}_searchlightRSA-{f_name}_radius-{radius}_stat-rmap{reliable_suffix}.nii.gz'
        images.append(rsa_img)

    group_maps[f_name] = get_grouped_wholebrainmap(images, nonparametric=True, correction='fdr')
    image_vectors[f_name] = images

print("Done calculating group maps for each feature!")

# ==== STEP 2: Difference Maps (feature1 - feature2) ====
for pair in tqdm(diff_pairs):
    diff_images = []
    for sub_id in sub_ids:
        output_dir = f'../derivatives/nilearn_analysis/rsa/sub-{sub_id}/'
        f1_img = nib.load(output_dir + f'sub-{sub_id}_searchlightRSA-{pair[0]}_radius-{radius}_stat-rmap{reliable_suffix}.nii.gz')
        f2_img = nib.load(output_dir + f'sub-{sub_id}_searchlightRSA-{pair[1]}_radius-{radius}_stat-rmap{reliable_suffix}.nii.gz')

        diff = f1_img.get_fdata() - f2_img.get_fdata()
        diff_im = nib.Nifti1Image(diff, f1_img.affine, f1_img.header)
        diff_images.append(diff_im)

    group_maps[pair] = get_grouped_wholebrainmap(diff_images, nonparametric=True, correction='fdr')
    image_vectors[pair] = diff_images

print("Done calculating difference group maps for SocialGNN/SIMPLE!")

# ==== STEP 3: Unique Variance (Semi-partial RSA) Group Maps ====
sr_group_maps = {}

for feature1, feature2 in tqdm(sr_comparisons):
    images12 = []  # variance uniquely explained by feature1 (controlling for feature2)
    images21 = []  # variance uniquely explained by feature2 (controlling for feature1)

    for sub_id in sub_ids:
        output_dir = f'../derivatives/nilearn_analysis/rsa/sub-{sub_id}/'
        sr1_img = output_dir + f'sub-{sub_id}_searchlightRSA-{feature1}{feature2}_radius-{radius}_stat-srmap{reliable_suffix}.nii.gz'
        sr2_img = output_dir + f'sub-{sub_id}_searchlightRSA-{feature2}{feature1}_radius-{radius}_stat-srmap{reliable_suffix}.nii.gz'

        images12.append(sr1_img)
        images21.append(sr2_img)

    sr_group_maps[(feature1, feature2)] = get_grouped_wholebrainmap(images12, nonparametric=True, correction='fdr')
    sr_group_maps[(feature2, feature1)] = get_grouped_wholebrainmap(images21, nonparametric=True, correction='fdr')

print("Done calculating uniqvar group maps for SocialGNN/SIMPLE!")

# ==== STEP 4: Plotting All Maps ====
print('Plotting now!')

for plot_style in plot_styles:
    print(f"Making {plot_style} plots")
    plot_style_suffix = "_projectedsurf" if plot_style == "projectedsurf" else ""
    vmax = 5 if plot_style == "projectedsurf" else 5.5

    # Thresholded Z maps
    for f_name in tqdm(features2test + diff_pairs):
        if plot_style == "projectedsurf":
            fig, _ = plot_on_surf_z(group_maps[f_name][out_type],thres=thres,vmax_inp=vmax)
        else:
            fig, _ = plot_on_surf(group_maps[f_name][out_type], thres=thres,vmax_inp=vmax)
        fig.suptitle(f"{f_name} (Group {out_type} map; thres={thres})")
        fig.savefig(plot_dir + f'standardrsa_{subj_group}groupmap_{f_name}{reliable_suffix}_{out_type}_{thres}{plot_style_suffix}.png', bbox_inches='tight', dpi=600)
        plt.close(fig)


    # Unique variance Z maps (uncorrected and corrected)
    for f1, f2 in tqdm(sr_comparisons):
        for fwd, rev in [(f1, f2), (f2, f1)]:
            if plot_style == "projectedsurf":
                fig, _ = plot_on_surf_z(sr_group_maps[(fwd, rev)][out_type], thres=thres,vmax_inp=vmax)
            else:
                fig, _ = plot_on_surf(sr_group_maps[(fwd, rev)][out_type],thres=thres,vmax_inp=vmax)
            fig.suptitle(f"SR {fwd} controlling for {rev} (Group {out_type} map; thres=1.96)")
            fig.savefig(plot_dir + f'standardrsa_uniqvar_{subj_group}groupmap_{fwd}controlling{rev}{reliable_suffix}_{out_type}_{thres}{plot_style_suffix}.png', bbox_inches='tight', dpi=600)
            plt.close(fig)

