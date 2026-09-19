 #!/usr/bin/env python
# coding: utf-8

import argparse
import glob
from pathlib import Path

from glmsingle.glmsingle import GLM_single
from natsort import natsorted
import numpy as np
import pandas as pd
import nibabel as nib
import warnings
from nilearn.image import smooth_img

warnings.filterwarnings('ignore')


def task_2_stimdur(task):
    stimdur = dict()
    stimdur['main'] = 10  # entire video is 10s
    stimdur['tom'] = 14
    stimdur['sipsts'] = 16
    stimdur['physics'] = 11.5
    return stimdur[task]


def get_metadata(file):
    img = nib.load(file)
    return img.shape[-1], img.affine, img.header


def mk_design(conditions, chunk_labels, file, tr, n_scans, task):
    event = pd.read_csv(file, sep='\t')
    event['frame_num'] = np.round_(event.onset, decimals=1) / tr

    if task == "main":
        X = pd.DataFrame(np.zeros((n_scans, len(chunk_labels)), dtype='int'), columns=chunk_labels)
        chunk_duration = 2  # fixed for chunking

        for video in conditions.video_name:
            frame = event.loc[event.identifier == video, 'frame_num']
            if not frame.empty:
                frame = frame.item()
                if not np.isnan(frame):
                    for c in range(5):  # hardcoded 5 chunks per 10s video
                        start_frame = int(frame + (chunk_duration / tr) * c)
                        column_name = f'{video}_chunk_{c+1}'
                        if start_frame < n_scans:
                            X.loc[start_frame, column_name] = 1

    else:
        X = pd.DataFrame(np.zeros((n_scans, len(conditions)), dtype='int'), columns=conditions)
        for video in conditions:
            frames = event.loc[event.trial_type == video, 'frame_num'].astype(int).to_list()
            X.loc[frames, video] = 1

    return X.to_numpy(), X


def mask_img(img, mask):
    if isinstance(img, nib.Nifti1Image):
        masked_img = np.array(img.dataobj)
        mask = np.array(mask.dataobj)
    else:
        masked_img = img.copy()
    mask = np.invert(mask.astype('bool'))
    i, j, k = np.where(mask)
    masked_img[i, j, k] = 0.
    if isinstance(img, nib.Nifti1Image):
        masked_img = nib.Nifti1Image(masked_img, img.affine, img.header)
    return masked_img


def load_data(img_file, mask_file=None, fwhm=3):
    img = nib.load(img_file)
    if fwhm is not None:
        img = smooth_img(img, fwhm=fwhm)

    if mask_file is not None:
        mask = nib.load(mask_file)
        img_masked = mask_img(img, mask)
    else:
        img_masked = img
    return np.array(img_masked.dataobj)


class GLM:
    def __init__(self, args):
        self.sid = args.sid.zfill(2)
        self.task = args.task
        self.space = args.space
        self.tr = args.tr
        self.fracridge = args.fracridge
        self.denoise = args.denoise
        self.hrf = args.hrf
        self.smooth = args.smooth

        self.func_dir = f'{args.bids_dir}/derivatives/fmriprep/sub-{self.sid}/func'
        self.raw_dir = f'{args.bids_dir}/sub-{self.sid}/func'
        self.out_dir = f'{args.out_dir}/sub-{self.sid}/task-{self.task}_space-{self.space}'

        if self.task == "main":
            self.conditions = pd.read_csv(f'{args.out_dir}/conditions.csv')
            # Define chunk labels only ONCE here
            self.chunk_labels = [f'{video}_chunk_{c+1}' for video in self.conditions.video_name for c in range(5)]
            self.stimdur_for_glmsingle = 2  # chunk duration for GLMsingle
        elif self.task == "sipsts":
            self.conditions = ['interact', 'non_interact']
            self.stimdur_for_glmsingle = task_2_stimdur(self.task)
        elif self.task == "tom":
            self.conditions = ['photo', 'belief']
            self.stimdur_for_glmsingle = task_2_stimdur(self.task)
        elif self.task == "physics":
            self.conditions = ['social', 'physics']
            self.stimdur_for_glmsingle = task_2_stimdur(self.task)

        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        Path(f'{self.out_dir}/files').mkdir(parents=True, exist_ok=True)
        Path(f'{self.out_dir}/figures').mkdir(parents=True, exist_ok=True)
        Path(f'{self.out_dir}/design').mkdir(parents=True, exist_ok=True)
        print(vars(self))

    def mk_opts(self):
        opt = {
            'wanthdf5': 1,
            'wantlibrary': int(self.hrf),
            'wantglmdenoise': int(self.denoise),
            'wantfracridge': int(self.fracridge),
            'wantmemoryoutputs': [0, 0, 0, 0],
            'n_jobs': 6,
            'wantpercentbold': 1,
            'wantautoscale': 1
        }
        print(opt)
        return opt

    def load_data_and_design(self, masks, epis, events, n_scans):
        data = []
        design = []
        for i, ((mask, epi), event) in enumerate(zip(zip(masks, epis), events)):
            run = str(i + 1).zfill(2)
            cur_design_np, cur_design_df = mk_design(
                self.conditions,
                getattr(self, 'chunk_labels', self.conditions),
                event,
                self.tr,
                n_scans,
                self.task
            )
            design.append(cur_design_np)
            data.append(load_data(epi, mask, fwhm=self.smooth))
            # Save design matrix for verification
            cur_design_df.to_csv(f'{self.out_dir}/design/run_{run}.csv')
        return data, design

    def run(self):
        epi_files = natsorted(glob.glob(f'{self.func_dir}/*{self.task}*{self.space}*bold.*ii*'))
        if self.space != 'fsnative':
            mask_files = natsorted(glob.glob(f'{self.func_dir}/*{self.task}*{self.space}*mask.*ii*'))
        else:
            mask_files = [None for _ in range(len(epi_files))]
        event_files = natsorted(glob.glob(f'{self.raw_dir}/*{self.task}*events.tsv'))
        n_scans, affine, header = get_metadata(epi_files[0])

        data, design = self.load_data_and_design(mask_files, epi_files, event_files, n_scans)

        opt = self.mk_opts()

        glmsingle_obj = GLM_single(opt)
        glmsingle_obj.fit(
            design,
            data,
            self.stimdur_for_glmsingle,
            self.tr,
            outputdir=f'{self.out_dir}/files',
            figuredir=f'{self.out_dir}/figures'
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', '-s', type=str, default='1')
    parser.add_argument('--task', '-task', type=str, default='main')
    parser.add_argument('--space', type=str, default='T1w')
    parser.add_argument('--tr', type=float, default=2)
    parser.add_argument('--smooth', type=int, default=None)
    parser.add_argument('--hrf', action='store_true')
    parser.add_argument('--no-hrf', dest='hrf', action='store_false')
    parser.set_defaults(hrf=True)
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--no-denoise', dest='denoise', action='store_false')
    parser.set_defaults(denoise=True)
    parser.add_argument('--fracridge', action='store_true')
    parser.add_argument('--no-fracridge', dest='fracridge', action='store_false')
    parser.set_defaults(fracridge=True)
    parser.add_argument('--bids_dir', type=str, default='/home/mmalik16/data-lisik3/manasimalik/SI_Comp_fMRI/fMRI_data')
    parser.add_argument('--out_dir', type=str, default='/home/mmalik16/data-lisik3/manasimalik/SI_Comp_fMRI/fMRI_data/derivatives/GLMsingle_2schunks')

    args = parser.parse_args()
    GLM(args).run()


if __name__ == '__main__':
    main()
