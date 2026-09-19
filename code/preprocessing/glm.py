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
import matplotlib.pyplot as plt
import warnings
from nilearn.image import smooth_img

warnings.filterwarnings('ignore')


def task_2_stimdur(task):
    stimdur = dict()
    stimdur['main'] = 10 #since each video is 10s
    stimdur['tom'] = 14 #since stimdur=5*TR and quesdur = 2*TR
    stimdur['sipsts'] = 16 #block duration = 15
    stimdur['physics'] = 11.5 #trial duration = 11.5 or 26s figure out
    return stimdur[task]


def get_metadata(file):
    img = nib.load(file)
    return img.shape[-1], img.affine, img.header


def mk_design(conditions, file, tr, n_scans, out_path, run, task):
    #from nilearn.plotting import plot_design_matrix
    event = pd.read_csv(file, sep='\t')
    #event.onset = event.onset - tr/2
    event['frame_num'] = np.round_(event.onset, decimals=1) / tr

    if task == "main":
        X = pd.DataFrame(np.zeros((n_scans, len(conditions)), dtype='int'),
                         columns=conditions.video_name.to_list())
        
        for i, video in enumerate(conditions.video_name):
            frame = event.loc[event.identifier == video, 'frame_num']
            if not frame.empty:
                frame = frame.item()
                if not np.isnan(frame):
                    frame = int(frame)
                    X.loc[frame, video] = 1
        
    elif task == "sipsts" or task == "tom" or task == "physics":
        X = pd.DataFrame(np.zeros((n_scans, len(conditions)), dtype='int'),
                         columns=conditions)
        for i, video in enumerate(conditions):
            # one condition occurs multiple times in a run for these tasks, so code slightly diff
            # all frames associated with that condition are set as 1
            frames = event.loc[event.trial_type == video, 'frame_num'].astype(int).to_list()
            X.loc[frames,video] = 1


    X.to_csv(f'{out_path}/design/run_{run}.csv')

    #plot_design_matrix(X, output_file=f'{out_path}/design/run_{run}.png')
    return X.to_numpy()


def mask_img(img, mask):
    if type(img) is nib.nifti1.Nifti1Image:
        masked_img = np.array(img.dataobj)
        mask = np.array(mask.dataobj)
    else:
        masked_img = img.copy()
    mask = np.invert(mask.astype('bool'))
    i, j, k = np.where(mask)
    masked_img[i, j, k] = 0.
    if type(img) is nib.nifti1.Nifti1Image:
        masked_img = nib.Nifti1Image(masked_img, img.affine, img.header)
    return masked_img


def load_data(img_file, mask_file=None, fwhm=3):
    img = nib.load(img_file)
    if fwhm is not None:
        img = smooth_img(img, fwhm=fwhm)

    if mask_file is not None:
        mask = nib.load(mask_file)
        img_masked = mask_img(img, mask)
    return np.array(img_masked.dataobj)


class GLM:
    def __init__(self, args):
        self.process = 'class_name'
        self.sid = args.sid.zfill(2)
        self.task = args.task
        self.space = args.space
        self.stimdur = task_2_stimdur(self.task)
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
        elif self.task == "sipsts":
            self.conditions = ['interact', 'non_interact']
        elif self.task == "tom":
            self.conditions = ['photo', 'belief']
        elif self.task == "physics":
            self.conditions = ['social', 'physics']

        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        Path(f'{self.out_dir}/files').mkdir(parents=True, exist_ok=True)
        Path(f'{self.out_dir}/figures').mkdir(parents=True, exist_ok=True)
        Path(f'{self.out_dir}/design').mkdir(parents=True, exist_ok=True)
        print(vars(self))

    def mk_opts(self):
        # we will rely on the default hyperparameters for this run
        opt = dict()
        opt['wanthdf5'] = 1  # outputs as hdf5
        if self.hrf:
            opt['wantlibrary'] = 1  # hrf fitting
        else:
            opt['wantlibrary'] = 0

        if self.denoise:
            opt['wantglmdenoise'] = 1  # denoise
        else:
            opt['wantglmdenoise'] = 0

        if self.fracridge:
            opt['wantfracridge'] = 1  # fracridge
        else:
            opt['wantfracridge'] = 0

        opt['wantmemoryoutputs'] = [0, 0, 0, 0]  # don't save outputs to memory
        opt['n_jobs'] = 6
        opt['wantpercentbold'] = 1
        opt['wantautoscale'] = 1
        print(opt)
        return opt

    def load_data_and_design(self, masks, epis, events, n_scans):
        data = []
        design = []
        for i, ((mask, epi), event) in enumerate(zip(zip(masks, epis), events)):
            run = str(i+1).zfill(2)
            cur_design = mk_design(self.conditions, event, self.tr, n_scans, self.out_dir, run, self.task)
            design.append(cur_design)
            data.append(load_data(epi, mask, fwhm=self.smooth))
        return data, design

    def run(self):
        epi_files = natsorted(glob.glob(f'{self.func_dir}/*{self.task}*{self.space}*bold.*ii*'))
        if self.space != 'fsnative':
            mask_files = natsorted(glob.glob(f'{self.func_dir}/*{self.task}*{self.space}*mask.*ii*'))
        else:
            mask_files = [None for i in range(len(epi_files))]
        print(f'{self.func_dir}/*{self.task}*{self.space}*bold.nii.gz')
        print('\n')
        print(epi_files)
        event_files = natsorted(glob.glob(f'{self.raw_dir}/*{self.task}*events.tsv'))
        n_scans, affine, header = get_metadata(epi_files[0])
        data, design = self.load_data_and_design(mask_files, epi_files, event_files, n_scans)
        
        opt = self.mk_opts()
        print(opt)

        print(len(data), len(epi_files), len(mask_files), len(event_files))

        print(f'\nThere are {len(data)} runs in total')
        print(f'N = {data[0].shape[3]} TRs per run')
        print(f'Total scan length: {n_scans * self.tr} s')
        print(f'The dimensions of the data for each run are: {data[0].shape}')
        print(f'The stimulus duration is {self.stimdur} seconds')
        print(f'XYZ dimensionality is: {data[0].shape[:3]}')
        print(f'Numeric precision of data is: {type(data[0][0, 0, 0, 0])}\n')
        
        glmsingle_obj = GLM_single(opt)
        glmsingle_obj.fit(design,
                          data,
                          self.stimdur,
                          self.tr,
                          outputdir=f'{self.out_dir}/files',
                          figuredir=f'{self.out_dir}/figures')
        


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
    parser.add_argument('--bids_dir', type=str,
                        default='../../')
    parser.add_argument('--out_dir', type=str,
                        default='../../derivatives/GLMsingle')

    args = parser.parse_args()
    GLM(args).run()


if __name__ == '__main__':
    main()