# GLMsingle preprocessing

Functional data in MNI152NLin2009cAsym space were spatially smoothed
with a 4 mm FWHM kernel using `nilearn.image.smooth_img`.

GLMsingle was run with:

- voxel-wise HRF fitting
- GLMdenoise
- fractional ridge regression
- `wantpercentbold = 1`
- `wantautoscale = 1`

## Standard single-trial estimates

Each 10-second video trial was modeled with one response estimate.

```
python glm.py --sid=$1 --task=main --space=MNI152NLin2009cAsym --smooth=4
```

## 2-second chunk estimates

For the time-resolved analyses, each 10-second video was divided into five
2-second chunks, with a separate response estimate for each chunk.

```
python glm_2schunks.py --sid=$1 --task=main --space=MNI152NLin2009cAsym --smooth=4
```

