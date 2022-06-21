# SimSiam-GOES17

Implementation of SimSiam for GOES-17 Dataset.

## 1. Brief introduction

Dataset is first taken from `s3` bucket and then download to local machine, where it is preprocessed (resize, colormap mode `hsv`).


## 2. Dataset dimensionality:

- 4:00:00 UTC - 10:00:00 UTC from day 70 - 100 (31 days ~ a month)
- CONUS (RadC)
- 2088 x 3000 x 5000

After preprocess, the dimension reduces to 2088 x 300 x 500.


## 3. Model:

Implementation from SimSiam on https://arxiv.org/abs/2011.10566 (with smaller `input_dim` and `resnet18()`). Right now, we are resizing it to square (3 x 224 x 224) inside the `Dataset`. However, for future improvement, we will try with 
- 1. Bigger images
- 2. Original shape (rectangle)
- 3. Sliding/shifting windows with Transformer
- 4. Web deploy
- 5. Real time prediction


## Update

### Jun 20, 2022:

- [x] Refactor
- [x] Singularity script
- [ ] Larger file
- [ ] Distributed running