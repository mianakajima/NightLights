To create conda environment: 

`conda env create -f environment.yml`

## Repository Organization

### Data
- The `data` folder contains an image samples downloaded from DMSP-OLS using Google Earth Engine. The folder is arranged by population class following image organization needed to use `ImageFolder()` pytorch loading function. 
  - A total of 200 cities were sampled from 3 census years (2000, 2005, 2010) resulting in 600 images. Five hundred images are included in the training set and 100 images each are in the dev and test sets. 
  - The images are already calibrated and formatted to jpg from the raw TIFF files downloaded. 
  
#### Image Processing
- Script to download satellite data to Google Drive and process images: [scripts/data_download.py](https://github.com/mianakajima/NightLights/blob/main/scripts/data_download.py)
  - Lines 5 - 10 may be modified to download more images
- The images have been calibrated. Notebook exploring calibration is [here](https://github.com/mianakajima/NightLights/blob/main/notebooks/06_DMSP_OLS_calibration.ipynb). 
- Cities were sampled based on population. Notebook to develop cities to sample is [here](https://github.com/mianakajima/NightLights/blob/main/notebooks/07_sample_cities.ipynb). 

### Modeling 

- Modeling was done in this notebook: [09_train_CNN_nightlights.ipynb](https://github.com/mianakajima/NightLights/blob/main/notebooks/09_train_CNN_nightlights.ipynb)
  - GPU Used: GeForce GTX 1080 
      - (To be completely honest, this project didn't necessitate a GPU but I wanted to try using one for kicks, and it seemed to speed up training by about 6 times). 

- At the end of the notebook, I also played around with using Pytorch lightining and Tensorboard which was useful for debugging modeling. Pytorch lightning model code referenced in notebook can be found here: [scripts/models.py](https://github.com/mianakajima/NightLights/blob/main/scripts/models.py). 

- Final trained CNN model is saved in `results` folder. 

### Notebooks 

- Notebooks used for analysis are saved in `notebooks` folder. Earlier exploratory analysis is saved in `notebooks/exploratory` folder. Numbering is preserved for self-reference. 
