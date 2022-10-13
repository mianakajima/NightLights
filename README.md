To create conda environment: 

`conda env create -f environment.yml`

## Repository Organization

### Data
- The `data` folder contains image samples downloaded from DMSP-OLS using Google Earth Engine. The folder is arranged by population class following image organization needed to use `ImageFolder()` pytorch loading function. 
  - A total of 200 cities were sampled from 3 census years (2000, 2005, 2010) resulting in 600 images. Five hundred images are included in the training set and 100 images each are in the dev and test sets. 
  - The images are already calibrated and formatted to jpg from the raw TIFF files downloaded. 

#### Image Classes

Classes were developed based on populations of sampled images. The populations were split into quintiles. Since downloading many images from Google Earth Engine can take considerable time, having 5 classes seemed reasonable from training data size (for training a CNN). In the future, it would be interesting to download more data and create more granular classes.

- Class 1: Less than 5.7 million 
- Class 2: Between 5.7 million and 9.9 million
- Class 3: Between 9.9 million and 15.4 million
- Class 4: Between 15.4 million and 28 million
- Class 5: Between 28 million and 159 million
  
#### Image Processing
- Script to download satellite data to Google Drive and process images: [scripts/data_download.py](https://github.com/mianakajima/NightLights/blob/main/scripts/data_download.py)
  - Lines 5 - 10 may be modified to download more images
  - If using, a [Google Earth Engine Developer account](https://developers.google.com/earth-engine) needs to be created first
- Notebook exploring calibration is [here](https://github.com/mianakajima/NightLights/blob/main/notebooks/06_DMSP_OLS_calibration.ipynb). 
- Notebook to develop cities to sample is [here](https://github.com/mianakajima/NightLights/blob/main/notebooks/07_sample_cities.ipynb). 

### Modeling 

- Modeling was done in this notebook: [09_train_CNN_nightlights.ipynb](https://github.com/mianakajima/NightLights/blob/main/notebooks/09_train_CNN_nightlights.ipynb)
  - GPU Used: GeForce GTX 1080 
      - (To be completely honest, this project didn't necessitate a GPU but I wanted to try using one for kicks, and it seemed to speed up training by about 6 times). 

- At the end of the notebook, I also played around with using Pytorch lightining and Tensorboard which was useful for debugging modeling. Pytorch lightning model code referenced in notebook can be found here: [scripts/models.py](https://github.com/mianakajima/NightLights/blob/main/scripts/models.py). 

- Final trained CNN model is saved in `results` folder. 

### Notebooks 

- Notebooks used for analysis are saved in `notebooks` folder. Earlier exploratory analysis is saved in `notebooks/exploratory` folder. Numbering is preserved for self-reference. 
