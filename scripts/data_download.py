from scripts.data_helpers_night_lights import *
import pandas as pd

# configure sample, save folder, region, which collections to download
samples = pd.read_csv('../data/city_samples_0.csv')
save_folder = 'CensusExport0905_0'
buffer_region = 2e5
download_DMSP = False
download_census = True

# check index is correct
assert((samples.index == range(200)).all())

crs, crsTransform = get_crs("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS")

for i in range(samples.shape[0]):

    print(f'City {i + 1}/200')

    if download_DMSP:
        download_gee_DMSP(lon = samples.lng[i], lat = samples.lat[i], folder_name=save_folder,
                      start_year=2000, end_year=2010, interval=5, buffer_region=buffer_region)


    if download_census:
        download_yearly_collection(lon=samples.lng[i], lat=samples.lat[i], folder_name=save_folder,
                                   start_year=2000, end_year=2010, interval=5, buffer_region=buffer_region,
                                   crs=crs, crsTransform=crsTransform)