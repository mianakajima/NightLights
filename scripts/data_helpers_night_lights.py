# Helper functions for downloading night lights data from Google Earth Engine
import ee
import time
import xarray as xr
import rioxarray
import glob
import numpy as np
import pandas as pd

calibration = pd.read_csv('../data/Elvidge_DMSP_intercalib_coef.csv')

def get_calib_coefficients(id, year):
    """
    Return calibration coefficients for DMSP-OLS
    `calibration` is a pandas dataframe read in above.
    """
    satellite = id[-7:-4]
    coefficient_df = calibration.loc[(calibration['satellite'] == satellite) & (calibration['year'] == year)].reset_index()

    return coefficient_df['c0'][0], coefficient_df['c1'][0], coefficient_df['c2'][0]

def calibrate_DMSP_OLS(image, year):
    """
    Return calibrated DMSP_OLS image
    image: Google Earth Engine image
    year: ie., 2008
    """

    id = image.getInfo()['id']
    c0, c1, c2 = get_calib_coefficients(id, year)

    # apply transformation
    calibrated = image.expression(f'{c0} + ({c1} * X) + ({c2} * X * X)',
                                            {'X': image.select('stable_lights')})
    # clip image
    calibrated = calibrated.where(calibrated.gt(63), 63).where(calibrated.lte(6), 0)
    return calibrated

def get_gee_collection(collection_id):
    """ Returns GEE image collection"""

    # Must have access to GEE to use; authenticate and initialize
    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()

    collection_im = ee.ImageCollection(collection_id)

    return collection_im



def download_gee_DMSP(lon, lat, folder_name,
                 collection_id = "NOAA/DMSP-OLS/NIGHTTIME_LIGHTS",
                 start_year = 1992, end_year = 2012, buffer_region = 1.5e6):
    """ Downloads one image per year from NOAA/DMSP-OLS
    lat: float
        Latitude
    lon: float
        Longitude
    folder_name: str
        Name of folder in Google Drive you would like to export to
    collection_id: optional, str
        Name of collection
    start_year: optional, int
        Start year you would like to download data for
    end_year: optional, int
        End year you would like to download data for (inclusive)
    buffer_region: optional, int
        Region around latitude/longitude coordinate to download image for

    """

    collection = get_gee_collection(collection_id).select('stable_lights')

    years = range(start_year, end_year + 1)

    for year in years:
        print('Year: ', year)
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'

        collection_filtered = collection.filterDate(start_date, end_date)
        collection_year = collection_filtered.first()

        geo_point = ee.Geometry.Point(lon, lat)
        region = geo_point.buffer(buffer_region).bounds()

        try:
            calibrated = calibrate_DMSP_OLS(collection_year, year)
        except:
            # Try second image
            collection_year = collection_filtered.sort('system:index', False).first()
            calibrated = calibrate_DMSP_OLS(collection_year, year)
            print("Calibrated second image.")
        finally:
            export_gee_image(calibrated, folder_name, f'lat_{lat}_lon_{lon}_{year}', region)

    print('Finished Exporting')


def export_gee_image(image, folder_name, image_name, region):
    """Start GEE export task and print status until completion"""

    # start export task
    task = ee.batch.Export.image.toDrive(image=image,
                                         folder=folder_name,
                                         fileNamePrefix=image_name,
                                         region=region)
    task.start()
    print(f'Starting task for: {image_name}')
    status = task.status()['state']
    while status != 'COMPLETED':
        print(f'Status: {status}')

        if status == 'FAILED':
            print(task.status())
            break

        time.sleep(10)
        status = task.status()['state']


def convert_tiff_to_xarray(folder_path, band_name):
    """Convert folder of GeoTIFF files to xarray dataset in order to visualize in plotly"""

    geotiff_list, years = get_tiff_list(folder_path)

    time_var = xr.Variable('time', np.array(years))
    geotiff_da = xr.concat([rioxarray.open_rasterio(i) for i in geotiff_list], dim = time_var)
    geotiff_ds = geotiff_da.to_dataset('band')
    geotiff_ds = geotiff_ds.rename({1: band_name})

    return geotiff_ds


def get_tiff_list(folder_path):
    """Gets tiff files and year of satellite image"""
    geotiff_list = sorted(glob.glob(f'{folder_path}/*.tif'))
    years = [path[-8:-4] for path in geotiff_list]  # extract years from TIFF paths

    years_int = [int(year) for year in years]
    years_int_sorted = years_int.copy()
    years_int_sorted.sort()
    assert years_int == years_int_sorted

    return geotiff_list, years


# test download/plotting
if __name__ == "__main__":

    download = False
    plot = True

    if download:
        # coordinates are for Madrid, Spain
        download_gee_DMSP(-3.7038, 40.4168, 'NightLightsExports0903', buffer_region=2.5e5, start_year=2007)

    if plot:
        import plotly.express as px
        geotiff_ds = convert_tiff_to_xarray('../data/NightLightsExports0903', band_name = 'stable_lights')

        fig = px.imshow(geotiff_ds.stable_lights, animation_frame = 'time', zmin= 0, zmax = 63, color_continuous_scale = 'gray', binary_string = True
                                                                                                                )
        fig.update_layout(
            yaxis = dict(autorange="reversed")
        )
        fig.show(renderer="browser")
