# Helper functions for downloading data from Google Earth Engine
import ee
import time
import xarray as xr
import rioxarray
import glob
import numpy as np

def download_gee(lon, lat, band_name, folder_name,
                 collection_id = "NOAA/DMSP-OLS/NIGHTTIME_LIGHTS",
                 start_year = 1992, end_year = 2012, buffer_region = 1.5e6):
    """ Downloads one image per year from NOAA/DMSP-OLS
    lat: float
        Latitude
    lon: float
        Longitude
    band_name: str
        Name of band you would like to export
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

    # Must have access to GEE to use; authenticate and initialize
    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()

    collection = ee.ImageCollection(collection_id).select(band_name)

    years = range(start_year, end_year + 1)

    for year in years:

        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'

        collection_year = collection.filterDate(start_date, end_date)

        geo_point = ee.Geometry.Point(lon, lat)
        region = geo_point.buffer(buffer_region).bounds()

        # start export task

        task = ee.batch.Export.image.toDrive(image = collection_year.first(),
                                             folder = folder_name,
                                             fileNamePrefix= f'lat_{lat}_lon_{lon}_{year}',
                                             region = region)

        task.start()
        print(f'Starting task for year:{year}')

        status = task.status()['state']
        while status != 'COMPLETED':
            print(f'Status: {status}')

            if status == 'FAILED':
                print(task.status())
                break

            time.sleep(10)
            status = task.status()['state']

    print('Finished Exporting')


def convert_tiff_to_xarray(folder_path, band_name):
    """Convert folder of GeoTIFF files to xarray dataset in order to visualize in plotly"""

    geotiff_list = glob.glob(f'{folder_path}/*.tif')
    years = [path[-8:-4] for path in geotiff_list] # extract years from TIFF paths
    # TODO: order chronologically
    # TODO: I need to calibrate images probably... https://worldbank.github.io/OpenNightLights/tutorials/mod5_1_DMSP-OLS_intercalibration.html#elvidge2009fifteen

    time_var = xr.Variable('time', np.array(years))
    geotiff_da = xr.concat([rioxarray.open_rasterio(i) for i in geotiff_list], dim = time_var)
    geotiff_ds = geotiff_da.to_dataset('band')
    geotiff_ds = geotiff_ds.rename({1: band_name})

    return geotiff_ds


# test download/plotting
if __name__ == "__main__":

    download = True
    plot = False

    if download:
        # coordinates are for Madrid, Spain
        download_gee(-3.7038, 40.4168, 'avg_vis', 'NightLightsExports', buffer_region=2.5e5)

    if plot:
        import plotly.express as px
        geotiff_ds = convert_tiff_to_xarray('../data/NightLightsExports', band_name = 'stable_lights')

        fig = px.imshow(geotiff_ds.stable_lights, animation_frame = 'time', zmin= 0, zmax = 63, color_continuous_scale = 'gray', binary_string = True
                                                                                                                )
        fig.update_layout(
            yaxis = dict(autorange="reversed")
        )
        fig.show(renderer="browser")
