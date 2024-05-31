import numpy as np
import xarray as xr


def save_valid_grid_points_fc_and_hist(
    dataset_fc: xr.Dataset, dataset_hist: xr.Dataset, output_path: str
):
    """
    Saves the (lat, lon) tuples of valid grid points present in both
    forecast and historical datasets to a NumPy file.
    """
    dataset_fc.load()
    dataset_hist.load()
    lats = dataset_hist["lat"]
    lons = dataset_hist["lon"]
    variables = ["hurs", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]

    valid_grid_points = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            lat = lats[i].values
            lon = lons[j].values
            data_point_hist = dataset_hist[variables].sel(
                lat=lat, lon=lon, method="nearest"
            )
            data_point_fc = dataset_fc[variables].sel(
                lat=lat, lon=lon, method="nearest"
            )
            if not (
                data_point_hist["pr"].isnull().all()
                or data_point_fc["pr"].isnull().all()
            ):
                valid_grid_points.append((lat, lon))

    # Create a structured array for efficient storage
    valid_grid_points = np.array(
        valid_grid_points,
        dtype=np.dtype([("lat", np.float32), ("lon", np.float32)])
    )

    # Save the array to disk
    np.save(output_path, valid_grid_points)


dataset_fc = xr.open_dataset("netcdf_files/forecasts/20240501/r1i1p1/merged_20240501_r1i1p1.nc")
dataset_hist = xr.open_dataset("netcdf_files/amber/2024/zalf_merged_amber_2024_v1-0.nc")
save_valid_grid_points_fc_and_hist(
    dataset_fc, dataset_hist, "valid_grid_points_fc_and_hist.npy"
)