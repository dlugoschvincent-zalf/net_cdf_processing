import os
import time
import glob
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import xarray as xr
import numpy as np
import lz4.frame

class ClimateDataExtractor:
    def __init__(self):
        self.amber_ds = None
        self.forecast_ds = None

    def _convert_forecast_to_amber_units(self, forecast_df: pd.DataFrame):
        """Converts forecast data units to match amber data units."""
        forecast_df[["tas", "tasmax", "tasmin"]] -= 273.15  # Kelvin to Celsius
        forecast_df[["rsds"]] *= 24  # Daily mean to total daily radiation
        return forecast_df

    def _extract_point_data(
        self, dataset: xr.Dataset, lat: float, lon: float, variables: list[str]
    ):
        """Extracts time series data for specified variables at a lat/lon."""
        df = dataset[variables].sel(lat=lat, lon=lon, method="nearest").to_dataframe()
        df["iso-date"] = pd.to_datetime(df.index).date
        return df.set_index("iso-date")[variables]

    def _combine_dataframes(self, amber_df: pd.DataFrame, forecast_df: pd.DataFrame):
        """Combines amber and forecast dataframes, removing overlaps."""
        last_valid_amber = amber_df.last_valid_index()
        amber_df = amber_df.loc[:last_valid_amber]
        combined_df = pd.concat([amber_df, forecast_df], axis=0)
        return combined_df[~combined_df.index.duplicated(keep="first")]

    def _load_amber_data(
        self,
        folder_pattern: str,
        file_name_pattern: str,
        start_year: int,
        end_year: int,
    ):
        """Loads NetCDF files within the year range into an xarray dataset."""
        file_paths = []
        for year in range(start_year, end_year + 1):
            folder = folder_pattern.format(year=year)
            file_path = os.path.join(folder, file_name_pattern.format(year=year))
            if os.path.exists(file_path):
                file_paths.append(file_path)

        if not file_paths:
            print(f"No matching files found for years {start_year} to {end_year}.")
            return None
        return xr.open_mfdataset(file_paths)

    def _load_forecast_data(
        self, folder_pattern: str, file_name_pattern: str, ensemble: str
    ):
        """Loads NetCDF files within the year range into an xarray dataset."""
        folder = folder_pattern.format(ensemble=ensemble)
        file_path = os.path.join(folder, file_name_pattern.format(ensemble=ensemble))

        if not os.path.exists(file_path):
            print(f"No matching files found for ensemble {ensemble}.")
            return None
        return xr.open_dataset(file_path)

    def extract_amber_data(
        self,
        netcdf_folder_pattern: str,
        variables: list[str],
        file_name_pattern: str,
        start_year: int,
        end_year: int,
        ndarray_of_valid_lat_lon_tuples,
        max_points: int = None,
    ):

        self.amber_ds = self._load_amber_data(
            netcdf_folder_pattern, file_name_pattern, start_year, end_year
        ).load()

        if not self.amber_ds:
            return {}

        amber_df_dict = {}
        start_time = time.time()

        for count, (lat, lon) in enumerate(ndarray_of_valid_lat_lon_tuples):

            if max_points and count >= max_points:
                break  # Stop processing if max_points limit is reached

            amber_df = self._extract_point_data(self.amber_ds, lat, lon, variables)

            amber_df_dict[f"{lat},{lon}"] = amber_df

        self.amber_ds.close()
        print(f"Processed {count } points in {time.time() - start_time:.2f} seconds.")
        return amber_df_dict

    def extract_forecast_data(
        self,
        netcdf_folder_pattern: str,
        variables: list[str],
        file_name_pattern: str,
        ndarray_of_valid_lat_lon_tuples,
        ensemble: str,
        max_points: int = None,
    ):

        self.forecast_ds = self._load_forecast_data(
            netcdf_folder_pattern, file_name_pattern, ensemble=ensemble
        ).load()

        if not self.forecast_ds:
            return {}

        forecast_df_dict = {}
        start_time = time.time()

        for count, (lat, lon) in enumerate(ndarray_of_valid_lat_lon_tuples):

            if max_points and count >= max_points:
                break  # Stop processing if max_points limit is reached

            forecast_df = self._extract_point_data(
                self.forecast_ds, lat, lon, variables
            )
            forecast_df = self._convert_forecast_to_amber_units(forecast_df.copy())

            forecast_df_dict[f"{lat},{lon}"] = forecast_df

        self.forecast_ds.close()
        print(f"Processed {count } points in {time.time() - start_time:.2f} seconds.")
        return forecast_df_dict
    
    def save_data(self, data_dict: dict, directory_name:str, file_name: str):
        """Saves the data dictionary to a compressed pickle file."""
        os.makedirs(directory_name, exist_ok=True)
        file_path = os.path.join(directory_name, file_name)
        with lz4.frame.open(file_path, "wb") as f:
            t0 = time.time()
            pickle.dump(data_dict, f)
            t1 = time.time()
            print(f"Saved file {file_path} in {t1 - t0:.2f} seconds")

    def load_data(self, file_path: str):
        """Loads the data dictionary from a compressed pickle file."""
        with lz4.frame.open(file_path, "rb") as f:
            t0 = time.time()
            data_dict = pickle.load(f)
            t1 = time.time()
            print(f"Loaded file {file_path} in {t1 - t0:.2f} seconds")
        return data_dict



if __name__ == "__main__":
    extractor = ClimateDataExtractor()
    max_points = 2000
    ndarray_of_valid_lat_lon_tuples = np.load("valid_grid_points.npy")

    amber_data_file = f"cache/amber/2023_2024_{max_points}_points_extracted.pkl.lz4"
    forecast_data_file = f"cache/forecast/r1i1p1_{max_points}_points_extracted.pkl.lz4"


    amber_df_dict = extractor.extract_amber_data(
        netcdf_folder_pattern="netcdf_files/amber/{year}",
        variables=["hurs", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin"],
        file_name_pattern="zalf_merged_amber_{year}_v1-0.nc",
        start_year=2023,
        end_year=2024,
        ndarray_of_valid_lat_lon_tuples=ndarray_of_valid_lat_lon_tuples,
        max_points=max_points,
    )
    extractor.save_data(amber_df_dict, amber_data_file)

    forecast_df_dict = extractor.extract_forecast_data(
        netcdf_folder_pattern="netcdf_files/forecasts/20240501/{ensemble}",
        variables=["hurs", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin"],
        file_name_pattern="merged_20240501_{ensemble}.nc",
        ndarray_of_valid_lat_lon_tuples=ndarray_of_valid_lat_lon_tuples,
        max_points=max_points,
        ensemble="r1i1p1",
    )

    extractor.save_data(forecast_df_dict,forecast_data_file)

    print(len(amber_df_dict))
    print(next(iter(amber_df_dict.values())))