import os
import time
import glob
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import xarray as xr
import numpy as np


class ClimateDataExtractor:
    def __init__(self):
        self.historical_ds = None
        self.forecast_ds = None

    def _convert_forecast_to_historical_units(self, forecast_df: pd.DataFrame):
        """Converts forecast data units to match historical data units."""
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

    def _combine_dataframes(
        self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame
    ):
        """Combines historical and forecast dataframes, removing overlaps."""
        last_valid_historical = historical_df.last_valid_index()
        historical_df = historical_df.loc[:last_valid_historical]
        combined_df = pd.concat([historical_df, forecast_df], axis=0)
        return combined_df[~combined_df.index.duplicated(keep="first")]

    def _load_netcdf_data(self, folder: str, file_pattern: str, start_year: int, end_year: int):
        """Loads NetCDF files within the year range into an xarray dataset."""
        file_paths = []
        for year in range(start_year, end_year + 1):
            file_paths.extend(
                glob.glob(os.path.join(folder, file_pattern.format(year=year)))
            )

        if not file_paths:
            print(f"No matching files found for years {start_year} to {end_year}.")
            return None
        return xr.open_mfdataset(file_paths)

    def _process_point_batch(
        self, historical_ds, forecast_ds, variables, lat_batch, lon_batch, max_points=None
    ):
        """Processes a batch of points, optionally limiting the total processed points."""
        batch_start_time = time.time()
        batch_results = {}
        for i, (lat, lon) in enumerate(zip(lat_batch, lon_batch)):
            if max_points and i >= max_points:
                break  # Stop if max_points limit is reached within the batch

            historical_df = self._extract_point_data(historical_ds, lat, lon, variables)
            forecast_df = self._extract_point_data(forecast_ds, lat, lon, variables)
            forecast_df = self._convert_forecast_to_historical_units(forecast_df)
            batch_results[f"{lat},{lon}"] = self._combine_dataframes(
                historical_df, forecast_df
            )
        batch_end_time = time.time()
        print(f"Batch processed in {batch_end_time - batch_start_time:.2f} seconds")
        return batch_results

    def extract_variables_over_years(
        self,
        netcdf_folder: str,
        variables: list[str],
        file_name_pattern: str,
        start_year: int,
        end_year: int,
        mask_path: str,
        max_points: int = None,
    ):
        """Extracts and combines data for valid grid points."""
        mask = np.load(mask_path)

        self.historical_ds = self._load_netcdf_data(
            netcdf_folder, file_name_pattern, start_year, end_year
        ).load()
        self.forecast_ds = self._load_netcdf_data(
            "netcdf_files/forecasts_2024_05/r1i1p1", "combined.nc", 2024, 2024
        ).load()

        if not self.historical_ds or not self.forecast_ds:
            return {}

        lats, lons = self.historical_ds["lat"].values, self.historical_ds["lon"].values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        valid_lat_indices, valid_lon_indices = np.where(mask)

        df_dict = {}
        start_time = time.time()
        for count, (lat_idx, lon_idx) in enumerate(
            zip(valid_lat_indices, valid_lon_indices)
        ):
            if max_points and count >= max_points:
                break  # Stop processing if max_points limit is reached

            lat, lon = lat_grid[lat_idx, lon_idx], lon_grid[lat_idx, lon_idx]

            historical_df = self._extract_point_data(
                self.historical_ds, lat, lon, variables
            )
            forecast_df = self._extract_point_data(
                self.forecast_ds, lat, lon, variables
            )
            forecast_df = self._convert_forecast_to_historical_units(forecast_df.copy())

            df_dict[f"{lat},{lon}"] = self._combine_dataframes(
                historical_df, forecast_df
            )

        self.historical_ds.close()
        self.forecast_ds.close()
        print(
            f"Processed {count + 1} points in {time.time() - start_time:.2f} seconds."
        )
        return df_dict

    def extract_variables_over_years_parallel(
        self,
        netcdf_folder: str,
        variables: list[str],
        file_name_pattern: str,
        start_year: int,
        end_year: int,
        mask_path: str,
        batch_size: int = None,
        max_points: int = None,  # Add max_points parameter here
    ):
        """Extracts and combines data for valid grid points, processing in batches."""
        
        mask = np.load(mask_path)

        # Calculate batch size if not provided
        if batch_size is None:
            num_processes = os.cpu_count() or 1  # Get CPU count (default to 1)
            num_valid_points = np.count_nonzero(mask)
            batch_size = max(1, num_valid_points // num_processes)

        self.historical_ds = self._load_netcdf_data(
            netcdf_folder, file_name_pattern, start_year, end_year
        )
        self.forecast_ds = self._load_netcdf_data(
            "netcdf_files/forecasts_2024_05/r1i1p1", "combined.nc", 2024, 2024
        )

        if not self.historical_ds or not self.forecast_ds:
            return {}

        lats, lons = self.historical_ds["lat"].values, self.historical_ds["lon"].values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        valid_lat_indices, valid_lon_indices = np.where(mask)

        df_dict = {}
        start_time = time.time()
        points_processed = 0 # keep track of how many points we've processed

        with ProcessPoolExecutor() as executor:
            # Submit all batches as futures
            futures = []
            for i in range(0, len(valid_lat_indices), batch_size):
                # Pass max_points to _process_point_batch
                if max_points and points_processed >= max_points:
                    break  # Stop submitting batches if max_points limit is reached

                lat_batch = lat_grid[
                    valid_lat_indices[i : i + batch_size],
                    valid_lon_indices[i : i + batch_size],
                ]
                lon_batch = lon_grid[
                    valid_lat_indices[i : i + batch_size],
                    valid_lon_indices[i : i + batch_size],
                ]
                futures.append(
                    executor.submit(
                        self._process_point_batch,
                        self.historical_ds,
                        self.forecast_ds,
                        variables,
                        lat_batch,
                        lon_batch,
                        max_points=(max_points - points_processed)
                        if max_points
                        else None,  # Adjust max_points for each batch
                    )
                )
                points_processed += len(lat_batch)  # Update points_processed after submitting a batch

            # Collect results from completed futures
            for future in as_completed(futures):
                df_dict.update(future.result())

        self.historical_ds.close()
        self.forecast_ds.close()
        print(
            f"Processed {len(df_dict)} points in {time.time() - start_time:.2f} seconds."
        )
        return df_dict


if __name__ == "__main__":
    extractor = ClimateDataExtractor()
    max_points = 2000
    # df_dict = extractor.extract_variables_over_years(
    #     netcdf_folder="netcdf_files/combined",
    #     variables=["hurs", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin"],
    #     file_name_pattern="zalf_combined_amber_{year}_v1-0_uncompressed.nc",
    #     start_year=2023,
    #     end_year=2024,
    #     mask_path="data_availability_mask_h_and_fc.npy",
    #     max_points=max_points
    # )

    df_dict = extractor.extract_variables_over_years_parallel(
        netcdf_folder="netcdf_files/combined",
        variables=["hurs", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin"],
        file_name_pattern="zalf_combined_amber_{year}_v1-0_uncompressed.nc",
        start_year=2023,
        end_year=2024,
        mask_path="data_availability_mask_h_and_fc.npy",
        batch_size=100,
        max_points=max_points  # Pass max_points to the parallel method
    )
    print(len(df_dict))
    df = df_dict["54.80027617931778,9.647312950670669"]
    print(df)

    with open(f"{max_points}_points_extracted.pkl", "wb") as f:
        t0 = time.time()
        pickle.dump(df_dict, f)
        t1 = time.time()
        print(f"Saved file {max_points}_points_extracted.pkl in {t1-t0} seconds")