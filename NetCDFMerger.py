import os

import xarray as xr
import glob

class NetCDFMerger:
    def __init__(self):
        # You might not need any initialization here, but you can add if necessary
        pass 

    def _merge_netcdf_files(self, file_pattern, output_file):
        """Merges multiple netCDF files matching a pattern into a single file.

        Args:
            file_pattern (str): A glob pattern to match the netCDF files.
            output_file (str): The path to save the merged netCDF file.
        """
        files = [f for f in glob.glob(file_pattern) if "merged_" not in os.path.basename(f)]
        if files:
            datasets = [xr.open_dataset(file) for file in files]
            merged_dataset = xr.merge(datasets, compat="override")
            encoding = {var: {"zlib": False} for var in merged_dataset.data_vars}
            merged_dataset.to_netcdf(output_file, encoding=encoding)
            print(f"Merged files to: {output_file}")
        else:
            print(f"No files found matching pattern: {file_pattern}")

    def merge_esgf_files(self, base_dir: str, year: int, month: int, ensembles: list[str]):
        """Merges ESGF netCDF files for a single year-month combination and all ensembles.

        Args:
            base_dir (str): The base directory containing start date subfolders (e.g., '20240501').
            year (int): The year of the data to merge.
            month (int): The month of the data to merge (1-12). 
        """
        start_date_str = f"{year}{month:02}01" 
        start_date_dir = os.path.join(base_dir, start_date_str)

        if os.path.exists(start_date_dir):
            for ensemble in ensembles:
                ensemble_dir = os.path.join(start_date_dir, ensemble)
                file_pattern = os.path.join(ensemble_dir, "*.nc")
                output_file = os.path.join(ensemble_dir, f"merged_{start_date_str}_{ensemble}.nc")
                self._merge_netcdf_files(file_pattern, output_file)

    def merge_amber_files(self, base_dir: str, start_year: int, end_year: int):
        """Merges AMBER netCDF files organized by year.

        Args:
            base_dir (str): The base directory containing year subfolders.
            start_year (int): The starting year for merging files.
            end_year (int): The ending year for merging files.
        """
        for year in range(start_year, end_year + 1):
            file_pattern = os.path.join(base_dir, str(year), "*.nc")
            output_file = os.path.join(base_dir, str(year), f"zalf_merged_amber_{year}_v1-0.nc")
            self._merge_netcdf_files(file_pattern, output_file)