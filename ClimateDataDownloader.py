import requests
from datetime import datetime, timedelta
import os
import ftplib
import dotenv

dotenv.load_dotenv()


class ClimateDataDownloader:
    def __init__(self):
        pass

    def _calculate_end_date(self, start_date_str: str):
        """
        Calculates the end date (last day) of a month, six months in the future
        from the given year and month.

        Args:
            year: The year (int).
            month: The month (int, 1-12).

        Returns:
            A string representing the end date in the format 'YYYYMMDD'.
        """

        start_date = datetime.strptime(start_date_str, "%Y%m%d")

        # Calculate the year and month 6 months in the future
        future_year = start_date.year + (start_date.month + 6) // 12
        future_month = (start_date.month + 6) % 12
        if future_month == 0:
            future_month = 12

        # Get the last day of the future month
        future_date = datetime(future_year, future_month, 1) + timedelta(days=-1)

        return future_date.strftime("%Y%m%d")

    def _download_file(self, url: str, save_path: str):
        """Downloads a file from a given URL and saves it to the specified path."""
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"File '{os.path.basename(save_path)}' saved to '{save_path}'")

    def retrieve_seasonal_forecasts(
        self,
        experiment: str,
        driving_model: str,
        cast_package_prefix: str,
        cast_type_prefix: str,
        domain: str,
        variables: list[str],
        ensembles: list[str],
        version_suffix: str,
        target_folder: str,
        year: int,  # How does it look when the forecast spans multiple years?
        start_month: int,
    ):
        start_date_str = f"{year}{start_month:02}01"
        end_date_str = self._calculate_end_date(start_date_str)

        cast_package = f"{cast_package_prefix}{start_month:02}01"
        cast_type = f"{cast_type_prefix}{start_date_str}"
        version = f"v{year}{start_month:02}{version_suffix}"
        """Retrieves seasonal forecast data from the specified source."""

        base_url = "https://esgf.dwd.de/thredds/fileServer/esgf3_1/climatepredictionsde"

        for variable in variables:
            for ensemble in ensembles:
                folder_path = f"{experiment}/output/public/{domain}/DWD/{driving_model}/{cast_package}/{cast_type}/{ensemble}/DWD-EPISODES2022/v1-r1/day/{variable}/{version}"
                file_name = f"{variable}_day_{driving_model}--DWD-EPISODES2022--{domain}_{cast_type}_{ensemble}_{start_date_str}-{end_date_str}.nc"
                download_url = f"{base_url}/{folder_path}/{file_name}"

                # Create the subfolder if it doesn't exist
                subfolder_path = os.path.join(target_folder, start_date_str, ensemble)
                os.makedirs(subfolder_path, exist_ok=True)

                save_path = os.path.join(subfolder_path, file_name)
                self._download_file(download_url, save_path)

    def retrieve_amber_files(self, target_folder: str, start_year: int, end_year: int, ftp_hostname:str, ftp_user:str, ftp_password:str):
        """Retrieves NetCDF files from an FTP server for the specified year range.

        Args:
            target_folder (str): The path to the local directory to save the files.
            start_year (int): The starting year for the range of years.
            end_year (int): The ending year for the range of years.
        """
        if not (start_year and end_year):
            raise ValueError("Please provide both 'start_year' and 'end_year'.")

        with ftplib.FTP_TLS(host=ftp_hostname) as ftps:
            ftps.login(user=ftp_user, passwd=ftp_password)
            ftps.prot_p()
            ftps.cwd("DWD_SpreeWasser_N")

            for year in range(start_year, end_year + 1):
                year_str = str(year)
                for filename in ftps.nlst():
                    if year_str in filename:
                        local_filepath = os.path.join(target_folder, year_str, filename)
                        os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
                        with open(local_filepath, "wb") as file:
                            ftps.retrbinary("RETR " + filename, file.write)
                        print(f"Downloaded: {filename}. Saved to: {local_filepath}")
