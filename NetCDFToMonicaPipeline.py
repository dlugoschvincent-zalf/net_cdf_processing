import numpy as np
from ClimateDataDownloader import ClimateDataDownloader
from ClimateDataExtractor import ClimateDataExtractor
from NetCDFMerger import NetCDFMerger
import dotenv
import os

dotenv.load_dotenv()


class NetCDFToMonicaPipeline:
    def __init__(self):
        self.downloader = ClimateDataDownloader()
        self.merger = NetCDFMerger()
        self.extractor = ClimateDataExtractor()

    def forecast_pipeline(
        self,
        ensembles: list[str],
        variables: list[str],
        forecast_directory: str,
        year: int,
        month: int,
        version_suffix:str
    ):
        self.downloader.retrieve_seasonal_forecasts(
            experiment="seasonal",
            driving_model="GCFS21",
            cast_package_prefix="svhYYYY",
            cast_type_prefix="sfc",
            domain="DE-0075x005",
            variables=variables,
            ensembles=ensembles,
            version_suffix=version_suffix,
            start_month=month,
            year=year,
            target_folder=forecast_directory,
        )
        self.merger.merge_esgf_files(
            base_dir=forecast_directory, year=year, month=month, ensembles=ensembles
        )

        ndarray_of_valid_lat_lon_tuples = np.load("valid_grid_points.npy")

        for enseble in ensembles:
            forecast_data_directory = f"cache/forecast/{year}{month:02}01"
            forecast_data_file = f"{enseble}_extracted.pkl.lz4"

            forecast_df_dict = self.extractor.extract_forecast_data(
                netcdf_folder_pattern="netcdf_files/forecasts/20240501/{ensemble}",
                variables=variables,
                file_name_pattern="merged_20240501_{ensemble}.nc",
                ndarray_of_valid_lat_lon_tuples=ndarray_of_valid_lat_lon_tuples,
                ensemble=enseble,
                max_points=2000,
            )

        self.extractor.save_data(
            forecast_df_dict, forecast_data_directory, forecast_data_file
        )

    def amber_pipeline(
        self,
        variables: list[str],
        amber_directory: str,
        start_year: int,
        end_year: int,
        ftp_hostname: str,
        ftp_user: str,
        ftp_password: str,
    ):
        self.downloader.retrieve_amber_files(
            target_folder=amber_directory,
            start_year=start_year,
            end_year=end_year,
            ftp_hostname=ftp_hostname,
            ftp_user=ftp_user,
            ftp_password=ftp_password,
        )
        self.merger.merge_amber_files(
            base_dir=amber_directory, start_year=start_year, end_year=end_year
        )

        ndarray_of_valid_lat_lon_tuples = np.load("valid_grid_points.npy")
        amber_data_directory = f"cache/amber"
        amber_data_file = f"{start_year}_to_{end_year}_extracted.pkl.lz4"

        amber_df_dict = self.extractor.extract_amber_data(
            netcdf_folder_pattern="netcdf_files/amber/{year}",
            variables=variables,
            file_name_pattern="zalf_merged_amber_{year}_v1-0.nc",
            start_year=start_year,
            end_year=end_year,
            ndarray_of_valid_lat_lon_tuples=ndarray_of_valid_lat_lon_tuples,
            max_points=2000,
        )
        self.extractor.save_data(amber_df_dict, amber_data_directory, amber_data_file)


if __name__ == "__main__":
    pipeline = NetCDFToMonicaPipeline()
    variables = ["hurs", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]
    ensembles = [
        "r10i1p1",
        # "r11i1p1",
        # "r12i1p1",
        # "r13i1p1",
        # "r14i1p1",
        # "r15i1p1",
        # "r16i1p1",
        # "r17i1p1",
        # "r18i1p1",
        # "r19i1p1",
        # "r1i1p1",
        # "r20i1p1",
        # "r21i1p1",
        # "r22i1p1",
        # "r23i1p1",
        # "r24i1p1",
        # "r25i1p1",
        # "r26i1p1",
        # "r27i1p1",
        # "r28i1p1",
        # "r29i1p1",
        # "r2i1p1",
        # "r30i1p1",
        # "r31i1p1",
        # "r32i1p1",
        # "r33i1p1",
        # "r34i1p1",
        # "r35i1p1",
        # "r36i1p1",
        # "r37i1p1",
        # "r38i1p1",
        # "r39i1p1",
        # "r3i1p1",
        # "r40i1p1",
        # "r41i1p1",
        # "r42i1p1",
        # "r43i1p1",
        # "r44i1p1",
        # "r45i1p1",
        # "r46i1p1",
        # "r47i1p1",
        # "r48i1p1",
        # "r49i1p1",
        # "r4i1p1",
        # "r50i1p1",
        # "r5i1p1",
        # "r6i1p1",
        # "r7i1p1",
        # "r8i1p1",
        # "r9i1p1",
    ]

    pipeline.forecast_pipeline(
        ensembles=ensembles,
        variables=variables,
        forecast_directory=os.getenv("FORECAST_DIRECTORY"),
        year=int(os.getenv("FORECAST_YEAR")),
        month=int(os.getenv("FORECAST_START_MONTH")),
        version_suffix='08'
    )

    # pipeline.amber_pipeline(

    #     variables=variables,
    #     amber_directory=os.getenv("AMBER_DIRECTORY"),
    #     start_year=2023,
    #     end_year=2024,
    #     ftp_hostname=os.getenv("FTP_HOSTNAME"),
    #     ftp_password=os.getenv("FTP_PASSWORD"),
    #     ftp_user=os.getenv("FTP_USER"),
    # )
