from income.exception import IncomeException
from income.logger import logging
from income.entity.config_entity import DataIngestionConfig
from income.entity.artifact_entity import DataIngestionArtifact
import sys, os 
import pandas as pd
import numpy as np 
from income.constant import *
import urllib.request 


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ") 
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise IncomeException(e, sys) from e 


    def download_income_data(self):
        try:
            download_url =  self.data_ingestion_config.dataset_download_url
            download_data_dir = self.data_ingestion_config.download_data_dir

            os.makedirs(download_data_dir, exist_ok = True)

            income_filename = 'adult.data'

            download_file_path = os.path.join(download_data_dir,income_filename)
            logging.info(f"Downloading file from :[{download_url}] into :[{download_file_path}]")

            urllib.request.urlretrieve(download_url, download_file_path)
            logging.info(f"File :[{income_filename}] has been downloaded successfully.")
            return download_file_path
        except Exception as e:
            raise IncomeException(e, sys) from e 


    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            pass
        except Exception as e:
            raise IncomeException(e, sys) from e 

    