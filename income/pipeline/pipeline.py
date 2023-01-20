from income.config.configuration import Configuration
from income.logger import logging
from income.exception import IncomeException

from income.entity.artifact_entity import DataIngestionArtifact
from income.entity.config_entity import DataIngestionConfig

from income.component.data_ingestion import DataIngestion

import sys


class Pipeline:

    def __init__(self, config : Configuration = Configuration()):
        try:
            self.config = config 
        except Exception as e:
            raise IncomeException(e,sys) from e
        
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config()) 
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise IncomeException(e,sys) from e
        

    def run_pipeline(self):
        try:
            logging.info('Initiating Pipeline')
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise IncomeException(e,sys) from e