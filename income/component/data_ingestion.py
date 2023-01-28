from income.exception import IncomeException
from income.logger import logging
from income.entity.config_entity import DataIngestionConfig
from income.entity.artifact_entity import DataIngestionArtifact
import sys, os 
import pandas as pd
import numpy as np 
from income.constant import *
import urllib.request 
from imblearn.over_sampling import RandomOverSampler
from income.config.configuration import Configuration  
from sklearn.model_selection import StratifiedShuffleSplit


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ") 
            self.data_ingestion_config = data_ingestion_config 
        except Exception as e:
            raise IncomeException(e, sys) from e 


    def download_income_data(self):
        try:
            global download_file_path
            download_url =  self.data_ingestion_config.dataset_download_url
            download_data_dir = self.data_ingestion_config.download_data_dir

            os.makedirs(download_data_dir, exist_ok = True)

            income_filename = 'adult'

            download_file_path = os.path.join(download_data_dir,income_filename)
            logging.info(f"Downloading file from :[{download_url}] into :[{download_file_path}]")

            urllib.request.urlretrieve(download_url, download_file_path)
            logging.info(f"File :[{income_filename}] has been downloaded successfully.")
            return download_file_path
        except Exception as e:
            raise IncomeException(e, sys) from e 


    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            income_file_path = download_file_path
            #file_name = os.listdir(income_file_path)[0]
            file_name = 'adult'

            logging.info(f"Reading csv file: {income_file_path}")

            income_dataframe = pd.read_csv(income_file_path, names = ['age','workclass','fnlwgt','education','education-num', 'marital-status',\
                                                                    'occupation','relationship','race', 'sex', 'capital-gain',\
                                                                    'capital-loss','hours-per-week','native-country','income'])

            logging.info('Splitting data into Train and test')
            strat_train_set = None
            strat_test_set = None
            
            '''(Data Transformation)
            replace_special = ['workclass', 'occupation','native-country']
            for i in replace_special:
                income_dataframe[i] = income_dataframe[i].replace(' ?',np.nan)
                income_dataframe[i] = income_dataframe[i].fillna(income_dataframe[i].mode([0]))

            drop_columns = ['workclass','fnlwgt', 'education', 'occupation', 'native-country']
            income_dataframe = income_dataframe.drop(drop_columns, axis = 1)

            income_dataframe['sex'] = income_dataframe['sex'].map({' Male':1, ' Female':0})
            income_dataframe['income'] = pd.get_dummies(income_dataframe['income'], drop_first = True)

            categorical_columns = [col for col in income_dataframe.columns if income_dataframe[col].dtypes == 'object']
            income_dataframe = pd.get_dummies(income_dataframe, categorical_columns, drop_first = True)
            

            sampler = RandomOverSampler(random_state = 23)
            sampler.fit(X,y)
            X_sampled, y_sampled = sampler.fit_resample(X,y)
            '''

            X = income_dataframe.drop('income', axis = 1)
            y = income_dataframe['income']

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            for train_index,test_index in split.split(X, y):
                strat_train_set = income_dataframe.loc[train_index]
                strat_test_set = income_dataframe.loc[test_index]


            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, file_name)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir, file_name)

            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok = True)
                logging.info(f"Exporting train datset to path: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path, index = False)

            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok = True)
                logging.info(f"Exporting test datset to path: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path, index = False)


            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                            test_file_path=test_file_path,
                                                            is_ingested=True,
                                                            message=f"Data ingestion completed successfully.")

            return data_ingestion_artifact
            
        except Exception as e:
            raise IncomeException(e, sys) from e 

    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            download_file_path = self.download_income_data() 
            return self.split_data_as_train_test()
        except Exception as e:
            raise IncomeException(e,sys) from e 

    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")

    