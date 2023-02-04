import yaml 
import os,sys
import numpy as np
import dill
import pandas as pd 

from income.exception import IncomeException
from income.constant import *

def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise IncomeException(e,sys)
 
 
def read_yaml_file(file_path:str)->dict:
    '''Reads a YAML File and returns dictionary'''
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise e


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save Numpy array to file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise IncomeException(e, sys) from e



def load_numpy_array_data(file_path: str)->np.array:
    """
    Load the saved Numpy array
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj,allow_pickle=True)
    except Exception as e:
        raise IncomeException(e, sys) from e



def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise IncomeException(e,sys) from e



def load_object(file_path:str):
    """
    file_path: str
    """
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if os.path.exists(dir_path):
        try:
            with open(file_path, "rb") as file_obj:
                return dill.load(file_obj)
        except Exception as e:
            raise IncomeException(e,sys) from e
    else:
        raise FileNotFoundError(f"{file_path} does not exist")





def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
    try:
        dataset_schema = read_yaml_file(schema_file_path)  
        schema = dataset_schema[DATASET_SCHEMA_COLUMNS_KEY]

        dataframe = pd.read_csv(file_path)

        error_message = ''

        drop_column = ['Unnamed: 0']

        for column in dataframe.columns:
            if column == drop_column:
                dataframe.drop(drop_column, axis = 1, inplace = True)
            if column in list(schema.keys()):
                dataframe[column].astype(schema[column])
            else:
                error_message = f"{error_message}\n Column: {column} not in schema"

        if len(error_message) > 0:
            raise Exception(error_message)
        return dataframe

    except Exception as e:
        raise IncomeException(e,sys) from e 