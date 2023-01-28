from income.logger import logging
from income.exception import IncomeException
from income.entity.config_entity import DataTransformationConfig
from income.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact, DataTransformationArtifact
from income.config.configuration import Configuration
from income.util.util import read_yaml_file,save_object, save_numpy_array_data, load_data
from income.constant import * 

import os,sys
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer



class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact,
                                        data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"{'-'*20} Data Transformation Started {'-'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise IncomeException(e,sys) from e 



    def get_data_transformer(self)->ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(file_path=schema_file_path)
            
            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]
            '''
            num_pipeline = Pipeline(steps = [
                ('imputer', SimpleImputer()),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer()),
                ('one_hot_encoder',OneHotEncoder()),
                ('scaler',StandardScaler())
            ])'''

            logging.info(f"Numerical Columns : {numerical_columns}")
            logging.info(f"Categorial Columns : {categorical_columns}")

            '''
            preprocessing = ColumnTransformer(
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            )'''



            return numerical_columns, categorical_columns
        except Exception as e:
            raise IncomeException(e,sys) from e
        


    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            
            schema_file_path = self.data_validation_artifact.schema_file_path
            dataset_schema = read_yaml_file(file_path=schema_file_path)

            logging.info('Obtaining Preprocessing Object')

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]

            logging.info(f"Numerical Columns : {numerical_columns}")
            logging.info(f"Categorial Columns : {categorical_columns}")
            
            logging.info('Obtaining Training, Test and Schema file path')
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            schema_file_path = self.data_validation_artifact.schema_file_path

            logging.info('Loading Training and Testing as Pandas Dataframe')
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            test_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            
            schema = read_yaml_file(file_path = schema_file_path)
            target_column_name = schema[TARGET_COLUMN_KEY]

            logging.info('Splitting Input and Target in Train and Test Dataframe')
            input_feature_train_df = train_df.drop(target_column_name, axis = 1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(target_column_name, axis = 1)
            target_feature_test_df = test_df[target_column_name]

            
            replace_special = ['workclass', 'occupation','native-country']
            for i in replace_special:
                input_feature_train_df[i] = input_feature_train_df[i].replace(' ?',np.nan)
                input_feature_test_df[i] = input_feature_train_df[i].replace(' ?',np.nan)
                input_feature_train_df[i] = input_feature_train_df[i].fillna(input_feature_train_df[i].mode([0]))
                input_feature_test_df[i] = input_feature_train_df[i].fillna(input_feature_train_df[i].mode([0]))

            drop_columns = ['workclass','fnlwgt', 'education', 'occupation', 'native-country']
            input_feature_train_df = input_feature_train_df.drop(drop_columns, axis = 1)
            input_feature_test_df = input_feature_test_df.drop(drop_columns, axis = 1)

            categorical_columns = [col for col in input_feature_train_df.columns if input_feature_train_df[col].dtypes == 'object']
            input_feature_train_df = pd.get_dummies(input_feature_train_df, categorical_columns, drop_first = True)
            categorical_columns = [col for col in input_feature_test_df.columns if input_feature_test_df[col].dtypes == 'object']
            input_feature_test_df = pd.get_dummies(input_feature_test_df, categorical_columns, drop_first = True)
         

            transformed_train_dir_path = self.data_transformation_config.transformed_train_dir
            transformed_test_dir_path = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace('.csv','.npz')
            test_file_name = os.path.basename(test_file_path).replace('.csv','.npz') 

            transformed_train_file_path = os.path.join(transformed_train_dir_path,train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir_path,test_file_name)

            train_arr = np.c_[input_feature_train_df,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df,np.array(target_feature_test_df)]

            logging.info('Saving Transformed Train and Test array')
            save_numpy_array_data(file_path=transformed_train_file_path, array = train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path, array = test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            data_transformation_artifact = DataTransformationArtifact(
                is_transformed = True,
                message = 'Data Transformed successfully',
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                preprocessed_object_file_path=preprocessing_obj_file_path
            )

            logging.info(f'Data Transformation Artifact: {data_transformation_artifact}')

            return data_transformation_artifact
            
            '''logging.info('Applying preprocessing object on Train and Test Dataframe')
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace('.csv','.npz')
            test_file_name = os.path.basename(test_file_path).replace('.csv','.npz') 

            transformed_train_file_path = os.path.join(transformed_train_dir,train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir,test_file_name)

            logging.info('Saving Transformed Train and Test array')
            save_numpy_array_data(file_path=transformed_train_file_path, array = train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path, array = test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info('Saving the Preprocessing object')
            save_object(file_path = preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(
                is_transformed = True,
                message = 'Data Transformed successfully',
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                preprocessed_object_file_path=preprocessing_obj_file_path
            )

            logging.info(f'Data Transformation Artifact: {data_transformation_artifact}')'''

            #return data_transformation_artifact
        except Exception as e:
            raise IncomeException(e,sys) from e 