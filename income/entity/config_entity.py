from collections import namedtuple


#Configuration required to collect data
DataIngestionConfig = namedtuple('DataIngestionConfig', 
                            ['dataset_download_url','download_data_dir','raw_data_dir','ingested_dir','ingested_train_dir', 'ingested_test_dir',
                            #'raw_data_file'
                            ])


DataValidationConfig = namedtuple('DataValidationConfig',
                                ['schema_file_path','report_file_path','report_page_file_path'])


TrainingPipelineConfig = namedtuple('TrainingPipelineConfig', ['artifact_dir'])