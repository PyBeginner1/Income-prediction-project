from collections import namedtuple


#Configuration required to colelct data
DataIngestionConfig = namedtuple('DataIngestionConfig', 
                            ['dataset_download_url','download_data_dir','raw_data_dir','ingested_dir','ingested_train_dir', 'ingested_test_dir',
                            #'raw_data_file'
                            ])


TrainingPipelineConfig = namedtuple('TrainingPipelineConfig', ['artifact_dir'])