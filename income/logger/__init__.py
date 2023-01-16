import logging
from datetime import datetime
from income.constant import get_current_time_stamp
import os

LOG_DIR = 'housing_logs'
CURRENT_TIMESTAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

def get_log_file_name():
    return f"log_{get_current_time_stamp()}.log"

LOG_FILE_NAME = f"log_{CURRENT_TIMESTAMP}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE_NAME)

os.makedirs(LOG_DIR,exist_ok=True)

logging.basicConfig(filename = LOG_FILE_PATH, 
filemode='w', 
format = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
level = logging.INFO)

def get_log_dataframe(file_path):
    data=[]
    with open(file_path) as log_file:
        for line in log_file.readlines():
            data.append(line.split("^;"))

    log_df = pd.DataFrame(data)
    columns=["Time stamp","Log Level","line number","file name","function name","message"]
    log_df.columns=columns
    
    log_df["log_message"] = log_df['Time stamp'].astype(str) +":$"+ log_df["message"]

    return log_df[["log_message"]]