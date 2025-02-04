import pandas as pd
import gdown
import zipfile
import os
#Carregando o dataset
file_id = "1TYKt_2HRMBL_B_PmZrIoMJH40nBuGs2Y"
download_link = f"https://drive.google.com/uc?id={file_id}"
local_filename = "Merged01.csv"
gdown.download(download_link, local_filename, quiet=False)
data = pd.read_csv(local_filename)
print(data.head())