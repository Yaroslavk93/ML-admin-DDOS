# preprocessing/data_loader.py
# ---------------------------------------------
# Класс для загрузки и объединения CSV файлов CICIDS2017
# ---------------------------------------------
import os
import pandas as pd

class DataLoader:
    """
    Класс для загрузки сырых данных CICIDS2017 из data/raw/.
    Объединяет несколько CSV в один DataFrame.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.files = [
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        ]

    def load_data(self) -> pd.DataFrame:
        df_list = []
        for f in self.files:
            full_path = os.path.join(self.data_path, f)
            if os.path.exists(full_path):
                temp_df = pd.read_csv(full_path, low_memory=False)
                temp_df.columns = temp_df.columns.str.strip()
                temp_df['SourceFile'] = f
                df_list.append(temp_df)
            else:
                print(f"Файл не найден: {full_path}")
        if df_list:
            return pd.concat(df_list, ignore_index=True)
        else:
            print("Нет данных для обработки.")
            return pd.DataFrame()