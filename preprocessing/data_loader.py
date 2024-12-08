# preprocessing/data_loader.py
# ---------------------------------------------
# Загрузчик данных: читает все файлы CICIDS2017, объединяет в один DataFrame
# ---------------------------------------------
import os
import pandas as pd

class DataLoader:
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
                # Очистка названий столбцов от пробелов
                temp_df.columns = temp_df.columns.str.strip()
                temp_df['SourceFile'] = f
                df_list.append(temp_df)
            else:
                print(f"Файл не найден: {full_path}")

        if df_list:
            merged_df = pd.concat(df_list, ignore_index=True)
            return merged_df
        else:
            print("Нет данных для обработки.")
            return pd.DataFrame()