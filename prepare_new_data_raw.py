# prepare_new_data_raw.py
# ---------------------------------------------
# Скрипт для создания new_data_raw.csv из части исходных данных
# Берем немного DDoS и немного BENIGN для теста
# ---------------------------------------------
import pandas as pd

# Загрузка данных с DDoS
ddos_df = pd.read_csv("data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", low_memory=False)
ddos_df.columns = ddos_df.columns.str.strip()
ddos_sample = ddos_df[ddos_df['Label'] == 'DDoS'].sample(n=500)

# Загрузка данных с BENIGN
benign_df = pd.read_csv("data/raw/Wednesday-workingHours.pcap_ISCX.csv", low_memory=False)
benign_df.columns = benign_df.columns.str.strip()
benign_sample = benign_df[benign_df['Label'] == 'BENIGN'].sample(n=500)

# Объединяем выборки
new_data_raw = pd.concat([ddos_sample, benign_sample], ignore_index=True)
new_data_raw.to_csv("data/raw/new_data_raw.csv", index=False)