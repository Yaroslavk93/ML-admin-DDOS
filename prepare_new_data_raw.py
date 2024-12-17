# prepare_new_data_raw.py
# ---------------------------------------------
# Создание test-набора из BENIGN и DDoS для проверки инференса
# ---------------------------------------------
import pandas as pd

# Предполагаем, что файлы есть в data/raw/
ddos_df = pd.read_csv("data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", low_memory=False)
ddos_df.columns = ddos_df.columns.str.strip()
ddos_sample = ddos_df[ddos_df['Label'] == 'DDoS'].sample(n=500, random_state=42)

benign_df = pd.read_csv("data/raw/Wednesday-workingHours.pcap_ISCX.csv", low_memory=False)
benign_df.columns = benign_df.columns.str.strip()
benign_sample = benign_df[benign_df['Label'] == 'BENIGN'].sample(n=500, random_state=42)

new_data_raw = pd.concat([ddos_sample, benign_sample], ignore_index=True)
new_data_raw.to_csv("data/raw/new_data_raw.csv", index=False)
print("new_data_raw.csv создан, содержит 500 DDoS и 500 BENIGN.")