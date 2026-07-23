import os
import requests

DATA_DIR = "data/raw_cmapss"
TARGET_FILE = os.path.join(DATA_DIR, "train_FD001")

# Doğrudan ham (raw) txt metin dosyası bağlantısı
URL = "https://raw.githubusercontent.com/biswajitsahoo1111/rul_codes_open/master/data/cmapss_data/train_FD001"


def download_file():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TARGET_FILE):
        print(f" Real C-MAPSS verisi zaten mevcut: {TARGET_FILE}")
        return

    print("📥 Gerçek NASA C-MAPSS FD001 verisi indiriliyor...")
    try:
        response = requests.get(URL, stream=True, timeout=15)
        response.raise_for_status()

        with open(TARGET_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✅ NASA C-MAPSS Verisi başarıyla yüklendi -> {TARGET_FILE}")
    except Exception as e:
        print(f"❌ İndirme hatası oluştu: {e}")


if __name__ == "__main__":
    download_file()