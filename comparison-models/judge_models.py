import os
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# ================= 1. PERSIAPAN DATA =================
# Unzip dataset jika belum (Safety check)
if os.path.exists('golden_test_set.zip') and not os.path.exists('golden_test_set'):
    print("📂 Sedang mengekstrak 'golden_test_set.zip'...")
    with zipfile.ZipFile('golden_test_set.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

IMG_SIZE = (160, 160)
TEST_DIR = "golden_test_set"
CLASS_NAMES = ['anorganik', 'organik'] # 0, 1

# List File Model (Sesuaikan nama file di Colab)
model_files = {
    "Model 5k": "5k_mobilenetv3_waste_classifier.keras",
    "Model 6.6k": "6.6k_mobilenetv3_waste_classifier.keras",
    "Model 7.8k": "7.8k_mobilenetv3_waste_classifier.keras",
    "Model 20k": "20k_mobilenetv3_waste_classifier.keras"
}

# ================= 2. LOAD GAMBAR & HITUNG JUMLAH =================
test_images_paths = []
y_true = []

print("-" * 60)
print("🔍 DATASET REPORT:")
for label_idx, label_name in enumerate(CLASS_NAMES):
    folder_path = os.path.join(TEST_DIR, label_name)
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"   📂 {label_name.upper()}: {len(files)} gambar")
        for fname in files:
            test_images_paths.append(os.path.join(folder_path, fname))
            y_true.append(label_idx)

# Hitung Total per Kelas untuk Statistik
total_anorganik = y_true.count(0)
total_organik = y_true.count(1)
total_all = len(y_true)

if total_all == 0:
    print("❌ Error: Tidak ada gambar.")
    exit()

print(f"📸 Total Semua: {total_all} gambar")
print("-" * 60)

# ================= 3. ENGINE VALIDASI DETAIL =================
results_summary = []

for label_model, filename in model_files.items():
    print(f"\n🤖 Menguji: {label_model} ...")
    
    if not os.path.exists(filename):
        print(f"   ⚠️ File model '{filename}' tidak ditemukan. Skip.")
        continue
        
    try:
        model = load_model(filename, compile=False)
        
        # Counter Variable
        correct_anorganik = 0
        correct_organik = 0
        
        # --- PREDIKSI LOOP ---
        for i, img_path in enumerate(test_images_paths):
            # Preprocess
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            # Predict & Logic
            predictions = model.predict(img_array, verbose=0)
            score = tf.nn.sigmoid(predictions[0]).numpy()[0]
            pred_class = 1 if score > 0.5 else 0
            
            # Cek Kebenaran & Kategori
            if pred_class == y_true[i]:
                # Jika Benar, cek dia kelas apa
                if y_true[i] == 0: # Anorganik
                    correct_anorganik += 1
                else: # Organik
                    correct_organik += 1

        # --- HITUNG STATISTIK ---
        acc_total = ((correct_anorganik + correct_organik) / total_all) * 100
        acc_anorganik = (correct_anorganik / total_anorganik) * 100 if total_anorganik > 0 else 0
        acc_organik = (correct_organik / total_organik) * 100 if total_organik > 0 else 0
        
        print(f"   ✅ Total Akurasi: {acc_total:.2f}%")
        print(f"      - Anorganik Benar: {correct_anorganik} dari {total_anorganik} ({acc_anorganik:.1f}%)")
        print(f"      - Organik Benar  : {correct_organik} dari {total_organik} ({acc_organik:.1f}%)")
        
        results_summary.append({
            "Nama Model": label_model,
            "Akurasi Total": f"{acc_total:.2f}%",
            "Anorganik (Benar/Total)": f"{correct_anorganik}/{total_anorganik}",
            "Akurasi Anorganik": f"{acc_anorganik:.1f}%",
            "Organik (Benar/Total)": f"{correct_organik}/{total_organik}",
            "Akurasi Organik": f"{acc_organik:.1f}%"
        })
        
        del model
        tf.keras.backend.clear_session()
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

# ================= 4. KLASEMEN AKHIR DETAIL =================
print("\n" + "="*80)
print("🏆 HASIL AKHIR DETAIL (ABLATION STUDY)")
print("="*80)

if results_summary:
    df = pd.DataFrame(results_summary)
    # Urutkan berdasarkan Akurasi Total tertinggi
    df = df.sort_values(by="Akurasi Total", ascending=False)
    
    # Tampilkan tabel yang bersih
    print(df.to_string(index=False))
    
    print("\n💡 TIPS ANALISIS UNTUK BAB 4:")
    best = df.iloc[0]
    print(f"1. Model terbaik secara umum adalah '{best['Nama Model']}'.")
    print("2. Perhatikan kolom 'Akurasi Anorganik' vs 'Akurasi Organik'.")
    print("   Jika timpang jauh (misal Anorganik 90% tapi Organik 50%),")
    print("   berarti model 'Bias' ke Anorganik. Cari model yang seimbang.")