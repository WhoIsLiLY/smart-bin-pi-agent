import tensorflow as tf
import os
import numpy as np

# ==========================================
# 1. SETUP & KONFIGURASI
# ==========================================
MODEL_PATH = '20k_mobilenetv2_waste_classifier.keras' # Path model .keras Anda
IMG_SIZE = (224, 224)                    # Sesuaikan dengan input model (misal 224 atau 160)
BATCH_SIZE = 1                           # Batch size 1 untuk kalibrasi

print(f"Memuat model dari {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# 2. FUNGSI UNTUK MENGUKUR UKURAN FILE
# ==========================================
def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size / 1024 / 1024  # Konversi ke MB

# ==========================================
# SKENARIO A: BASELINE (FLOAT32)
# ==========================================
# Ini adalah model standar tanpa kompresi. Digunakan sebagai acuan akurasi tertinggi.
print("\n--- Mengonversi Model A: Baseline (Float32) ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Tidak ada flag optimizations = Float32 default
tflite_model_fp32 = converter.convert()

with open('model_baseline_float32.tflite', 'wb') as f:
    f.write(tflite_model_fp32)
print(f"Model Float32 tersimpan. Ukuran: {get_file_size('model_baseline_float32.tflite'):.2f} MB")

# ==========================================
# SKENARIO B: DYNAMIC RANGE QUANTIZATION
# ==========================================
# Ini yang Anda pakai sebelumnya. Bobot jadi Int8, kalkulasi tetap Float.
# Keseimbangan bagus antara ukuran kecil dan kecepatan.
print("\n--- Mengonversi Model B: Dynamic Range Quantization ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_dyn = converter.convert()

with open('model_dynamic_quant.tflite', 'wb') as f:
    f.write(tflite_model_dyn)
print(f"Model Dynamic Range tersimpan. Ukuran: {get_file_size('model_dynamic_quant.tflite'):.2f} MB")

try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set representative dataset
    converter.representative_dataset = representative_data_gen

    # Paksa operasi input/output menjadi integer juga (Opsional, tapi bagus untuk hardware murni)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # Input kamera nanti harus uint8
    converter.inference_output_type = tf.uint8 # Output prediksi uint8

    tflite_model_int8 = converter.convert()

    with open('model_full_integer_quant.tflite', 'wb') as f:
        f.write(tflite_model_int8)
    print(f"Model Full Integer tersimpan. Ukuran: {get_file_size('model_full_integer_quant.tflite'):.2f} MB")

except Exception as e:
    print(f"Gagal melakukan Full Integer Quantization. Pastikan path dataset benar. Error: {e}")

# ==========================================
# RANGKUMAN HASIL
# ==========================================
print("\n=== RANGKUMAN EKSPERIMEN UNTUK TESIS ===")
print(f"1. Baseline (Float32) : {get_file_size('model_baseline_float32.tflite'):.2f} MB")
print(f"2. Dynamic Range      : {get_file_size('model_dynamic_quant.tflite'):.2f} MB")
print("Gunakan data ini untuk Tabel Perbandingan di Bab 4.")