import time
import numpy as np
import tensorflow.lite as tflite
import os

# DAFTAR MODEL YANG AKAN DITES
models = [
    "mobilenetv3.tflite", 
    "mobilenetv2.tflite", 
    "efficientnetb0.tflite"
]

def benchmark_model(model_path):
    if not os.path.exists(model_path):
        print(f"❌ File {model_path} tidak ditemukan!")
        return

    print(f"\n🚀 Memulai Benchmark: {model_path}...")
    
    # Load TFLite
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    
    # Generate Dummy Data (Random Noise) - Biar murni ngetes kecepatan prosesor
    # Asumsi input float32. Kalau uint8 (quantized), ganti tipe datanya.
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    # WARMUP (Pemanasan) - Biar cache processor siap
    # Model pertama kali run biasanya lambat, jadi kita buang hasil pertamanya
    print("   🔥 Warming up (10 runs)...")
    interpreter.set_tensor(input_details[0]['index'], input_data)
    for _ in range(10):
        interpreter.invoke()

    # REAL BENCHMARK
    iterations = 50 # Jumlah pengulangan
    print(f"   ⏱️  Mengukur kecepatan rata-rata dari {iterations} kali inferensi...")
    
    start_time = time.time()
    for _ in range(iterations):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        # Ambil output biar prosesnya tuntas
        _ = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.time()

    # Hitung Statistik
    total_time = end_time - start_time
    avg_latency_ms = (total_time / iterations) * 1000
    fps = 1.0 / (total_time / iterations)

    print(f"   ✅ Selesai!")
    print(f"   📊 Latency Rata-rata : {avg_latency_ms:.2f} ms")
    print(f"   ⚡ Frame Rate (FPS)  : {fps:.2f} FPS")
    print("-" * 40)

# Jalankan Loop
print("=" * 40)
print(" BENCHMARK PERFORMA RASPBERRY PI 5 ")
print("=" * 40)

for model in models:
    benchmark_model(model)