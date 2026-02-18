import tensorflow as tf
import os

# Daftar file .h5 sumber
files = ["mobilenetv3.h5", "resnet50.h5", "vgg16.h5"]

print("Mulai Konversi ke TFLite...")

for h5_file in files:
    if os.path.exists(h5_file):
        print(f"Mengonversi {h5_file}...", end="")
        try:
            # Load Model
            model = tf.keras.models.load_model(h5_file)
            
            # Convert
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            # Save
            tflite_filename = h5_file.replace(".h5", ".tflite")
            with open(tflite_filename, 'wb') as f:
                f.write(tflite_model)
            print(f" SUKSES -> {tflite_filename}")
        except Exception as e:
            print(f" GAGAL: {e}")
    else:
        print(f"File {h5_file} tidak ditemukan, skip.")