import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random
import os

# ================= KONFIGURASI =================
MODEL_FILENAME = "20k_mobilenetv3_waste_classifier.keras" 
IMG_SIZE = (160, 160)
TEST_DIR = "golden_test_set"

# Nama layer pembungkus (Lihat output summary Anda tadi)
WRAPPER_LAYER_NAME = "MobileNetV3Large" 
# Nama layer conv terakhir di dalam MobileNetV3 (Biasanya 'Conv_1')
INNER_CONV_LAYER_NAME = "conv_1"

# ================= 1. FUNGSI GRAD-CAM NESTED =================
def get_gradcam_heatmap(img_array, full_model, wrapper_name, inner_layer_name, pred_index=None):
    
    # LANGKAH 1: AKSES INNER MODEL
    # Kita ambil model MobileNetV3 yang ada di dalam model utama
    wrapper_layer = full_model.get_layer(wrapper_name)
    
    # Ini tricky: Wrapper layer di Keras Functional API sebenarnya adalah model juga
    # Kita perlu membuat "sub-model" yang outputnya adalah output layer Conv_1
    inner_model = wrapper_layer 
    
    # Buat Grad Model khusus untuk Inner Model
    # Input: Input MobileNetV3
    # Output: [Layer Conv Terakhir, Output MobileNetV3]
    grad_model = Model(
        inputs=inner_model.inputs, 
        outputs=[inner_model.get_layer(inner_layer_name).output, inner_model.output]
    )

    # LANGKAH 2: REKAM GRADIENT
    with tf.GradientTape() as tape:
        # Kita butuh input yang sudah melewati preprocessing awal (Sequential layer)
        # Tapi di model Anda, 'sequential' layer sepertinya hanya Rescaling/Augmentasi.
        # Kita coba pass img_array langsung ke grad_model.
        
        # Perlu preprocessing manual karena kita skip layer 'sequential' di depan
        # img_array biasanya 0-255, MobileNetV3 butuh input spesifik tergantung training.
        # Asumsi: layer 'sequential' Anda melakukan rescaling 1./255.
        # Mari kita coba jalankan grad_model dengan input saat ini.
        
        conv_outputs, predictions = grad_model(img_array)
        
        # predictions di sini outputnya (None, 5, 5, 960) dari MobileNetV3
        # Kita butuh skor klasifikasi akhir. 
        # INI MASALAHNYA: Output Inner Model belum masuk Dense Layer klasifikasi.
        # Kita tidak bisa pakai Grad-CAM standar cara ini untuk Nested Model.
        pass

    # --- STRATEGI ALTERNATIF YANG LEBIH ROBUST UNTUK NESTED MODEL ---
    # Kita buat model baru yang MENGGABUNGKAN bagian inner + bagian classifier
    
    # 1. Input baru
    new_input = tf.keras.Input(shape=IMG_SIZE + (3,))
    
    # 2. Jalan lewat Preprocessing (Sequential)
    x = full_model.get_layer("sequential")(new_input)
    
    # 3. Jalan lewat MobileNet (Akses layer conv-nya juga)
    mobilenet = full_model.get_layer(wrapper_name)
    
    # Kita ingin output intermediate dari dalam mobilenet
    target_layer_output = mobilenet.get_layer(inner_layer_name).output
    mobilenet_output = mobilenet.output
    
    # Buat model partial dari MobileNet
    partial_mobilenet = Model(inputs=mobilenet.inputs, outputs=[target_layer_output, mobilenet_output])
    
    conv_out, mobilenet_out = partial_mobilenet(x)
    
    # 4. Jalan lewat Classifier (GlobalAvg -> Dense -> Dropout -> Dense)
    x = full_model.get_layer("global_average_pooling2d_1")(mobilenet_out)
    x = full_model.get_layer("dense_2")(x)
    x = full_model.get_layer("dropout_1")(x)
    final_output = full_model.get_layer("dense_3")(x)
    
    # Model Spesial Grad-CAM
    grad_model_final = Model(inputs=new_input, outputs=[conv_out, final_output])
    
    # MULAI ULANG GRADIENT TAPE DENGAN MODEL BARU
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model_final(img_array)
        score = preds[0]
        if pred_index == 0: score = 1 - score # Invert untuk kelas 0
            
    grads = tape.gradient(score, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ================= 2. FUNGSI DISPLAY =================
def display_gradcam(img_path, heatmap, alpha=0.4):
    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

# ================= 3. MAIN EXECUTION =================
if not os.path.exists(MODEL_FILENAME):
    print(f"❌ File {MODEL_FILENAME} tidak ditemukan!")
else:
    print("⏳ Loading Model & Reconstructing Graph...")
    model = load_model(MODEL_FILENAME, compile=False)
    
    # Ambil sample
    sample_images = []
    anorg_path = os.path.join(TEST_DIR, 'anorganik')
    org_path = os.path.join(TEST_DIR, 'organik')
    
    if os.path.exists(anorg_path) and os.listdir(anorg_path):
        sample_images.append(os.path.join(anorg_path, random.choice(os.listdir(anorg_path))))
    if os.path.exists(org_path) and os.listdir(org_path):
        sample_images.append(os.path.join(org_path, random.choice(os.listdir(org_path))))

    if not sample_images:
        print("❌ Tidak ada gambar sample.")
    else:
        plt.figure(figsize=(10, 6))
        
        for i, img_path in enumerate(sample_images):
            try:
                img_tensor = img_to_array(load_img(img_path, target_size=IMG_SIZE))
                img_tensor = np.expand_dims(img_tensor, axis=0) # (1, 160, 160, 3)
                
                # Prediksi Biasa
                raw_preds = model.predict(img_tensor, verbose=0)
                score = tf.nn.sigmoid(raw_preds[0]).numpy()[0]
                pred_label = "Organik" if score > 0.5 else "Anorganik"
                pred_idx = 1 if score > 0.5 else 0
                conf = score if score > 0.5 else 1 - score
                
                # Buat Heatmap (Panggil Fungsi Kompleks tadi)
                heatmap = get_gradcam_heatmap(
                    img_tensor, model, 
                    WRAPPER_LAYER_NAME, INNER_CONV_LAYER_NAME, 
                    pred_index=pred_idx
                )
                
                final_img = display_gradcam(img_path, heatmap)
                
                # Plot
                plt.subplot(2, 2, i*2 + 1)
                plt.imshow(load_img(img_path, target_size=IMG_SIZE))
                plt.title(f"Asli: {os.path.basename(os.path.dirname(img_path)).upper()}")
                plt.axis('off')
                
                plt.subplot(2, 2, i*2 + 2)
                plt.imshow(final_img)
                plt.title(f"Grad-CAM: {pred_label} ({conf*100:.1f}%)")
                plt.axis('off')
                
            except Exception as e:
                print(f"❌ Error processing image {i}: {e}")
                # Print layer names in inner model to debug
                print("Daftar layer dalam MobileNetV3Large:")
                inner = model.get_layer(WRAPPER_LAYER_NAME)
                for l in inner.layers[-5:]: # Cek 5 layer terakhir
                    print(f" - {l.name}")

        plt.tight_layout()
        plt.show()