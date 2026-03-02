import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import zoom
import random
import os
import warnings
warnings.filterwarnings('ignore')

# ================= KONFIGURASI =================
MODEL_FILENAME = "20k_mobilenetv3_waste_classifier_new.keras"
IMG_SIZE = (160, 160)
TEST_DIR = "golden_test_set"
CLASS_NAMES = ["Anorganik", "Organik"]

# Layer conv yang akan dipakai untuk Grad-CAM
# Urutan preferensi: layer lebih awal = resolusi lebih tinggi = heatmap lebih tajam
# Untuk MobileNetV3Large, pilihan yang bagus:
CANDIDATE_LAYERS = [
    "expanded_conv_14_project_BN",   # Late block, resolusi lebih tinggi dari conv_1
    "Conv_1_bn",                      # Batch norm setelah Conv_1
    "conv_1",                         # Default fallback
]

# ================= UTILITAS =================
def find_best_conv_layer(mobilenet_model):
    """
    Cari layer conv/BN terbaik berdasarkan nama yang diketahui ada di MobileNetV3Large.
    Urutan prioritas: layer lebih awal = resolusi spatial lebih tinggi = heatmap lebih tajam.
    """
    # Prioritas dari resolusi tinggi ke rendah
    priority_layers = [
        "expanded_conv_14_project_bn",
        "expanded_conv_13_project_bn",
        "expanded_conv_12_project_bn",
        "conv_1_bn",
        "activation_19",
        "conv_1",
    ]

    available = {l.name for l in mobilenet_model.layers}
    print(f"\n   Layer tersedia ({len(available)} total), mencari kandidat...")

    for candidate in priority_layers:
        if candidate in available:
            print(f"   → '{candidate}' ditemukan ✅")
            return candidate

    # Last resort: Conv2D atau BatchNorm pertama dari belakang
    for layer in reversed(mobilenet_model.layers):
        ltype = layer.__class__.__name__
        if any(t in ltype for t in ['BatchNormalization', 'Conv2D']):
            print(f"   → Last resort: '{layer.name}'")
            return layer.name

    return None

def list_inner_layers(model, wrapper_name="MobileNetV3Large", last_n=10):
    """Tampilkan layer-layer terakhir dalam MobileNetV3 untuk debugging"""
    print(f"\n{'='*50}")
    print(f"Layer terakhir dalam {wrapper_name}:")
    inner = model.get_layer(wrapper_name)
    for layer in inner.layers[-last_n:]:
        try:
            out = layer.output_shape
        except AttributeError:
            out = "(shape tidak tersedia)"
        print(f"  [{layer.__class__.__name__:25s}] {layer.name:45s} -> {out}")
    print('='*50)

# ================= GRAD-CAM CORE =================
def build_gradcam_model(full_model, conv_layer_name, wrapper_name="MobileNetV3Large"):
    """
    Bangun model khusus untuk Grad-CAM yang meng-expose:
    - Output conv layer yang dipilih
    - Output final (logit) model
    
    Menangani arsitektur nested (model di dalam model).
    """
    mobilenet = full_model.get_layer(wrapper_name)

    # Cek apakah layer ada
    layer_names = [l.name for l in mobilenet.layers]
    if conv_layer_name not in layer_names:
        print(f"⚠️  Layer '{conv_layer_name}' tidak ditemukan. Mencari otomatis...")
        conv_layer_name = find_best_conv_layer(mobilenet)
        print(f"✅ Menggunakan layer: '{conv_layer_name}'")

    # Bangun partial MobileNet: input → [conv_target, mobilenet_output]
    partial_mobilenet = Model(
        inputs=mobilenet.inputs,
        outputs=[
            mobilenet.get_layer(conv_layer_name).output,
            mobilenet.output
        ]
    )

    # Susun model lengkap: raw_input → augmentasi → mobilenet → classifier
    new_input = tf.keras.Input(shape=IMG_SIZE + (3,))

    # Pass lewat data augmentation (Sequential layer)
    try:
        x = full_model.get_layer("sequential")(new_input, training=False)
    except:
        x = new_input  # Jika tidak ada sequential layer

    # Pass lewat mobilenet (dapat 2 output)
    conv_out, mob_out = partial_mobilenet(x)

    # Pass lewat classifier head
    # Cari layer GAP, Dense, Dropout secara dinamis
    head_layers = []
    found_gap = False
    for layer in full_model.layers:
        ltype = layer.__class__.__name__
        if 'GlobalAveragePooling' in ltype:
            found_gap = True
        if found_gap:
            head_layers.append(layer)

    x = mob_out
    for layer in head_layers:
        try:
            x = layer(x, training=False)
        except:
            x = layer(x)

    final_output = x

    grad_model = Model(inputs=new_input, outputs=[conv_out, final_output])
    return grad_model, conv_layer_name

def compute_gradcam(img_array, grad_model, pred_class_idx=None):
    """
    Hitung Grad-CAM heatmap.
    
    Args:
        img_array: (1, H, W, 3) numpy array, pixel 0-255
        grad_model: model dengan output [conv_out, logit]
        pred_class_idx: 0 = anorganik, 1 = organik, None = pakai prediksi
    
    Returns:
        heatmap: numpy array (H, W) sudah dinormalisasi 0-1
        pred_class: kelas yang diprediksi
        confidence: confidence score
    """
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor)

        # Hitung score & class
        prob = tf.sigmoid(predictions[0, 0]).numpy()
        if pred_class_idx is None:
            pred_class_idx = 1 if prob > 0.5 else 0

        confidence = prob if pred_class_idx == 1 else (1 - prob)

        # Score untuk backprop
        # Untuk kelas 1 (organik): gunakan logit langsung
        # Untuk kelas 0 (anorganik): negate logit agar gradien mengarah ke kelas ini
        if pred_class_idx == 1:
            score = predictions[0, 0]
        else:
            score = -predictions[0, 0]

    # Gradien terhadap output conv layer
    grads = tape.gradient(score, conv_outputs)

    # Global Average Pooling gradien → bobot tiap channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]  # Hilangkan batch dim
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # ReLU: hanya ambil aktivasi positif
    heatmap = tf.nn.relu(heatmap)

    # Normalisasi dengan epsilon agar tidak divide by zero
    heatmap = heatmap.numpy()
    eps = 1e-8
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + eps)

    return heatmap, pred_class_idx, confidence

def overlay_heatmap(img_path, heatmap, alpha=0.45, colormap='jet'):
    """
    Overlay heatmap ke gambar asli dengan smoothing untuk hasil yang tajam.
    
    Perbedaan dari kode lama: 
    - Pakai scipy zoom (bicubic) bukan PIL resize → lebih smooth
    - Gaussian smoothing tambahan agar tidak kotak-kotak
    """
    from scipy.ndimage import zoom, gaussian_filter

    # Load gambar asli
    img = img_to_array(load_img(img_path, target_size=IMG_SIZE))

    # Upscale heatmap ke ukuran gambar pakai bicubic interpolation
    zoom_factor = (IMG_SIZE[0] / heatmap.shape[0], IMG_SIZE[1] / heatmap.shape[1])
    heatmap_resized = zoom(heatmap, zoom_factor, order=3)  # order=3 = bicubic

    # Gaussian smoothing untuk menghilangkan efek "kotak-kotak"
    heatmap_smooth = gaussian_filter(heatmap_resized, sigma=8)

    # Normalisasi ulang setelah smoothing
    eps = 1e-8
    heatmap_smooth = (heatmap_smooth - heatmap_smooth.min()) / (heatmap_smooth.max() - heatmap_smooth.min() + eps)

    # Terapkan colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_smooth)[:, :, :3]  # Ambil RGB, buang alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Overlay
    img_normalized = img / 255.0
    heatmap_normalized = heatmap_colored / 255.0
    superimposed = (1 - alpha) * img_normalized + alpha * heatmap_normalized
    superimposed = np.clip(superimposed, 0, 1)

    return superimposed, heatmap_smooth

# ================= MAIN VISUALIZATION =================
def visualize_gradcam_batch(model_path, test_dir, n_per_class=3,
                             wrapper_name="MobileNetV3Large",
                             conv_layer=None):
    """
    Visualisasi Grad-CAM untuk n gambar per kelas.
    Menampilkan: Gambar Asli | Heatmap Saja | Overlay
    """
    if not os.path.exists(model_path):
        print(f"❌ Model '{model_path}' tidak ditemukan!")
        return

    print("⏳ Loading model...")
    model = load_model(model_path, compile=False)

    # Debug: tampilkan layer MobileNetV3
    list_inner_layers(model, wrapper_name)

    # Pilih conv layer terbaik
    if conv_layer is None:
        mobilenet = model.get_layer(wrapper_name)
        conv_layer = find_best_conv_layer(mobilenet)
        print(f"\n🎯 Conv layer yang dipakai: '{conv_layer}'")

    # Build Grad-CAM model
    print("⏳ Building Grad-CAM model...")
    try:
        grad_model, conv_layer_used = build_gradcam_model(model, conv_layer, wrapper_name)
        print(f"✅ Grad-CAM model siap (layer: '{conv_layer_used}')")
    except Exception as e:
        print(f"❌ Gagal build Grad-CAM model: {e}")
        return

    # Kumpulkan gambar sample
    classes = ["anorganik", "organik"]
    all_images = []
    for cls in classes:
        cls_path = os.path.join(test_dir, cls)
        if not os.path.exists(cls_path):
            print(f"⚠️  Folder '{cls_path}' tidak ditemukan, skip.")
            continue
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        sampled = random.sample(files, min(n_per_class, len(files)))
        for f in sampled:
            all_images.append((os.path.join(cls_path, f), cls))

    if not all_images:
        print("❌ Tidak ada gambar ditemukan.")
        return

    # Plot: setiap baris = 1 gambar, kolom = [asli, heatmap, overlay]
    n = len(all_images)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle('Grad-CAM Visualization — MobileNetV3', fontsize=16, fontweight='bold', y=1.01)

    for row, (img_path, true_label) in enumerate(all_images):
        try:
            # Prep input
            img_array = img_to_array(load_img(img_path, target_size=IMG_SIZE))
            img_input = np.expand_dims(img_array, axis=0)

            # Hitung Grad-CAM
            heatmap, pred_idx, confidence = compute_gradcam(img_input, grad_model)

            pred_label = CLASS_NAMES[pred_idx]
            is_correct = pred_label.lower() == true_label.lower()
            status = "✅" if is_correct else "❌"

            # Overlay
            overlay, heatmap_smooth = overlay_heatmap(img_path, heatmap)

            # Plot kolom 1: Gambar asli
            axes[row][0].imshow(img_array.astype(np.uint8))
            axes[row][0].set_title(f"Asli: {true_label.upper()}", fontsize=11)
            axes[row][0].axis('off')

            # Plot kolom 2: Heatmap saja
            axes[row][1].imshow(heatmap_smooth, cmap='jet', vmin=0, vmax=1)
            axes[row][1].set_title(f"Heatmap (raw)", fontsize=11)
            axes[row][1].axis('off')
            plt.colorbar(axes[row][1].get_images()[0], ax=axes[row][1], fraction=0.046)

            # Plot kolom 3: Overlay
            axes[row][2].imshow(overlay)
            axes[row][2].set_title(
                f"{status} Pred: {pred_label} ({confidence*100:.1f}%)",
                fontsize=11,
                color='green' if is_correct else 'red'
            )
            axes[row][2].axis('off')

        except Exception as e:
            print(f"❌ Error pada {img_path}: {e}")
            import traceback; traceback.print_exc()

    plt.tight_layout()
    plt.savefig("gradcam_output.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ Selesai! Output disimpan di 'gradcam_output.png'")


# ================= JALANKAN =================
visualize_gradcam_batch(
    model_path=MODEL_FILENAME,
    test_dir=TEST_DIR,
    n_per_class=3,           # Berapa gambar per kelas yang ditampilkan
    wrapper_name="MobileNetV3Large",
    conv_layer=None          # None = otomatis cari layer terbaik
)