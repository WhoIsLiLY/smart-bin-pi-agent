import time
import cv2
import numpy as np
import csv
import os
import json
from datetime import datetime
from gpiozero import DistanceSensor, AngularServo
from gpiozero.pins.lgpio import LGPIOFactory
import tensorflow as tf

# === PICAMERA2 untuk Pi 5 ===
try:
    from picamera2 import Picamera2
    print("[INIT] Menggunakan Picamera2 (Native Pi 5)")
    USE_PICAMERA2 = True
except ImportError:
    print("[WARNING] Picamera2 tidak terinstall, fallback ke OpenCV")
    USE_PICAMERA2 = False

# === TFLITE ===
try:
    from tflite_runtime.interpreter import Interpreter
    print("[INIT] Menggunakan tflite-runtime")
except ImportError:
    try:
        import tensorflow.lite as tflite
        Interpreter = tflite.Interpreter
        print("[INIT] Menggunakan full tensorflow")
    except ImportError:
        print("[ERROR] Library TFLite belum terinstall!")
        exit()

# ==========================================
# 1. KONFIGURASI
# ==========================================
factory      = LGPIOFactory()
sensor       = DistanceSensor(echo=24, trigger=23, max_distance=2.0, pin_factory=factory)
servo        = AngularServo(18, min_angle=-90, max_angle=90,
                            min_pulse_width=0.0005, max_pulse_width=0.0025,
                            pin_factory=factory, initial_angle=None)

MODEL_PATH   = "model.tflite"
INPUT_SIZE   = 160
TARGET_TRIALS = 30  # Minimum perulangan sesuai permintaan dosen

# ==========================================
# 2. LOGGER
# ==========================================
class TrialLogger:
    """
    Mencatat setiap percobaan klasifikasi ke:
    - CSV  : untuk diolah di Excel / laporan
    - JSON : untuk summary statistik akhir
    - TXT  : untuk log mentah real-time
    """

    def __init__(self):
        timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir  = f"eval_log_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)

        self.csv_path     = os.path.join(self.log_dir, "trial_log.csv")
        self.summary_path = os.path.join(self.log_dir, "summary.json")
        self.txt_path     = os.path.join(self.log_dir, "realtime.txt")

        # Inisialisasi CSV
        self.csv_fields = [
            "trial_no",         # Nomor percobaan
            "timestamp",        # Waktu percobaan
            "true_label",       # Label BENAR (diisi manual oleh penguji)
            "predicted_label",  # Prediksi model
            "confidence_pct",   # Confidence (%)
            "is_correct",       # Benar / Salah (diisi setelah true_label diketahui)
            "t_detection_ms",   # Waktu dari sensor detect → mulai capture (ms)
            "t_capture_ms",     # Waktu capture frame (ms)
            "t_inference_ms",   # Waktu inferensi model (ms)
            "t_servo_ms",       # Waktu servo bergerak ke posisi target (ms)
            "t_total_ms",       # Total waktu sistem (sensor detect → servo terbuka)
            "jarak_cm",         # Jarak sensor saat deteksi
            "notes",            # Catatan tambahan (opsional)
        ]

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            writer.writeheader()

        self.trials    = []
        self.trial_no  = 0
        print(f"[LOG] Folder log: {self.log_dir}/")
        print(f"[LOG] CSV: {self.csv_path}")

    def log_trial(self, data: dict):
        """Tulis satu baris percobaan ke CSV dan txt."""
        self.trial_no += 1
        data["trial_no"] = self.trial_no
        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Simpan ke list untuk summary akhir
        self.trials.append(data)

        # Tulis ke CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            writer.writerow({k: data.get(k, "") for k in self.csv_fields})

        # Tulis ke TXT log real-time
        line = (
            f"[Trial {self.trial_no:03d}] "
            f"{data['timestamp']} | "
            f"Pred: {data['predicted_label']:10s} ({data['confidence_pct']:.1f}%) | "
            f"Total: {data['t_total_ms']:.0f}ms | "
            f"Jarak: {data['jarak_cm']:.1f}cm"
        )
        with open(self.txt_path, "a") as f:
            f.write(line + "\n")

        print(f"   📝 {line}")
        return self.trial_no

    def save_summary(self):
        """Hitung dan simpan statistik akhir ke JSON."""
        if not self.trials:
            return

        t_inference_list = [t["t_inference_ms"] for t in self.trials if t.get("t_inference_ms")]
        t_total_list     = [t["t_total_ms"]     for t in self.trials if t.get("t_total_ms")]
        t_capture_list   = [t["t_capture_ms"]   for t in self.trials if t.get("t_capture_ms")]

        # Hitung akurasi jika true_label sudah diisi
        labeled = [t for t in self.trials if t.get("true_label") and t.get("is_correct") != ""]
        correct = [t for t in labeled if str(t.get("is_correct")).upper() in ["TRUE", "YA", "1", "BENAR"]]

        summary = {
            "total_trials":           len(self.trials),
            "labeled_trials":         len(labeled),
            "correct_predictions":    len(correct),
            "accuracy_pct":           round(len(correct) / len(labeled) * 100, 2) if labeled else "Belum diisi",

            "inference_time_ms": {
                "min":  round(min(t_inference_list), 2) if t_inference_list else 0,
                "max":  round(max(t_inference_list), 2) if t_inference_list else 0,
                "mean": round(np.mean(t_inference_list), 2) if t_inference_list else 0,
                "std":  round(np.std(t_inference_list), 2) if t_inference_list else 0,
            },
            "total_time_ms": {
                "min":  round(min(t_total_list), 2) if t_total_list else 0,
                "max":  round(max(t_total_list), 2) if t_total_list else 0,
                "mean": round(np.mean(t_total_list), 2) if t_total_list else 0,
                "std":  round(np.std(t_total_list), 2) if t_total_list else 0,
            },
            "capture_time_ms": {
                "min":  round(min(t_capture_list), 2) if t_capture_list else 0,
                "max":  round(max(t_capture_list), 2) if t_capture_list else 0,
                "mean": round(np.mean(t_capture_list), 2) if t_capture_list else 0,
            },

            "class_distribution": {
                "ORGANIK":   sum(1 for t in self.trials if t.get("predicted_label") == "ORGANIK"),
                "ANORGANIK": sum(1 for t in self.trials if t.get("predicted_label") == "ANORGANIK"),
            },

            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self._print_summary(summary)
        return summary

    def _print_summary(self, s):
        print("\n" + "=" * 65)
        print("📊 RINGKASAN EVALUASI SISTEM — BAB 6")
        print("=" * 65)
        print(f"  Total Percobaan         : {s['total_trials']}")
        print(f"  Akurasi (jika diisi)    : {s['accuracy_pct']}%")
        print(f"")
        print(f"  ⏱  Waktu Inferensi (model):")
        print(f"     Min  : {s['inference_time_ms']['min']} ms")
        print(f"     Max  : {s['inference_time_ms']['max']} ms")
        print(f"     Rata : {s['inference_time_ms']['mean']} ms")
        print(f"     Std  : {s['inference_time_ms']['std']} ms")
        print(f"")
        print(f"  ⏱  Total Waktu Sistem (sensor → servo terbuka):")
        print(f"     Min  : {s['total_time_ms']['min']} ms")
        print(f"     Max  : {s['total_time_ms']['max']} ms")
        print(f"     Rata : {s['total_time_ms']['mean']} ms")
        print(f"     Std  : {s['total_time_ms']['std']} ms")
        print(f"")
        print(f"  📦 Distribusi Prediksi:")
        print(f"     ORGANIK   : {s['class_distribution']['ORGANIK']}")
        print(f"     ANORGANIK : {s['class_distribution']['ANORGANIK']}")
        print(f"")
        print(f"  📁 File log disimpan di: {self.log_dir}/")
        print(f"     - trial_log.csv  → import ke Excel")
        print(f"     - summary.json   → statistik lengkap")
        print("=" * 65)

# ==========================================
# 3. KAMERA
# ==========================================
def init_camera_picamera2():
    try:
        picam = Picamera2()
        config = picam.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"})
        picam.configure(config)
        picam.start()
        time.sleep(2)
        frame = picam.capture_array()
        if frame is not None and frame.size > 0:
            print(f"[SUCCESS] Picamera2 siap! Resolusi: {frame.shape}")
            return picam
        picam.close()
        return None
    except Exception as e:
        print(f"[ERROR] Gagal init Picamera2: {e}")
        return None

def init_camera_opencv():
    for index in [0, 1]:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[SUCCESS] OpenCV kamera Index {index} siap!")
                    return cap
                time.sleep(0.1)
            cap.release()
    return None

def init_camera():
    if USE_PICAMERA2:
        cam = init_camera_picamera2()
        if cam:
            return cam, "picamera2"
    cam = init_camera_opencv()
    if cam:
        return cam, "opencv"
    return None, None

def capture_frame_picamera2(picam, max_retries=5):
    for attempt in range(max_retries):
        try:
            frame = picam.capture_array()
            if frame is not None and frame.size > 0:
                return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[RETRY] {attempt+1}/{max_retries}: {e}")
            time.sleep(0.1)
    return False, None

def capture_frame_opencv(cap, max_retries=5):
    for _ in range(3):
        cap.grab()
    for attempt in range(max_retries):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True, frame
        time.sleep(0.1)
    return False, None

# ==========================================
# 4. MODEL
# ==========================================
def load_model():
    print(f"[INIT] Memuat model {MODEL_PATH} menggunakan TensorFlow...")
    try:
        # Menggunakan modul tf.lite bawaan dari full TensorFlow
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        print(f"[SUCCESS] Model berhasil dimuat dengan TF versi: {tf.__version__}")
        return interpreter
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        return None

def predict_image(frame, interp, in_det, out_det):
    """
    Prediksi + catat waktu inferensi.
    Return: label, confidence, t_inference_ms
    """
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    # Frame dari Picamera2 sudah di-convert ke BGR di capture_frame,
    # untuk model kita kembalikan ke RGB
    img_rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32)

    t0 = time.perf_counter()
    interp.set_tensor(in_det[0]['index'], input_data)
    interp.invoke()
    output     = interp.get_tensor(out_det[0]['index'])
    t_infer    = (time.perf_counter() - t0) * 1000  # → ms

    logit       = float(output[0][0])
    probability = 1 / (1 + np.exp(-logit))

    if probability > 0.5:
        return "ORGANIK", probability, t_infer
    else:
        return "ANORGANIK", 1 - probability, t_infer

# ==========================================
# 5. SERVO
# ==========================================
def move_servo(angle):
    t0 = time.perf_counter()
    servo.angle = angle
    time.sleep(0.5)  # Beri waktu servo sampai posisi
    t_servo = (time.perf_counter() - t0) * 1000
    servo.value = None
    return t_servo

# ==========================================
# 6. MAIN LOOP DENGAN LOGGING
# ==========================================
def main():
    cam, cam_type = init_camera()
    if not cam:
        print("[FATAL] Kamera tidak tersedia!")
        return

    interp = load_model()
    if not interp:
        print("[FATAL] Model tidak tersedia!")
        return

    in_det  = interp.get_input_details()
    out_det = interp.get_output_details()

    logger = TrialLogger()

    print(f"[INFO] Kamera : {cam_type}")
    print(f"[INFO] Target : {TARGET_TRIALS} percobaan")
    print(f"\n⚠️  PETUNJUK PENGUJI:")
    print(f"   Setelah tiap prediksi, catat di kolom 'true_label' dan 'is_correct'")
    print(f"   di file: {logger.csv_path}\n")

    # Reset servo ke netral
    move_servo(-5)
    print("[SYSTEM] Sistem siap. Mulai deteksi sampah...\n")

    consecutive_errors = 0
    MAX_ERRORS = 3

    try:
        while logger.trial_no < TARGET_TRIALS:
            remaining = TARGET_TRIALS - logger.trial_no
            print(f"[INFO] Sisa percobaan: {remaining}/{TARGET_TRIALS} — Menunggu objek...")

            # --- Tunggu objek masuk jangkauan sensor ---
            # Timer BELUM dimulai — idle tidak ikut terhitung

            while True:
                jarak = sensor.distance * 100
                if 2 < jarak < 15:
                    break
                time.sleep(0.05)


            # Timer dimulai TEPAT saat sensor mendeteksi objek
            t_system_start = time.perf_counter()
            t_detection    = 0  # Titik t=0, referensi awal
            print(f"\n[DETECT] Objek terdeteksi di {jarak:.1f} cm")

            # Jeda singkat sebelum capture (stabilisasi)
            time.sleep(1.0)

            # --- Capture Frame ---
            t_cap_start = time.perf_counter()
            if cam_type == "picamera2":
                ret, frame = capture_frame_picamera2(cam)
            else:
                ret, frame = capture_frame_opencv(cam)
            t_capture = (time.perf_counter() - t_cap_start) * 1000

            if not ret:
                consecutive_errors += 1
                print(f"[ERROR] Capture gagal ({consecutive_errors}/{MAX_ERRORS})")
                if consecutive_errors >= MAX_ERRORS:
                    print("[FATAL] Terlalu banyak error kamera, berhenti.")
                    break
                continue

            consecutive_errors = 0

            # --- Inferensi ---
            try:
                label, conf, t_infer = predict_image(frame, interp, in_det, out_det)
            except Exception as e:
                print(f"[ERROR] Prediksi gagal: {e}")
                continue

            print(f"[RESULT] {label} ({conf*100:.1f}%) | Inferensi: {t_infer:.1f}ms")

            # --- Gerak Servo ---
            if label == "ORGANIK":
                t_servo = move_servo(-80)
            else:
                t_servo = move_servo(80)

            # Total waktu sistem: dari sensor deteksi → servo terbuka
            t_total = (time.perf_counter() - t_system_start) * 1000

            # --- Log Percobaan ---
            trial_no = logger.log_trial({
                "true_label":       "",          # ← ISI MANUAL setelah percobaan
                "predicted_label":  label,
                "confidence_pct":   round(conf * 100, 2),
                "is_correct":       "",          # ← ISI MANUAL: YA / TIDAK
                "t_detection_ms":   round(t_detection, 2),
                "t_capture_ms":     round(t_capture, 2),
                "t_inference_ms":   round(t_infer, 2),
                "t_servo_ms":       round(t_servo, 2),
                "t_total_ms":       round(t_total, 2),
                "jarak_cm":         round(jarak, 1),
                "notes":            "",
            })

            print(f"   ⏱  Breakdown: deteksi={t_detection:.0f}ms | "
                  f"capture={t_capture:.0f}ms | "
                  f"inferensi={t_infer:.0f}ms | "
                  f"servo={t_servo:.0f}ms | "
                  f"TOTAL={t_total:.0f}ms")

            # Tunggu sebelum servo kembali
            time.sleep(3)
            print("[SERVO] Kembali ke netral...")
            move_servo(-5)

            # Jeda antar percobaan (agar penguji siap memasukkan sampah berikutnya)
            time.sleep(2)

            if logger.trial_no >= TARGET_TRIALS:
                print(f"\n✅ Target {TARGET_TRIALS} percobaan tercapai!")
                break

    except KeyboardInterrupt:
        print("\n[STOP] Dihentikan manual.")

    finally:
        # Simpan summary
        logger.save_summary()

        # Cleanup hardware
        if cam_type == "picamera2":
            cam.close()
        else:
            cam.release()
        servo.value = None
        servo.close()
        print("[CLEANUP] Hardware dilepas dengan aman.")
        print(f"\n📁 Semua log tersimpan di folder: {logger.log_dir}/")
        print(f"   Import 'trial_log.csv' ke Excel untuk tabel Bab 6.")

if __name__ == "__main__":
    main()