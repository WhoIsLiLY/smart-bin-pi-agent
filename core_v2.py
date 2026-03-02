import time
import cv2
import numpy as np
from gpiozero import DistanceSensor, AngularServo
from gpiozero.pins.lgpio import LGPIOFactory
import tensorflow as tf

# === PICAMERA2 untuk Pi 5 (Wajib!) ===
try:
    from picamera2 import Picamera2
    print("[INIT] Menggunakan Picamera2 (Native Pi 5)")
    USE_PICAMERA2 = True
except ImportError:
    print("[WARNING] Picamera2 tidak terinstall, fallback ke OpenCV")
    print("Install dengan: sudo apt install -y python3-picamera2")
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
# 1. SETUP HARDWARE
# ==========================================
print("[INIT] Menghubungkan ke Hardware GPIO...")
factory = LGPIOFactory()

sensor = DistanceSensor(echo=24, trigger=23, max_distance=2.0, pin_factory=factory)
servo = AngularServo(18, min_angle=-90, max_angle=90, 
                     min_pulse_width=0.0005, max_pulse_width=0.0025, 
                     pin_factory=factory, initial_angle=None)

MODEL_PATH = "model.tflite"
INPUT_SIZE = 160

# ==========================================
# 2. FUNGSI KAMERA (PICAMERA2 + OPENCV)
# ==========================================
def init_camera_picamera2():
    """Inisialisasi kamera dengan Picamera2 (Pi 5 recommended)"""
    print("[INIT] Menginisialisasi Picamera2...")
    try:
        picam = Picamera2()
        
        # Konfigurasi resolusi
        config = picam.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam.configure(config)
        
        # Start kamera
        picam.start()
        print("[INIT] Warming up kamera...")
        time.sleep(2)  # Beri waktu auto-exposure/white balance
        
        # Test capture
        frame = picam.capture_array()
        if frame is not None and frame.size > 0:
            print(f"[SUCCESS] Picamera2 siap! Resolusi: {frame.shape}")
            return picam
        else:
            print("[ERROR] Frame Picamera2 kosong")
            picam.close()
            return None
            
    except Exception as e:
        print(f"[ERROR] Gagal init Picamera2: {e}")
        return None

def init_camera_opencv():
    """Fallback ke OpenCV (untuk webcam USB atau Pi model lama)"""
    print("[INIT] Mencoba OpenCV...")
    
    for index in [0, 1]:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test read
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[SUCCESS] OpenCV kamera Index {index} siap!")
                    return cap
                time.sleep(0.1)
            cap.release()
    
    print("[ERROR] OpenCV tidak bisa akses kamera")
    return None

def init_camera():
    """Universal camera init (auto-detect Pi 5 vs lainnya)"""
    if USE_PICAMERA2:
        cam = init_camera_picamera2()
        if cam:
            return cam, "picamera2"
    
    cam = init_camera_opencv()
    if cam:
        return cam, "opencv"
    
    print("\n=== TROUBLESHOOTING ===")
    print("Raspberry Pi 5 memerlukan Picamera2:")
    print("  sudo apt update")
    print("  sudo apt install -y python3-picamera2")
    print("\nUntuk webcam USB, pastikan tersambung dengan benar.")
    return None, None

# ==========================================
# 3. LOAD MODEL
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

interpreter = load_model()
if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

def predict_image(frame):
    """Prediksi gambar"""
    try:
        # Picamera2 sudah RGB, OpenCV BGR -> perlu convert
        # Kita asumsikan frame sudah dalam format yang benar
        img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        
        # Pastikan RGB format
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Jika dari OpenCV (BGR), convert ke RGB
            # Tapi Picamera2 sudah RGB, jadi kita cek dulu
            pass  # Kita skip convert karena bisa dari Picamera2
        
        input_data = np.expand_dims(img, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        score_logits = output_data[0][0]
        probability = 1 / (1 + np.exp(-score_logits))
        
        if probability > 0.5:
            return "ORGANIK", probability
        else:
            return "ANORGANIK", 1 - probability
    except Exception as e:
        print(f"[ERROR] Prediksi gagal: {e}")
        return None, 0.0

# ==========================================
# 4. SERVO CONTROL
# ==========================================
def move_servo(angle):
    print(f"[SERVO] Gerak ke {angle} derajat...")
    servo.angle = angle
    time.sleep(0.5)
    servo.value = None

# ==========================================
# 5. CAPTURE FUNCTIONS
# ==========================================
def capture_frame_picamera2(picam, max_retries=5):
    """Capture dari Picamera2"""
    for attempt in range(max_retries):
        try:
            frame = picam.capture_array()
            if frame is not None and frame.size > 0:
                # Picamera2 return RGB, kita perlu BGR untuk konsistensi OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame_bgr
        except Exception as e:
            print(f"[RETRY] Picamera2 capture {attempt+1}/{max_retries}: {e}")
            time.sleep(0.1)
    return False, None

def capture_frame_opencv(cap, max_retries=5):
    """Capture dari OpenCV"""
    # Clear buffer
    for _ in range(3):
        cap.grab()
    
    for attempt in range(max_retries):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return True, frame
        time.sleep(0.1)
    return False, None

# ==========================================
# 6. MAIN LOOP
# ==========================================
def main():
    # Init kamera
    cam, cam_type = init_camera()
    if not cam:
        print("[FATAL] Kamera tidak tersedia!")
        return
    
    if not interpreter:
        print("[FATAL] Model tidak tersedia!")
        if cam_type == "picamera2":
            cam.close()
        else:
            cam.release()
        return
    
    print(f"[INFO] Menggunakan kamera tipe: {cam_type}")
    
    # Setup servo
    print("[SERVO] Reset ke posisi netral...")
    move_servo(-5)
    print("\n[SYSTEM] Sistem Siap Deteksi!\n")
    
    consecutive_errors = 0
    MAX_ERRORS = 3
    
    try:
        while True:
            jarak = sensor.distance * 100
            
            if 2 < jarak < 25:
                print(f"\n[DETECT] Objek di {jarak:.1f} cm")
                time.sleep(1.0)
                
                # Capture sesuai tipe kamera
                if cam_type == "picamera2":
                    ret, frame = capture_frame_picamera2(cam)
                else:
                    ret, frame = capture_frame_opencv(cam)
                
                if ret:
                    consecutive_errors = 0
                    
                    # Prediksi
                    label, conf = predict_image(frame)
                    
                    if label:
                        print(f"[RESULT] {label} ({conf*100:.1f}%)")
                        
                        # Gerak servo
                        if label == "ORGANIK":
                            move_servo(-80)
                        else:
                            move_servo(80)
                        
                        time.sleep(3)
                        print("[SERVO] Balik Netral")
                        move_servo(-5)
                        time.sleep(2)
                else:
                    consecutive_errors += 1
                    print(f"[ERROR] Frame gagal ({consecutive_errors}/{MAX_ERRORS})")
                    
                    if consecutive_errors >= MAX_ERRORS:
                        print("[FATAL] Terlalu banyak error, restart program!")
                        break
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n[STOP] Program berhenti.")
    finally:
        # Cleanup
        if cam_type == "picamera2":
            cam.close()
        else:
            cam.release()
        servo.value = None
        servo.close()
        print("[CLEANUP] Hardware dilepas dengan aman.")

if __name__ == "__main__":
    main()