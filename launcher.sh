#!/bin/bash
echo "Menunggu sistem siap..."
sleep 5

# Masuk ke folder project
cd /home/willy/

# Activate Virtual Environment
echo "Mengaktifkan Virtual Environment..."
source myenv/bin/activate

# Jalankan Program Utama
echo "Menjalankan Core V2..."
python core_v2.py

# Tahan terminal biar tidak langsung nutup kalau error (untuk debugging)
echo "Program selesai. Tekan Enter untuk keluar."
read
