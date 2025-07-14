@echo off
color 0A
title İmza Eşleştirme - TAMAMEN OTOMATİK PAYLAŞIM
cls

echo.
echo     ████████████████████████████████████████████████████████
echo     █                                                      █
echo     █         🔥 OTOMATİK PAYLAŞIM SİSTEMİ                 █
echo     █                                                      █
echo     ████████████████████████████████████████████████████████
echo.
echo     🚀 Sistem tamamen otomatik başlıyor...
echo.

echo     1️⃣ Streamlit localhost'ta başlatılıyor...
start "Streamlit App" cmd /k "python -m streamlit run signature_matching_app.py --server.address 127.0.0.1 --server.port 8501"

echo     ⏳ Streamlit yükleniyor (10 saniye bekle)...
timeout /t 10 >nul

echo     2️⃣ Ngrok tüneli açılıyor...
echo.
echo     📡 PUBLIC URL oluşturuluyor...
echo.
echo     🌐 Link hazır olduğunda arkadaşlarına gönder!
echo.

ngrok http localhost:8501

echo.
echo     ❌ Sistem kapatıldı!
pause
