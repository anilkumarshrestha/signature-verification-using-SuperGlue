"""
K-means ön işleme hızlı test script'i
Noktalı kağıt problemini K-means ile çözer
"""

import cv2
import os
import numpy as np
from image_preprocessing import remove_background_noise_kmeans

def quick_kmeans_test(image_path, show_result=True):
    """
    Tek bir görüntü üzerinde hızlı K-means testi
    """
    # Görüntüyü yükle
    print(f"Test edilen: {os.path.basename(image_path)}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"HATA: Görüntü yüklenemedi!")
        return None
    
    # Gri seviyeye çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    print(f"Orijinal boyut: {gray.shape}")
    print(f"Pixel değer aralığı: {gray.min()}-{gray.max()}")
    
    # K-means uygula
    print("K-means uygulanıyor...")
    cleaned = remove_background_noise_kmeans(gray)
    
    print(f"Temizlenmiş pixel değerleri: {np.unique(cleaned)}")
    
    # Yan yana karşılaştırma
    comparison = np.hstack([gray, cleaned])
    
    # Başlık ekle
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'ORIGINAL', (10, 30), font, 1, (128), 2)
    cv2.putText(comparison, 'K-MEANS CLEANED', (gray.shape[1] + 10, 30), font, 1, (128), 2)
    
    # Kaydet
    output_dir = "kmeans_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_kmeans_comparison.png")
    
    cv2.imwrite(output_path, comparison)
    print(f"✅ Sonuç kaydedildi: {output_path}")
    
    if show_result:
        # Küçük boyutta göster (isteğe bağlı)
        h, w = comparison.shape
        if h > 600 or w > 1200:
            scale = min(600/h, 1200/w)
            new_h, new_w = int(h*scale), int(w*scale)
            comparison_small = cv2.resize(comparison, (new_w, new_h))
        else:
            comparison_small = comparison
            
        cv2.imshow('K-means Sonuç (ESC ile kapat)', comparison_small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return {
        'original': gray,
        'cleaned': cleaned,
        'comparison_path': output_path
    }

def batch_kmeans_test(folder_path):
    """
    Bir klasördeki tüm görüntüleri K-means ile test et
    """
    if not os.path.exists(folder_path):
        print(f"HATA: Klasör bulunamadı: {folder_path}")
        return
    
    # Desteklenen uzantılar
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    # Dosyaları bul
    files = []
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in extensions):
            files.append(file)
    
    if not files:
        print("Görüntü dosyası bulunamadı!")
        return
    
    print(f"Bulunan dosya sayısı: {len(files)}")
    print("=" * 50)
    
    results = []
    
    for i, filename in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] İşleniyor: {filename}")
        
        image_path = os.path.join(folder_path, filename)
        result = quick_kmeans_test(image_path, show_result=False)
        
        if result:
            results.append({
                'filename': filename,
                'path': result['comparison_path']
            })
            print(f"✅ Başarılı")
        else:
            print(f"❌ Başarısız")
    
    print(f"\n" + "=" * 50)
    print(f"Toplu test tamamlandı!")
    print(f"Başarılı: {len(results)}/{len(files)}")
    print(f"Sonuçlar: kmeans_test_results/ klasöründe")
    
    return results

if __name__ == "__main__":
    print("🎯 K-means Noktalı Kağıt Temizleme Testi")
    print("=" * 40)
    
    while True:
        print("\nSeçenekler:")
        print("1. Tek dosya test et")
        print("2. Klasör test et") 
        print("3. Çıkış")
        
        choice = input("Seçiminiz (1-3): ").strip()
        
        if choice == "1":
            path = input("Görüntü dosyası yolu: ").strip().replace('"', '')
            if os.path.isfile(path):
                quick_kmeans_test(path)
            else:
                print("❌ Dosya bulunamadı!")
                
        elif choice == "2":
            path = input("Klasör yolu: ").strip().replace('"', '')
            if os.path.isdir(path):
                batch_kmeans_test(path)
            else:
                print("❌ Klasör bulunamadı!")
                
        elif choice == "3":
            print("👋 Görüşürüz!")
            break
            
        else:
            print("❌ Geçersiz seçim!")
