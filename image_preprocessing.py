"""
İmza görüntülerini ön işleme modülü - K-means Odaklı
Noktalı kağıt arka planını K-means clustering ile temizler
"""

import cv2
import numpy as np

def remove_background_noise_kmeans(image, n_clusters=3):
    """
    K-means clustering ile arka plan gürültüsünü kaldırır
    NOKTALARI KAĞIT İÇİN EN ETKİLİ YÖNTEM!
    
    Args:
        image: Gri seviye görüntü
        n_clusters: Küme sayısı (3 = arka plan + imza + geçiş)
    
    Returns:
        Temizlenmiş binary görüntü (siyah imza, beyaz arka plan)
    """
    # Görüntüyü reshape et
    data = image.reshape((-1, 1))
    data = np.float32(data)
    
    # K-means clustering - daha stabil sonuç için parametreler optimize edildi
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(data, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Centers'ı sırala (en koyu = imza, en açık = arka plan)
    centers = centers.flatten()
    sorted_indices = np.argsort(centers)
    
    # En koyu cluster'ı imza olarak kabul et
    signature_label = sorted_indices[0]  # En düşük intensite = en koyu = imza
    
    # İmzayı siyah (0), arka planı beyaz (255) yap
    result = np.where(labels.reshape(image.shape) == signature_label, 0, 255).astype(np.uint8)
    
    # Küçük gürültüleri temizle
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    
    return result

def advanced_signature_preprocessing(image, method='kmeans'):
    """
    Gelişmiş imza ön işleme ana fonksiyonu
    K-means yöntemi varsayılan olarak kullanılır (en etkili sonuç)
    
    Args:
        image: Giriş görüntüsü (BGR veya gri seviye)
        method: 'kmeans' (önerilen ve ana yöntem)
    
    Returns:
        Temizlenmiş gri seviye görüntü
    """
    # BGR ise gri seviyeye çevir
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == 'kmeans':
        # K-means tabanlı arka plan kaldırma (ANA YÖNTEM)
        processed = remove_background_noise_kmeans(gray)
    else:
        # Geçmişte diğer yöntemler vardı, şimdi sadece K-means kullanılıyor
        print(f"Uyarı: '{method}' yöntemi desteklenmiyor. K-means kullanılıyor.")
        processed = remove_background_noise_kmeans(gray)
    
    return processed

if __name__ == "__main__":
    # Test için örnek kullanım
    print("🎯 K-means Odaklı İmza Ön İşleme Modülü Hazır!")
    print("Kullanım:")
    print("from image_preprocessing import advanced_signature_preprocessing")
    print("processed_image = advanced_signature_preprocessing(image, method='kmeans')")
    print("\n✨ Noktalı kağıt arka planları otomatik olarak temizlenir!")
