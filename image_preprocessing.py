"""
Imza görüntülerini ön işleme modülü
Noktalı kağıt arka planını temizler ve imza kalitesini artırır
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

def remove_dotted_background(image, debug=False):
    """
    Noktalı kağıt arka planını kaldırır ve imzayı öne çıkarır
    
    Args:
        image: Gri seviye görüntü
        debug: Debug modunda ara adımları gösterir
    
    Returns:
        Temizlenmiş görüntü
    """
    # Orijinal görüntüyü kopyala
    original = image.copy()
    
    # 1. Gaussian Blur ile küçük noktaları yumuşat
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    
    # 2. Morphological opening ile küçük noktaları kaldır
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel_small)
    
    # 3. Adaptive thresholding ile imza bölgelerini belirle
    adaptive_thresh = cv2.adaptiveThreshold(
        opened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 4. Büyük yapıları koruyarak küçük gürültüyü kaldır
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_large)
    
    # 5. Median filter ile son gürültü temizliği
    final_cleaned = cv2.medianBlur(cleaned, 3)
    
    if debug:
        return {
            'original': original,
            'blurred': blurred,
            'opened': opened,
            'adaptive_thresh': adaptive_thresh,
            'cleaned': cleaned,
            'final': final_cleaned
        }
    
    return final_cleaned

def enhance_signature_contrast(image):
    """
    İmza kontrastını artırır ve arka planı beyazlatır
    """
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) uygula
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    
    # Histogram normalizasyonu
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized

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
        method: 'kmeans' (önerilen), 'adaptive', 'combined'
    
    Returns:
        Temizlenmiş gri seviye görüntü
    """
    # BGR ise gri seviyeye çevir
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == 'kmeans':
        # K-means tabanlı arka plan kaldırma (EN ETKİLİ YÖNTEM)
        processed = remove_background_noise_kmeans(gray)
        
    elif method == 'adaptive':
        # Adaptive thresholding tabanlı temizlik
        processed = remove_dotted_background(gray)
        
    elif method == 'combined':
        # Kombine yaklaşım
        # Önce kontrast artırma
        enhanced = enhance_signature_contrast(gray)
        # Sonra adaptive temizlik
        processed = remove_dotted_background(enhanced)
        
    else:
        raise ValueError("Method must be 'kmeans', 'adaptive', or 'combined'")
    
    return processed

def visualize_preprocessing_steps(image, save_path=None):
    """
    Ön işleme adımlarını görselleştirir
    """
    # Gri seviyeye çevir
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Farklı yöntemleri uygula
    steps = remove_dotted_background(gray, debug=True)
    enhanced = enhance_signature_contrast(gray)
    kmeans_result = remove_background_noise_kmeans(gray)
    combined_result = advanced_signature_preprocessing(gray, method='combined')
    
    # Görselleştirme için birleştir
    row1 = np.hstack([steps['original'], steps['blurred'], steps['opened']])
    row2 = np.hstack([enhanced, kmeans_result, combined_result])
    
    # Başlıklar ekle
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(row1, 'Original', (10, 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(row1, 'Blurred', (gray.shape[1] + 10, 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(row1, 'Opened', (2*gray.shape[1] + 10, 30), font, 0.7, (0, 0, 0), 2)
    
    cv2.putText(row2, 'Enhanced', (10, 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(row2, 'K-means', (gray.shape[1] + 10, 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(row2, 'Combined', (2*gray.shape[1] + 10, 30), font, 0.7, (0, 0, 0), 2)
    
    final_viz = np.vstack([row1, row2])
    
    if save_path:
        cv2.imwrite(save_path, final_viz)
    
    return final_viz

# Test fonksiyonu
def test_preprocessing(image_path, output_dir="preprocessing_test"):
    """
    Ön işleme fonksiyonlarını test eder
    """
    import os
    
    # Görüntüyü yükle
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Görüntü yüklenemedi: {image_path}")
        return
    
    # Output dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Farklı yöntemleri test et
    methods = ['adaptive', 'kmeans', 'combined']
    
    for method in methods:
        processed = advanced_signature_preprocessing(image, method=method)
        output_path = os.path.join(output_dir, f"processed_{method}.png")
        cv2.imwrite(output_path, processed)
        print(f"Kaydedildi: {output_path}")
    
    # Adım adım görselleştirme
    viz_path = os.path.join(output_dir, "preprocessing_steps.png")
    visualize_preprocessing_steps(image, viz_path)
    print(f"Görselleştirme kaydedildi: {viz_path}")

if __name__ == "__main__":
    # Test için örnek kullanım
    print("İmza ön işleme modülü hazır!")
    print("Kullanım:")
    print("from image_preprocessing import advanced_signature_preprocessing")
    print("processed_image = advanced_signature_preprocessing(image, method='combined')")
