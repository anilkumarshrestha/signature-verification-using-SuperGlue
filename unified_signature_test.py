"""
Unified İmza Analiz Test Script'i
Hem K-means ön işleme hem de rotation analizi ile tek dosya testi
"""

import cv2
import os
from signature_analysis import analyze_signatures_with_rotation, create_visualization, add_text_overlay, load_superglue_model

def test_single_signature_pair(image1_path, image2_path, output_dir="unified_test_results"):
    """
    İki imza dosyasını unified sistemle test et
    """
    print("🎯 Unified İmza Analiz Sistemi Test Ediliyor...")
    print("=" * 50)
    
    # Görüntüleri yükle
    print(f"📁 İlk görüntü: {os.path.basename(image1_path)}")
    print(f"📁 İkinci görüntü: {os.path.basename(image2_path)}")
    
    im1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    if im1 is None or im2 is None:
        print("❌ HATA: Görüntü dosyaları yüklenemedi!")
        return None
    
    # Boyut uyumluluğu kontrolü
    if im1.shape != im2.shape:
        print(f"📏 Boyut uyumsuzluğu tespit edildi. İkinci görüntü yeniden boyutlandırılıyor...")
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
    
    print(f"✅ Görüntü boyutları: {im1.shape}")
    
    # Model yükle
    print("🤖 SuperGlue modeli yükleniyor...")
    matching, device = load_superglue_model()
    print(f"✅ Model hazır! Cihaz: {device.upper()}")
    
    # Output dizini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🔄 Analiz başlıyor...")
    print("   1. K-means ön işleme")
    print("   2. 8 açıda rotasyon analizi")
    print("   3. Güvenlik skoru hesaplama")
    print("   4. Dinamik eşik belirleme")
    
    # Unified analiz çalıştır
    result = analyze_signatures_with_rotation(
        im1, im2, matching, device,
        base_threshold=0.25,
        rotation_threshold=0.45,
        rotation_improvement_threshold=0.08,
        use_rotation=True,
        use_preprocessing=True,
        preprocessing_method='kmeans'
    )
    
    print(f"\n📊 ANALIZ SONUÇLARI:")
    print("=" * 30)
    print(f"🎯 Eşleşme Oranı: {result['ratio']*100:.1f}%")
    print(f"📈 Eşleşme Sayısı: {result['valid_matches']}/{result['total_keypoints']}")
    print(f"🔄 Rotasyon Açısı: {result['rotation_angle']}°")
    print(f"🔄 Rotasyon Kullanıldı: {'✅ Evet' if result['rotation_used'] else '❌ Hayır'}")
    print(f"🧹 K-means Ön İşleme: {'✅ Uygulandı' if result['preprocessing_used'] else '❌ Kullanılmadı'}")
    print(f"🛡️ Güvenlik Riski: {result['security_analysis']['risk_level']}")
    print(f"📊 Güvenlik Skoru: {result['security_analysis']['security_score']:.3f}")
    print(f"⚖️ Dinamik Eşik: {result['threshold']*100:.1f}%")
    print(f"⏱️ İşlem Süresi: {result['processing_time']:.2f} saniye")
    print(f"\n🎯 KARAR: {'✅ EŞLEŞME' if result['predicted_same'] else '❌ EŞLEŞMEME'}")
    
    # Görselleştirme oluştur
    print(f"\n🎨 Görselleştirme oluşturuluyor...")
    vis = create_visualization(result)
    vis_with_text = add_text_overlay(vis, result)
    
    # Dosya adları oluştur
    name1 = os.path.splitext(os.path.basename(image1_path))[0]
    name2 = os.path.splitext(os.path.basename(image2_path))[0]
    
    # Ana karşılaştırma görselini kaydet
    main_output = os.path.join(output_dir, f"{name1}_vs_{name2}_unified_analysis.png")
    cv2.imwrite(main_output, vis_with_text)
    print(f"💾 Ana analiz kaydedildi: {main_output}")
    
    # Ön işleme karşılaştırması
    preprocessing_comparison = create_preprocessing_comparison(
        result['original_image1'], result['final_image1'],
        result['original_image2'], result['final_image2']
    )
    preprocess_output = os.path.join(output_dir, f"{name1}_vs_{name2}_preprocessing_comparison.png")
    cv2.imwrite(preprocess_output, preprocessing_comparison)
    print(f"💾 Ön işleme karşılaştırması: {preprocess_output}")
    
    # Rotasyon analizi detayları
    if len(result['all_results']) > 1:
        print(f"\n📊 TÜM ROTASYON SONUÇLARI:")
        print("-" * 40)
        for res in result['all_results']:
            indicator = "🏆" if res['angle'] == result['rotation_angle'] else "  "
            print(f"{indicator} {res['angle']:3d}°: {res['ratio']*100:5.1f}% ({res['valid']}/{res['total']})")
    
    return {
        'result': result,
        'main_output': main_output,
        'preprocess_output': preprocess_output
    }

def create_preprocessing_comparison(orig1, proc1, orig2, proc2):
    """
    Ön işleme öncesi/sonrası karşılaştırma görselleştirmesi
    """
    # Üst satır: Orijinaller
    top_row = cv2.hstack([orig1, orig2])
    
    # Alt satır: İşlenmişler
    bottom_row = cv2.hstack([proc1, proc2])
    
    # Birleştir
    comparison = cv2.vstack([top_row, bottom_row])
    
    # BGR'ye çevir ve metin ekle
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Başlıklar
    h, w = orig1.shape
    cv2.putText(comparison_bgr, 'ORIGINAL 1', (10, 30), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(comparison_bgr, 'ORIGINAL 2', (w + 10, 30), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(comparison_bgr, 'K-MEANS CLEANED 1', (10, h + 30), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(comparison_bgr, 'K-MEANS CLEANED 2', (w + 10, h + 30), font, font_scale, (0, 255, 0), thickness)
    
    return comparison_bgr

if __name__ == "__main__":
    print("🎯 Unified İmza Analiz Test Aracı")
    print("=" * 40)
    print("Bu araç hem K-means ön işleme hem de rotation analizini")
    print("birleştirerek en gelişmiş imza karşılaştırmasını yapar.")
    print("")
    
    # Kullanıcıdan dosya yolları al
    print("📁 İlk imza dosyasının yolunu girin:")
    image1_path = input("   Dosya 1: ").strip().replace('"', '')
    
    print("📁 İkinci imza dosyasının yolunu girin:")
    image2_path = input("   Dosya 2: ").strip().replace('"', '')
    
    # Dosya kontrolü
    if not os.path.exists(image1_path):
        print(f"❌ HATA: İlk dosya bulunamadı: {image1_path}")
        exit(1)
    
    if not os.path.exists(image2_path):
        print(f"❌ HATA: İkinci dosya bulunamadı: {image2_path}")
        exit(1)
    
    # Test çalıştır
    result = test_single_signature_pair(image1_path, image2_path)
    
    if result:
        print(f"\n🎉 Test tamamlandı!")
        print(f"📁 Sonuçlar: unified_test_results/ klasöründe")
        print(f"📊 Ana analiz: {os.path.basename(result['main_output'])}")
        print(f"🧹 Ön işleme: {os.path.basename(result['preprocess_output'])}")
    else:
        print(f"❌ Test başarısız!")
