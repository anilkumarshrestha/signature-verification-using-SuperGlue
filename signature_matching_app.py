import streamlit as st
import cv2
import torch
import numpy as np
import time
from PIL import Image
import os
import tempfile
from signature_analysis import analyze_signatures_with_rotation, create_visualization, load_superglue_model

# Configure Streamlit page
st.set_page_config(
    page_title="İmza Eşleştirme Sistemi",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 30px 0;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        font-family: 'Poppins', sans-serif;
        animation: pulse 2s ease-in-out infinite alternate;
    }
    
    @keyframes pulse {
        from { box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        to { box-shadow: 0 15px 40px rgba(102,126,234,0.4); }
    }
    
    .result-box {
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
        position: relative;
        overflow: hidden;
    }
    
    .result-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .result-box:hover::before {
        left: 100%;
    }
    
    .match-positive {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border: 3px solid #28a745;
        color: #155724;
        animation: successGlow 1s ease-in-out;
    }
    
    @keyframes successGlow {
        0% { transform: scale(0.95); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .match-negative {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border: 3px solid #dc3545;
        color: #721c24;
        animation: warningGlow 1s ease-in-out;
    }
    
    @keyframes warningGlow {
        0% { transform: scale(0.95); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #ffd700;
        box-shadow: 0 8px 25px rgba(102,126,234,0.3);
        font-family: 'Poppins', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(240,147,251,0.4);
        transition: transform 0.3s ease;
        font-family: 'Poppins', sans-serif;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .upload-zone {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
    }
    
    .upload-zone:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #e0c3fc 0%, #9bb5ff 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(102,126,234,0.6);
    }
    
    .sidebar .stCheckbox {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    .instruction-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #ff6b6b;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(252,182,159,0.3);
        font-family: 'Poppins', sans-serif;
    }
    
    .footer {
        background: linear-gradient(135deg, #434343 0%, #000000 100%);
        color: white;
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        margin-top: 30px;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animated particles background */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .particle {
        position: absolute;
        width: 2px;
        height: 2px;
        background: #667eea;
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 1; }
        50% { transform: translateY(-20px) rotate(180deg); opacity: 0.5; }
    }
</style>

<script>
// Add animated particles
function createParticles() {
    const particleContainer = document.querySelector('.particles');
    if (!particleContainer) {
        const particles = document.createElement('div');
        particles.className = 'particles';
        document.body.appendChild(particles);
        
        for (let i = 0; i < 50; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 6 + 's';
            particles.appendChild(particle);
        }
    }
}
createParticles();
</script>
""", unsafe_allow_html=True)

# Title with amazing gradient effect
st.markdown('''
<div class="main-header">
    <h1 style="margin: 0; font-size: 3em; font-weight: 700;">
        🖋️ İmza Eşleştirme Sistemi
    </h1>
    <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">
        ✨ Yapay Zeka Destekli SuperGlue Teknolojisi ✨
    </p>
</div>
''', unsafe_allow_html=True)

# Initialize model
@st.cache_resource
def load_model():
    """Load the SuperGlue model"""
    return load_superglue_model()

def analyze_signatures(im1, im2, matching, device, use_rotation=True, use_preprocessing=True):
    """Wrapper function for the common signature analysis"""
    return analyze_signatures_with_rotation(
        im1, im2, matching, device,
        base_threshold=0.25,
        rotation_threshold=0.45, 
        rotation_improvement_threshold=0.08,
        use_rotation=use_rotation,
        use_preprocessing=use_preprocessing,
        preprocessing_method='kmeans'
    )

# Sidebar controls with enhanced styling
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h2 style="color: #667eea; font-family: 'Poppins', sans-serif;">⚙️ Kontrol Paneli</h2>
</div>
""", unsafe_allow_html=True)

# Model loading status with style
with st.sidebar:
    with st.spinner("🤖 AI Model yükleniyor..."):
        matching, device = load_model()
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                padding: 15px; 
                border-radius: 10px; 
                text-align: center; 
                margin: 10px 0;">
        <h4 style="margin: 0; color: #2c3e50;">✅ Model Hazır!</h4>
        <p style="margin: 5px 0 0 0; color: #34495e; font-weight: bold;">
            🚀 {device.upper()} Aceleration
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced settings
    st.markdown("""
    <div style="text-align: center; margin: 15px 0;">
        <h4 style="color: #667eea;">🎛️ Analiz Ayarları</h4>
    </div>
    """, unsafe_allow_html=True)
    
    use_rotation = st.checkbox(
        "🔄 Rotasyon analizi kullan", 
        value=True, 
        help="İmzaları farklı açılarda döndürerek analiz eder"
    )
    
    use_preprocessing = st.checkbox(
        "🧹 K-means ön işleme kullan", 
        value=True, 
        help="Noktalı kağıt arka planını K-means ile temizler"
    )
    
    show_all_angles = st.checkbox(
        "📊 Tüm açıları göster", 
        value=True, 
        help="Her rotasyon açısının sonuçlarını gösterir"
    )
    
    st.markdown("---")
    
    # Add performance info
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 15px; 
                border-radius: 10px; 
                text-align: center;">
        <h4 style="margin: 0 0 10px 0;">📈 Sistem Performansı</h4>
        <p style="margin: 5px 0; font-size: 0.9em;">⚡ Hız: ~2-3 saniye (GPU)</p>
        <p style="margin: 5px 0; font-size: 0.9em;">🔄 8 farklı açı analizi</p>
    </div>
    """, unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="upload-zone">
        <h3 style="color: #667eea; margin-bottom: 15px;">📄 İlk İmza</h3>
        <p style="color: #666; margin-bottom: 20px;">Referans imza dosyasını yükleyin</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file1 = st.file_uploader(
        "İlk imza dosyasını yükleyin",
        type=['png', 'jpg', 'jpeg'],
        key="file1",
        label_visibility="collapsed"
    )
    
    if uploaded_file1:
        image1 = Image.open(uploaded_file1)
        st.image(image1, caption="✅ İlk İmza Yüklendi", use_column_width=True)
        st.success("📤 Dosya başarıyla yüklendi!")

with col2:
    st.markdown("""
    <div class="upload-zone">
        <h3 style="color: #667eea; margin-bottom: 15px;">📄 İkinci İmza</h3>
        <p style="color: #666; margin-bottom: 20px;">Karşılaştırılacak imza dosyasını yükleyin</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file2 = st.file_uploader(
        "İkinci imza dosyasını yükleyin",
        type=['png', 'jpg', 'jpeg'],
        key="file2",
        label_visibility="collapsed"
    )
    
    if uploaded_file2:
        image2 = Image.open(uploaded_file2)
        st.image(image2, caption="✅ İkinci İmza Yüklendi", use_column_width=True)
        st.success("📤 Dosya başarıyla yüklendi!")

# Analysis button with enhanced styling
if uploaded_file1 and uploaded_file2:
    st.markdown("""
    <div style="text-align: center; margin: 30px 0;">
        <h3 style="color: #667eea; font-family: 'Poppins', sans-serif;">
            🎯 Her şey hazır! Analizi başlatalım
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button(
            "🔍 İmzaları Analiz Et", 
            type="primary", 
            use_container_width=True,
            help="AI algoritması ile imzaları karşılaştırır"
        )
    
    if analyze_button:
        
        with st.spinner("İmzalar analiz ediliyor..."):
            # Convert PIL to OpenCV format safely
            img1_array = np.array(image1)
            img2_array = np.array(image2)
            
            # Handle different image formats (RGB, RGBA, Grayscale)
            if len(img1_array.shape) == 3:
                if img1_array.shape[2] == 4:  # RGBA
                    img1_cv = cv2.cvtColor(img1_array, cv2.COLOR_RGBA2GRAY)
                elif img1_array.shape[2] == 3:  # RGB
                    img1_cv = cv2.cvtColor(img1_array, cv2.COLOR_RGB2GRAY)
                else:
                    img1_cv = img1_array[:,:,0]  # Take first channel
            else:  # Already grayscale
                img1_cv = img1_array
            
            if len(img2_array.shape) == 3:
                if img2_array.shape[2] == 4:  # RGBA
                    img2_cv = cv2.cvtColor(img2_array, cv2.COLOR_RGBA2GRAY)
                elif img2_array.shape[2] == 3:  # RGB
                    img2_cv = cv2.cvtColor(img2_array, cv2.COLOR_RGB2GRAY)
                else:
                    img2_cv = img2_array[:,:,0]  # Take first channel
            else:  # Already grayscale
                img2_cv = img2_array
            
            # Resize if necessary
            if img1_cv.shape != img2_cv.shape:
                img2_cv = cv2.resize(img2_cv, (img1_cv.shape[1], img1_cv.shape[0]))
            
            start_time = time.time()
            result = analyze_signatures(img1_cv, img2_cv, matching, device, use_rotation, use_preprocessing)
            processing_time = time.time() - start_time
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Analiz Sonuçları")
        
        # Main result box
        if result['predicted_same']:
            st.markdown(f"""
            <div class="result-box match-positive">
                <h3>✅ EŞLEŞME TESPİT EDİLDİ</h3>
                <p><strong>Eşleşme Oranı:</strong> {result['ratio']*100:.1f}%</p>
                <p><strong>Eşik Değer:</strong> {result['threshold']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box match-negative">
                <h3>❌ EŞLEŞME TESPİT EDİLMEDİ</h3>
                <p><strong>Eşleşme Oranı:</strong> {result['ratio']*100:.1f}%</p>
                <p><strong>Eşik Değer:</strong> {result['threshold']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed statistics with beautiful cards
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 2em;">🎯</h3>
                <h2 style="margin: 5px 0;">{result['valid_matches']}</h2>
                <p style="margin: 0; opacity: 0.9;">Eşleşme Sayısı</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 2em;">🔘</h3>
                <h2 style="margin: 5px 0;">{result['total_keypoints']}</h2>
                <p style="margin: 0; opacity: 0.9;">Toplam Nokta</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat3:
            rotation_text = f"{result['rotation_angle']}°" if result['rotation_used'] else "Yok"
            rotation_emoji = "🔄" if result['rotation_used'] else "🔒"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 2em;">{rotation_emoji}</h3>
                <h2 style="margin: 5px 0;">{rotation_text}</h2>
                <p style="margin: 0; opacity: 0.9;">Rotasyon</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat4:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 2em;">⏱️</h3>
                <h2 style="margin: 5px 0;">{processing_time:.2f}s</h2>
                <p style="margin: 0; opacity: 0.9;">İşlem Süresi</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization with enhanced presentation
        st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <h3 style="color: #667eea; font-family: 'Poppins', sans-serif;">
                🎨 Görsel Analiz - Eşleşen Noktalar
            </h3>
            <p style="color: #666; margin-bottom: 20px;">
                Yeşil çizgiler başarılı eşleştirmeleri gösterir (K-means ön işlemeli)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create and display visualization
        vis = create_visualization(result)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        st.image(vis_rgb, use_column_width=True)
        
        # Add a progress bar for visual appeal
        if result['predicted_same']:
            st.progress(result['ratio'])
            st.markdown(f"<p style='text-align: center; color: #28a745; font-weight: bold;'>✨ Eşleşme Güven Skoru: {result['ratio']*100:.1f}% ✨</p>", unsafe_allow_html=True)
        else:
            st.progress(result['ratio'])
            st.markdown(f"<p style='text-align: center; color: #dc3545; font-weight: bold;'>⚠️ Eşleşme Skoru: {result['ratio']*100:.1f}% (Yetersiz) ⚠️</p>", unsafe_allow_html=True)
        
        # Show all rotation results if requested
        if show_all_angles and use_rotation:
            st.subheader("📊 Tüm Rotasyon Sonuçları")
            
            angles_data = []
            for res in result['all_results']:
                angles_data.append({
                    'Açı': f"{res['angle']}°",
                    'Eşleşme Oranı': f"{res['ratio']*100:.1f}%",
                    'Eşleşme Sayısı': res['valid'],
                    'Toplam Nokta': res['total']
                })
            
            st.dataframe(angles_data, use_container_width=True)
        
        # Additional info box
        st.markdown(f"""
        <div class="stats-box">
            <h4>ℹ️ Analiz Detayları</h4>
            <ul>
                <li><strong>Ön İşleme:</strong> {'K-means uygulandı' if use_preprocessing else 'Kullanılmadı'}</li>
                <li><strong>Rotasyon Stratejisi:</strong> {'Kullanıldı' if result['rotation_used'] else 'Kullanılmadı'}</li>
                <li><strong>En İyi Açı:</strong> {result['rotation_angle']}°</li>
                <li><strong>Güvenlik Riski:</strong> {result['security_analysis']['risk_level']}</li>
                <li><strong>Güvenilirlik:</strong> {'Yüksek' if result['total_keypoints'] >= 20 else 'Orta'}</li>
                <li><strong>Eşik Değer:</strong> {result['threshold']*100:.1f}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    # Enhanced instructions with beautiful cards
    st.markdown("""
    <div style="text-align: center; margin: 40px 0;">
        <h2 style="color: #667eea; font-family: 'Poppins', sans-serif;">
            👆 Lütfen analiz etmek istediğiniz iki imza dosyasını yükleyin
        </h2>
        <p style="color: #666; font-size: 1.1em; margin: 20px 0;">
            Desteklenen formatlar: PNG, JPG, JPEG
        </p>
    </div>
    """, unsafe_allow_html=True)
    