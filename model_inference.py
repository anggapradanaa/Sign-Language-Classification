import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import gradio as gr
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Cek versi Gradio
print(f"Gradio version: {gr.__version__}")

# ==================== LOAD MODEL TERBAIK ====================
MODEL_PATH = r"D:\Perkuliahan\Data Science and Machine Learning\SignLanguage\models\best_sibi_model.keras"

# Cek apakah model ada
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model tidak ditemukan di {MODEL_PATH}")
    print("Pastikan file model ada di direktori yang benar!")
else:
    print(f"Loading model dari {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model berhasil di-load!")

# ==================== DEFINISI KELAS SIBI ====================
# 24 kelas SIBI (tanpa J dan Z)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

print(f"Jumlah kelas: {len(classes)}")
print(f"Kelas: {classes}")
print("Note: J dan Z tidak ada dalam SIBI karena menggunakan gerakan")

# ==================== PARAMETER MODEL ====================
IMG_SIZE = (224, 224)  # Ukuran input MobileNetV2

# ==================== FUNGSI PREDIKSI ====================
def predict_sibi(image):
    """
    Fungsi untuk memprediksi gambar SIBI
    Args:
        image: PIL Image atau numpy array
    Returns:
        str: Top 3 prediksi dengan confidence score dalam format HTML
    """
    if image is None:
        return """
        <div style="text-align: center; padding: 20px; border: 2px dashed #ccc; border-radius: 10px;">
            <h3 style="color: #666;">📸 Belum ada gambar</h3>
            <p>Upload gambar untuk melihat prediksi</p>
        </div>
        """
    
    try:
        # Konversi PIL Image ke numpy array jika perlu
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Preprocessing gambar
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image
            img_processed = cv2.resize(image, IMG_SIZE)
        else:
            # Grayscale atau format lain
            img_processed = cv2.resize(image, IMG_SIZE)
            if len(img_processed.shape) == 2:
                img_processed = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2RGB)
        
        # Normalisasi ke [0, 1]
        img_processed = img_processed.astype(np.float32) / 255.0
        
        # Tambah batch dimension
        img_batch = np.expand_dims(img_processed, axis=0)
        
        # Prediksi
        predictions = model.predict(img_batch, verbose=0)
        predictions = predictions[0]  # Ambil prediksi pertama dari batch
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]  # 3 tertinggi, descending
        
        # Format hasil sebagai HTML dengan styling yang lebih menarik
        results = """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 20px; border-radius: 15px; color: white; margin: 10px 0;">
            <h3 style="margin: 0 0 15px 0; text-align: center;">🎯 HASIL PREDIKSI</h3>
        </div>
        """
        
        for i, idx in enumerate(top_3_indices):
            class_name = classes[idx]
            confidence = float(predictions[idx]) * 100
            
            # Tentukan warna berdasarkan peringkat
            if i == 0:
                color = "#4CAF50"  # Hijau untuk prediksi terbaik
                emoji = "🥇"
            elif i == 1:
                color = "#FF9800"  # Orange untuk kedua
                emoji = "🥈"
            else:
                color = "#2196F3"  # Biru untuk ketiga
                emoji = "🥉"
            
            # Buat progress bar
            progress_width = confidence
            
            results += f"""
            <div style="background: white; margin: 10px 0; padding: 15px; 
                       border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 18px; font-weight: bold; color: {color};">
                        {emoji} Huruf {class_name}
                    </span>
                    <span style="font-size: 16px; font-weight: bold; color: {color};">
                        {confidence:.1f}%
                    </span>
                </div>
                <div style="background: #f0f0f0; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: {color}; height: 100%; width: {progress_width}%; 
                               transition: width 0.3s ease;"></div>
                </div>
            </div>
            """
        
        return results
        
    except Exception as e:
        return f"""
        <div style="background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; 
                   padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h4 style="margin: 0 0 10px 0;">❌ Error saat prediksi</h4>
            <p style="margin: 0;">Detail: {str(e)}</p>
        </div>
        """

# ==================== FUNGSI UNTUK RESET INTERFACE ====================
def reset_interface():
    """Reset interface ke kondisi awal"""
    return None, """
    <div style="text-align: center; padding: 20px; border: 2px dashed #ccc; border-radius: 10px;">
        <h3 style="color: #666;">🔄 Interface di-reset</h3>
        <p>Siap untuk prediksi baru!</p>
    </div>
    """

# ==================== SETUP GRADIO INTERFACE ====================
# CSS custom untuk styling yang lebih menarik
custom_css = """
.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
}

.main-header {
    text-align: center;
    padding: 30px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 30px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.upload-section {
    background: #f8f9fa;
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    border: 2px dashed #dee2e6;
    transition: all 0.3s ease;
}

.upload-section:hover {
    border-color: #667eea;
    background: #f0f4ff;
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    padding: 12px 30px !important;
    border-radius: 25px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.btn-secondary {
    background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
    border: none !important;
    padding: 12px 30px !important;
    border-radius: 25px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

.btn-secondary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(108, 117, 125, 0.4) !important;
}

.result-container {
    background: white;
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    min-height: 300px;
}

.info-card {
    background: white;
    border-radius: 15px;
    padding: 25px;
    margin: 15px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-left: 5px solid #667eea;
}
"""

# Interface
with gr.Blocks(
    title="SIBI Classifier - Upload File",
    theme=gr.themes.Soft(),
    css=custom_css
) as interface:
    
    # Header 
    gr.HTML("""
    <div class="main-header">
        <h1 style="font-size: 2.5em; margin: 0 0 10px 0;">SIBI Classifier</h1>
        <p style="font-size: 1.2em; margin: 0 0 10px 0;">Sistem Klasifikasi Isyarat Bahasa Indonesia</p>
        <p style="font-size: 1em; opacity: 0.9; margin: 0;">Powered by Deep Learning & MobileNetV2</p>
    </div>
    """)
    
    # Tab untuk berbagai input method
    with gr.Tabs():
        # Tab 1: Upload File
        with gr.TabItem("📁 Upload & Analisis Gambar"):
            gr.HTML("""
            <div style="text-align: center; margin: 20px 0;">
                <h3>Upload gambar gestur tangan SIBI</h3>
                <p>Pastikan gambar jelas dan tangan terlihat dengan baik</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        file_input = gr.Image(
                            label="🖼️ Pilih Gambar",
                            type="pil",
                            height=400,
                            elem_classes="upload-section"
                        )
                        
                        with gr.Row():
                            file_button = gr.Button(
                                "🔍 Prediksi", 
                                variant="primary",
                                size="lg",
                                elem_classes="btn-primary"
                            )
                            file_reset = gr.Button(
                                "🔄 Reset", 
                                variant="secondary",
                                size="lg",
                                elem_classes="btn-secondary"
                            )
                
                with gr.Column(scale=1):
                    file_output = gr.HTML(
                        label="Hasil Prediksi",
                        value="""
                        <div style="text-align: center; padding: 20px; border: 2px dashed #ccc; border-radius: 10px;">
                            <h3 style="color: #666;">📸 Belum ada gambar</h3>
                            <p>Upload gambar untuk melihat prediksi</p>
                        </div>
                        """,
                        elem_classes="result-container"
                    )
            
            # Tips untuk upload yang lebih detail
            gr.HTML("""
            <div class="info-card">
                <h4>📱 Panduan Upload Gambar:</h4>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 10px;">
                    <ol>
                        <li><strong>Klik area upload</strong> atau drag & drop gambar</li>
                        <li><strong>Pilih file gambar</strong> (JPG, PNG, atau format lain)</li>
                        <li><strong>Tunggu gambar ter-load</strong> - Preview akan muncul</li>
                        <li><strong>Klik "Analisis Gambar"</strong> untuk prediksi</li>
                        <li><strong>Lihat hasil</strong> - Top 3 prediksi dengan confidence score</li>
                    </ol>
                </div>
                
                <h4>🎯 Tips untuk Hasil Terbaik:</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                    <div>
                        <h5>✅ Lakukan</h5>
                        <ul>
                            <li>Gunakan pencahayaan yang cukup</li>
                            <li>Posisikan tangan di tengah gambar</li>
                            <li>Pastikan gestur sesuai standar SIBI</li>
                            <li>Gunakan background yang kontras</li>
                            <li>Gunakan gambar berkualitas tinggi</li>
                        </ul>
                    </div>
                    <div>
                        <h5>❌ Hindari</h5>
                        <ul>
                            <li>Gambar blur atau tidak fokus</li>
                            <li>Tangan terpotong atau tidak lengkap</li>
                            <li>Background yang mengganggu</li>
                            <li>Pencahayaan terlalu gelap/terang</li>
                            <li>Resolusi gambar terlalu rendah</li>
                        </ul>
                    </div>
                </div>
            </div>
            """)
    
        # Tab 2: Informasi yang lebih lengkap
        with gr.TabItem("ℹ️ Panduan & Informasi"):
            gr.HTML("""
            <div class="info-card">
                <h3>🎯 Tentang SIBI Classifier</h3>
                <p><strong>SIBI (Sistem Isyarat Bahasa Indonesia)</strong> adalah sistem komunikasi visual yang digunakan 
                oleh komunitas Tuli di Indonesia. Aplikasi ini menggunakan teknologi Deep Learning dengan 
                arsitektur MobileNetV2 untuk mengklasifikasikan gestur tangan menjadi huruf A-Z.</p>
            </div>
            
            <div class="info-card">
                <h3>📝 Cara Penggunaan</h3>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 15px;">
                    <h4>📁 Upload & Analisis Gambar</h4>
                    <ol>
                        <li><strong>Upload gambar:</strong> Klik area upload atau drag & drop file gambar</li>
                        <li><strong>Preview:</strong> Pastikan gambar ter-load dengan benar</li>
                        <li><strong>Analisis:</strong> Klik tombol "Analisis Gambar" untuk prediksi</li>
                        <li><strong>Hasil:</strong> Lihat top 3 prediksi dengan confidence score</li>
                        <li><strong>Reset:</strong> Klik "Reset" untuk mengupload gambar baru</li>
                    </ol>
                </div>
            </div>
            
            <div class="info-card">
                <h3>🤚 Alphabet SIBI yang Didukung</h3>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 10px;">
                    <p style="text-align: center; font-size: 1.5em; font-weight: bold; color: #667eea;">
                        A B C D E F G H I K L M N O P Q R S T U V W X Y
                    </p>
                    <p style="text-align: center; color: #666; margin: 10px 0 0 0;">
                        <strong>Catatan:</strong> Huruf J dan Z tidak tersedia karena menggunakan gerakan dinamis
                    </p>
                </div>
            </div>
            
            <div class="info-card">
                <h3>⚡ Spesifikasi Teknis</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                    <div>
                        <h4>🧠 Model</h4>
                        <ul>
                            <li><strong>Arsitektur:</strong> MobileNetV2</li>
                            <li><strong>Transfer Learning:</strong> ImageNet</li>
                            <li><strong>Input Size:</strong> 224×224 pixels</li>
                            <li><strong>Framework:</strong> TensorFlow/Keras</li>
                        </ul>
                    </div>
                    <div>
                        <h4>📊 Dataset</h4>
                        <ul>
                            <li><strong>Classes:</strong> 24 huruf SIBI</li>
                            <li><strong>Format:</strong> RGB Images</li>
                            <li><strong>Preprocessing:</strong> Normalization</li>
                            <li><strong>Output:</strong> Probability distribution</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="info-card">
                <h3>📊 Interpretasi Hasil</h3>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 10px;">
                    <h4>🎯 Confidence Score</h4>
                    <ul>
                        <li><strong>90-100%:</strong> Prediksi sangat yakin - kemungkinan benar tinggi</li>
                        <li><strong>70-89%:</strong> Prediksi cukup yakin - hasil dapat diandalkan</li>
                        <li><strong>50-69%:</strong> Prediksi ragu-ragu - perlu verifikasi manual</li>
                        <li><strong>< 50%:</strong> Prediksi tidak yakin - coba gambar yang lebih jelas</li>
                    </ul>
                    
                    <h4>🔍 Tips Membaca Hasil</h4>
                    <ul>
                        <li>Perhatikan perbedaan confidence score antara prediksi 1, 2, dan 3</li>
                        <li>Jika prediksi 1 dan 2 memiliki score yang dekat, ada kemungkinan ambiguitas</li>
                        <li>Gunakan prediksi dengan confidence score tertinggi</li>
                        <li>Jika semua score rendah, coba upload gambar yang lebih jelas</li>
                    </ul>
                </div>
            </div>
            """)
    
    # Event handlers dengan feedback yang lebih baik
    file_button.click(
        fn=predict_sibi,
        inputs=file_input,
        outputs=file_output
    )
    
    # Reset handlers
    file_reset.click(
        fn=reset_interface,
        outputs=[file_input, file_output]
    )
    
    # Auto-predict saat gambar di-upload (optional)
    file_input.change(
        fn=predict_sibi,
        inputs=file_input,
        outputs=file_output
    )
    
    # Footer dengan informasi lebih lengkap
    gr.HTML("""
    <div style="text-align: center; margin-top: 40px; padding: 25px; 
               background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
               border-radius: 15px; border-top: 3px solid #667eea;">
        <p style="font-size: 1.1em; margin: 0 0 10px 0; color: #495057;">
            🚀 <strong>SIBI Classifier v2.0</strong> - Upload File Interface
        </p>
        <p style="margin: 0; color: #6c757d;">
            Dibuat dengan ❤️ menggunakan TensorFlow, Keras & Gradio | 
            🧠 Deep Learning
        </p>
    </div>
    """)

# ==================== LAUNCH INTERFACE ====================
if __name__ == "__main__":
    print("🚀 Meluncurkan SIBI Classifier - Upload File Only...")
    print("📁 Interface sederhana dengan fitur upload file")
    print("📱 Interface akan terbuka di browser Anda")
    print("🔗 Atau akses melalui link yang muncul di bawah")
    
    # Launch dengan konfigurasi yang optimal
    try:
        interface.launch(
            share=True,              # Buat public link
            server_name="127.0.0.1", # Localhost
            server_port=7860,        # Port default
            debug=True,              # Debug mode
            show_error=True,         # Tampilkan error
            inbrowser=True,          # Auto open browser
            favicon_path=None,       # Bisa ditambahkan favicon custom
            app_kwargs={
                "docs_url": None,    # Disable docs
                "redoc_url": None,   # Disable redoc
            }
        )
    except Exception as e:
        print(f"❌ Error saat launch: {e}")
        print("🔄 Mencoba dengan konfigurasi sederhana...")
        interface.launch(share=True, inbrowser=True)