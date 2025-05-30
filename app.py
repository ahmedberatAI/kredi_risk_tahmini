import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import warnings

# Uyarıları gizle
warnings.filterwarnings('ignore')

# Matplotlib backend ayarı (Windows için)
import matplotlib

matplotlib.use('Agg')

# Sayfa ayarları
st.set_page_config(page_title="Kredi Riski Tahmini", page_icon="💳", layout="centered")

# Başlık
st.title("💳 Kredi Riski Tahmini Uygulaması")
st.markdown("""
Kredi kartı kullanıcılarının temerrüde düşme olasılığını tahmin etmek için bu uygulamayı kullanabilirsiniz.  
Tahminler **LightGBM** makine öğrenmesi modeliyle yapılmaktadır.
""", unsafe_allow_html=True)

# Model dosyalarının varlığını kontrol et
try:
    # Model ve özellikleri yükle
    if os.path.exists("lightgbm_credit_model.pkl") and os.path.exists("selected_features.pkl"):
        model = joblib.load("lightgbm_credit_model.pkl")
        features = joblib.load("selected_features.pkl")
        model_loaded = True
    else:
        st.error(
            "❌ Model dosyaları bulunamadı! 'lightgbm_credit_model.pkl' ve 'selected_features.pkl' dosyalarının aynı klasörde olduğundan emin olun.")
        model_loaded = False
        # Demo için varsayılan özellikler
        features = ["LIMIT_BAL", "AGE", "AVG_BILL", "AVG_PAY_AMT", "TOTAL_PAY_AMT", "MAX_DELAY", "AVG_DELAY"]

except Exception as e:
    st.error(f"❌ Model yüklenirken hata oluştu: {str(e)}")
    model_loaded = False
    features = ["LIMIT_BAL", "AGE", "AVG_BILL", "AVG_PAY_AMT", "TOTAL_PAY_AMT", "MAX_DELAY", "AVG_DELAY"]

# Girdi alanı
st.header("🧮 Müşteri Bilgilerini Girin")

# İngilizce -> Türkçe eşleştirme sözlüğü
feature_labels = {
    "LIMIT_BAL": "Kredi Limiti (TL)",
    "AGE": "Yaş",
    "AVG_BILL": "Ortalama Borç Miktarı (TL)",
    "AVG_PAY_AMT": "Ortalama Ödeme Miktarı (TL)",
    "TOTAL_PAY_AMT": "Toplam Ödeme Miktarı (TL)",
    "MAX_DELAY": "En Yüksek Gecikme Süresi (Ay)",
    "AVG_DELAY": "Ortalama Gecikme Süresi (Ay)"
}

# Girdileri al
cols = st.columns(2)
user_input = {}

for i, feature in enumerate(features):
    label = feature_labels.get(feature, feature)
    with cols[i % 2]:
        if feature == "AGE":
            user_input[feature] = st.number_input(label, min_value=18, max_value=100, value=25)
        elif "LIMIT" in feature or "BILL" in feature or "PAY_AMT" in feature:
            user_input[feature] = st.number_input(label, min_value=0.0, value=10000.0, step=1000.0)
        elif "DELAY" in feature:
            user_input[feature] = st.number_input(label, min_value=0, max_value=12, value=0)
        else:
            user_input[feature] = st.number_input(label, value=0.0)

# Tahmin butonu
if st.button("📊 Temerrüt Riskini Hesapla"):
    if model_loaded:
        try:
            input_array = np.array([list(user_input.values())])
            prediction = model.predict(input_array)[0]
            proba = model.predict_proba(input_array)[0][1]

            st.subheader("📈 Tahmin Sonucu")

            # Risk seviyesine göre renk kodlaması
            if proba < 0.3:
                st.metric("Temerrüt Olasılığı", f"{proba:.2%}")
                st.success("✅ Düşük Risk: Kredi verilebilir.")
            elif proba < 0.7:
                st.metric("Temerrüt Olasılığı", f"{proba:.2%}")
                st.warning("⚠️ Orta Risk: Dikkatli değerlendirme gerekli.")
            else:
                st.metric("Temerrüt Olasılığı", f"{proba:.2%}")
                st.error("🚨 Yüksek Risk: Temerrüt ihtimali yüksek.")

            # SHAP analizi için alternatif yaklaşım
            st.subheader("🧠 Model Açıklaması")
            st.write("Bu tahmini etkileyen faktörler:")

            try:
                # SHAP kullanmaya çalış
                import shap

                # LightGBM modeli için TreeExplainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_array)

                # SHAP değerlerini feature isimlerle eşleştir
                shap_dict = {}
                for i, feature in enumerate(features):
                    shap_dict[feature_labels.get(feature, feature)] = shap_values[0][i]

                # Pozitif ve negatif etkileri ayır
                positive_effects = {k: v for k, v in shap_dict.items() if v > 0}
                negative_effects = {k: v for k, v in shap_dict.items() if v < 0}

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Riski Artıran Faktörler:**")
                    for feature, value in sorted(positive_effects.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"• {feature}: +{value:.3f}")

                with col2:
                    st.write("**Riski Azaltan Faktörler:**")
                    for feature, value in sorted(negative_effects.items(), key=lambda x: x[1]):
                        st.write(f"• {feature}: {value:.3f}")

                # Basit bar chart oluştur
                fig, ax = plt.subplots(figsize=(10, 6))
                features_tr = [feature_labels.get(f, f) for f in features]
                values = shap_values[0]
                colors = ['red' if v > 0 else 'green' for v in values]

                bars = ax.barh(features_tr, values, color=colors, alpha=0.7)
                ax.set_xlabel('SHAP Değeri (Risk Üzerindeki Etki)')
                ax.set_title('Özelliklerin Risk Tahmini Üzerindeki Etkisi')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

                # Değerleri çubukların üzerine yaz
                for i, (bar, value) in enumerate(zip(bars, values)):
                    ax.text(value + (0.001 if value >= 0 else -0.001), i,
                            f'{value:.3f}', ha='left' if value >= 0 else 'right',
                            va='center', fontsize=9)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except ImportError:
                st.warning("⚠️ SHAP kütüphanesi yüklü değil. Basit analiz gösteriliyor.")

                # SHAP olmadan basit özellik önemi analizi
                feature_importance = model.feature_importances_
                importance_dict = dict(zip(features, feature_importance))

                st.write("**Model Özellik Önemleri:**")
                for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
                    label = feature_labels.get(feature, feature)
                    st.write(f"• {label}: {importance:.3f}")

            except Exception as e:
                st.warning(f"SHAP analizi yapılamadı: {str(e)}")
                st.write("Temel tahmin sonucu gösteriliyor.")

        except Exception as e:
            st.error(f"❌ Tahmin yapılırken hata oluştu: {str(e)}")
    else:
        st.error("❌ Model yüklenmediği için tahmin yapılamıyor.")

# Yan panel bilgileri
with st.sidebar:
    st.header("ℹ️ Uygulama Bilgileri")
    st.write("**Model:** LightGBM")
    st.write("**Veri Seti:** UCI Credit Card Default Dataset")
    st.write("**Amaç:** Kredi kartı temerrüt riski tahmini")

    st.header("📋 Kullanım Talimatları")
    st.write("""
    1. Sol taraftaki formu doldurun
    2. 'Temerrüt Riskini Hesapla' butonuna tıklayın
    3. Sonuçları ve açıklamaları inceleyin
    """)

    if not model_loaded:
        st.header("⚠️ Dikkat")
        st.error("""
        Model dosyaları bulunamadı!

        Gerekli dosyalar:
        - lightgbm_credit_model.pkl
        - selected_features.pkl

        Bu dosyaları Python script'inle aynı klasöre koyun.
        """)

    st.header("🔧 Gereksinimler")
    st.code("""
    pip install streamlit
    pip install joblib
    pip install lightgbm
    pip install matplotlib
    pip install numpy
    pip install shap
    """)

# Alt bilgi
st.markdown("---")
st.caption("Bu uygulama eğitim amaçlıdır. Gerçek kredi kararları için profesyonel değerlendirme yapılmalıdır.")