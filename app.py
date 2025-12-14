import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import nltk
import gensim.downloader as api
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import os
import sys
import altair as alt 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib 

# --- SETUP NLTK & FUNGSI ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
    
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Clean text untuk AWE
def clean_text_for_ml(text):
    text = str(text).lower()
    text = re.sub(r'[\u200b\u200c\u200d\uFEFF]', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^\w\s]", " ", text) 
    text = re.sub(r'\s+', ' ', text).strip()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words] 
    return " ".join(words)

# AWE Vektor
def document_vector(text, wv):
    words = text.split()
    word_vecs = [wv[word] for word in words if word in wv]
    if not word_vecs:
        return np.zeros(100) 
    return np.mean(word_vecs, axis=0)

# --- MUAT MODEL & DATA CACHE ---
@st.cache_resource
def load_resources():
    model_svm_loaded = None
    if os.path.exists('model_svm_winner.joblib'):
        try:
            with open('model_svm_winner.joblib', 'rb') as file:
                model_svm_loaded = joblib.load(file)
        except Exception as e:
            st.error(f"Gagal memuat Model SVM: {e}. PASTIKAN FILE MODEL BERADA DI 'model_svm_winner.joblib'.")
            
    word_vectors_loaded = None
    try:
        word_vectors_loaded = api.load("glove-wiki-gigaword-100") 
    except Exception as e:
        st.error(f"Gagal memuat Word Vectors dari Gensim: {e}")

    df_summary = pd.DataFrame()
    if os.path.exists('final_summary_results.csv'):
        df_summary = pd.read_csv('final_summary_results.csv')
    
    df_eda = pd.DataFrame()
    if os.path.exists('eda_data.csv'):
         df_eda = pd.read_csv('eda_data.csv')
         df_eda['word_count'] = df_eda['review'].apply(lambda x: len(str(x).split()))
         analyzer = SentimentIntensityAnalyzer()
         df_eda['label_vader'] = df_eda['review'].apply(lambda x: 'positive' if analyzer.polarity_scores(x)['compound'] >= 0 else 'negative')

    df_raw = pd.DataFrame()
    if os.path.exists('raw_data_stats.csv'):
        df_raw = pd.read_csv('raw_data_stats.csv')
        df_raw = df_raw.dropna(subset=['rating']) 

    return model_svm_loaded, word_vectors_loaded, df_eda, df_summary, df_raw

model_svm, word_vectors, df_eda, df_summary, df_raw = load_resources()
le_classes = {0: 'negative', 1: 'positive'} 

st.set_page_config(layout="wide")
st.title("🌟 Proyek Analisis Sentimen Film La La Land")
st.markdown("Metodologi Klasifikasi dengan Word Embeddings (AWE) dan Penanganan Imbalance Data")
st.markdown("---")

# --- NAVIGASI DASHBOARD (5 TAB) ---
tab1, tab5, tab2, tab3, tab4 = st.tabs(["🚀 Prediksi Real-Time", "⚙️ Pipeline Data & Proses Awal", "📊 Dashboard Analisis Data", "🔍 Komparasi Pelabelan", "🏆 Kinerja Model (9 Skenario)"])


# =======================================================
# TAB 5: PIPELINE DATA & PROSES AWAL
# =======================================================
with tab5:
    st.header("Pipeline Data & Proses Awal ⚙️")
    
    if not df_raw.empty:
        st.subheader("1. Statistik Data Awal")
        
        # Row 1: Distribusi Rating
        col_r1_1, col_r1_2 = st.columns([1, 1])
        with col_r1_1:
            st.caption("Distribusi Rating Asli")
            fig_rating, ax_rating = plt.subplots(figsize=(4, 3))
            sns.countplot(x=df_raw['rating'], ax=ax_rating, palette="viridis")
            ax_rating.set_title('Distribusi Rating')
            ax_rating.set_xlabel('Rating')
            ax_rating.set_ylabel('Count')
            st.pyplot(fig_rating)
            st.info("**Insight:** Dominasi rating tinggi (4.5 & 5.0) mengindikasikan *class imbalance* kuat sebelum pelabelan.")

        # Row 2: Boxplot dan Statistik Dasar 
        col_r2_1, col_r2_2 = st.columns([1, 1])
        
        with col_r2_1:
            st.caption("Boxplot Panjang Review (Outlier)")
            fig_box_len, ax_box_len = plt.subplots(figsize=(4, 3))
            sns.boxplot(df_raw["review_length"], ax=ax_box_len, color='orange')
            ax_box_len.set_title("Panjang Karakter Review")
            st.pyplot(fig_box_len)
            st.warning("**Preprocessing:** Diperlukan penanganan Outlier jika klasifikasi berbasis frekuensi, tetapi AWE lebih toleran.")
        
        with col_r2_2:
            st.caption("Ringkasan Statistik Jumlah Kata")
            st.dataframe(df_raw[['word_count']].describe().round(1), use_container_width=True)
            st.success(f"**Data Size:** {len(df_raw)} ulasan")
            st.markdown(f"**Rata-rata Kata:** {df_raw['word_count'].mean():.1f}")
        
        st.markdown("---")

        # 2. Proses Preprocessing dan Vektorisasi
        st.subheader("2. Preprocessing & Feature Extraction")
        
        col_prep, col_vect = st.columns(2)
        
        with col_prep:
            st.caption("Contoh Ulasan Asli vs Bersih")
            col_in1, col_in2 = st.columns([1, 1])
            with col_in1:
                st.code(f"Review Asli: {df_raw['review'].iloc[15][:100]}...")
                st.info("**Preprocessing:** Meliputi Lowercasing, menghilangkan Punctuation/URL, dan Lemmatisasi.")
            
            with col_in2:
                st.code(f"Review Bersih: {clean_text_for_ml(df_raw['review'].iloc[15])}")
                st.markdown("**Tujuan:** Menstandarisasi kosakata sebelum feature extraction.")

            
        with col_vect:
            st.caption("Vektorisasi Fitur")
            st.markdown(
                """
                Fitur diubah dari teks menjadi numerik menggunakan **Average Word Embeddings (AWE)** dari GloVe 100-dimensi.
                * **Metode:** GloVe setiap kata diubah menjadi vektor, kemudian vektor rata-rata dihitung untuk seluruh ulasan.
                * **Output:** Setiap ulasan menjadi 1 vektor dengan **100 dimensi**.
                """
            )
            if word_vectors is not None:
                st.success("✅ Model Word Embeddings (GloVe 100D) Berhasil Dimuat.")
            else:
                st.error("🚨 Model GloVe gagal dimuat.")


# =======================================================
# TAB 1: PREDIKSI REAL-TIME 
# =======================================================
with tab1:
    st.header("Deteksi Sentimen Ulasan Baru 💬")
    st.subheader("Model Pemenang: **SVM LinearSVC (AWE + SMOTE)**")
    
    col_input, col_pred = st.columns([2, 1])
    
    with col_input:
        user_input = st.text_area(
            "Tulis ulasan film *La La Land* di sini:", 
            height=200, 
            placeholder="Contoh: The jazz music was excellent, but the plot dragged on way too long and I found the main characters annoying."
        )

    with col_pred:
        st.markdown("---")
        if st.button("PREDIKSI SENTIMEN", use_container_width=True, type="primary"):
            if not user_input:
                st.warning("⚠️ Mohon masukkan ulasan terlebih dahulu.")
            elif model_svm is None or word_vectors is None:
                st.error("🚨 Model belum berhasil dimuat. Coba refresh halaman.")
            else:
                with st.spinner('Memproses...'):
                    cleaned_review = clean_text_for_ml(user_input)
                    AWE_vector = document_vector(cleaned_review, word_vectors).reshape(1, -1)

                    prediction_id = model_svm.predict(AWE_vector)[0]
                    prediction_label = le_classes.get(prediction_id, 'Unknown')
                    
                    st.subheader("Hasil:")
                    if prediction_label == 'positive':
                        st.success(f"**POSITIF** 🎉", icon="👍")
                        st.markdown("Ulasan ini cenderung berisi **apresiasi**.")
                    else:
                        st.error(f"**NEGATIF** 🙁", icon="👎")
                        st.markdown("Ulasan ini mendeteksi adanya **kritik**.")
                
                    st.caption(f"Input Setelah Preprocessing: {cleaned_review}")

# =======================================================
# TAB 2: DASHBOARD ANALISIS DATA (EDA - Final Bersih)
# =======================================================
with tab2:
    st.header("Analisis Data Eksplorasi (EDA) Ulasan 🔬")
    
    if not df_eda.empty:
        
        # 1. VISUALISASI UTAMA: Word Cloud vs Histogram
        st.subheader("1. Analisis Struktur Teks")
        col2_1, col2_2 = st.columns(2)

        with col2_1:
            st.caption("Histogram: Distribusi Panjang Ulasan")
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
            sns.histplot(df_eda['word_count'], ax=ax_hist, kde=True, bins=30, color='#1f77b4')
            ax_hist.set_title("Distribusi Jumlah Kata per Ulasan")
            ax_hist.set_xlabel("Jumlah Kata")
            st.pyplot(fig_hist)
            st.info("Mayoritas ulasan berada pada rentang singkat (<50 kata), membenarkan fitur AWE.")


        with col2_2:
            st.caption("Word Cloud Kata Kunci Ulasan")
            text = " ".join(df_eda['review'].astype(str))
            wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(text)
            fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
            st.info("Kata kunci utama menunjukkan fitur semantik yang dominan ('music', 'love', 'city').")
        
        st.markdown("---")

        # 2. BOX PLOT SENTIMEN
        st.subheader("2. Analisis Perilaku Penulisan (Panjang vs Sentimen)")
        
        col_box, col_box_info = st.columns([2, 1])
        with col_box:
            st.caption("Box Plot: Panjang Ulasan vs. Sentimen (Berdasarkan Label VADER)")
            if 'label_vader' in df_eda.columns:
                fig_box, ax_box = plt.subplots(figsize=(8, 4))
                sns.boxplot(x='label_vader', y='word_count', data=df_eda, palette={'positive': '#33FF57', 'negative': '#FF5733'}, ax=ax_box)
                ax_box.set_title('Distribusi Panjang Kata Berdasarkan Sentimen')
                ax_box.set_xlabel('Sentimen (VADER)')
                ax_box.set_ylabel('Jumlah Kata')
                st.pyplot(fig_box)
            else:
                st.error("Kolom label_vader tidak tersedia untuk Box Plot.")
            
        with col_box_info:
            st.markdown("---")
            st.warning(
                """
                **Insight Box Plot:** Ulasan Negatif cenderung memiliki variasi panjang yang lebih besar, menunjukkan perlunya detail lebih saat mengkritik.
                """
            )
        
        st.markdown("---")


# =======================================================
# TAB 3: KOMPARASI PELABELAN
# =======================================================
with tab3:
    st.header("Komparasi Hasil Pelabelan Otomatis 🔎")
    
    if all(col in df_eda.columns for col in ['label_vader', 'label_flair', 'label_zeroshot']):
        
        # 1. VISUALISASI PERBANDINGAN (Tiga Grafik Terpisah dalam Kolom)
        st.subheader("1. Distribusi Label dari Setiap Metode")
        
        col_vader, col_flair, col_zeroshot = st.columns(3)
        
        # --- Fungsi Pembantu untuk Membuat Grafik Batang ---
        def create_bar_chart(df, column, title):
            counts = df[column].value_counts().reset_index()
            counts.columns = ['Sentimen', 'Count']
            
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.barplot(x='Sentimen', y='Count', data=counts, palette={'positive': '#33FF57', 'negative': '#FF5733'}, ax=ax)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('Count', fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=8)
            plt.close(fig) 
            return fig
        # --------------------------------------------------

        with col_vader:
            st.caption("Grafik 1: Pelabelan VADER")
            st.pyplot(create_bar_chart(df_eda, 'label_vader', 'VADER'))
            
        with col_flair:
            st.caption("Grafik 2: Pelabelan FLAIR")
            st.pyplot(create_bar_chart(df_eda, 'label_flair', 'FLAIR'))
            
        with col_zeroshot:
            st.caption("Grafik 3: Pelabelan ZERO-SHOT")
            st.pyplot(create_bar_chart(df_eda, 'label_zeroshot', 'ZERO-SHOT'))

        st.markdown("---")
        
        # 2. RINGKASAN DATA (Presisi Tabel sebagai Komparasi)
        st.subheader("2. Ringkasan Komparasi Numerik dan Justifikasi")
        
        col_table, col_justifikasi = st.columns([1, 1])

        with col_table:
            summary_data = {
                'Metode': ['VADER', 'FLAIR', 'ZERO-SHOT'],
                'Positif': [
                    df_eda['label_vader'].value_counts().get('positive', 0),
                    df_eda['label_flair'].value_counts().get('positive', 0),
                    df_eda['label_zeroshot'].value_counts().get('positive', 0)
                ],
                'Negatif': [
                    df_eda['label_vader'].value_counts().get('negative', 0),
                    df_eda['label_flair'].value_counts().get('negative', 0),
                    df_eda['label_zeroshot'].value_counts().get('negative', 0)
                ]
            }
            summary_table = pd.DataFrame(summary_data).set_index('Metode')
            
            st.caption("Tabel Komparasi Jumlah Ulasan Positif/Negatif:")
            st.dataframe(summary_table, use_container_width=True)
        
        with col_justifikasi:
            st.info("Justifikasi Pilihan *Ground Truth* (VADER)")
            st.markdown(
                """
                * **Fokus Visual:** Grafik terpisah menunjukkan distribusi kelas setiap metode secara detail.
                * **VADER** dipilih karena menghasilkan rasio Positif/Negatif yang **paling realistis** (mayoritas positif) untuk ulasan film populer, dan konsisten (*lexicon-based*).
                * Tabel ringkasan berfungsi sebagai komparasi utama antar metode.
                """
            )
        
    else:
        st.error("Data EDA tidak mengandung semua kolom labeling yang diperlukan (VADER, FLAIR, Zero-Shot).")

# =======================================================
# TAB 4: KINERJA MODEL 
# =======================================================
with tab4:
    st.header("Ringkasan Kinerja Model Berdasarkan F1-Score (Weighted) 🏆")
    st.info("F1-Score adalah metrik terbaik untuk mengukur kinerja pada data yang tidak seimbang.")
    
    if not df_summary.empty:
        df_summary_display = df_summary.copy()
        
        # 1. VISUALISASI PERBANDINGAN 9 SKENARIO (Horizontal Bar Chart Menurun)
        st.subheader("1. F1-Score dari Semua 9 Skenario (Diurutkan Menurun)")
        
        chart_data = df_summary_display[[
            'Skenario', 
            'F1-Score (Weighted Avg)',
            'Akurasi'
        ]].sort_values(by='F1-Score (Weighted Avg)', ascending=False).reset_index(drop=True)
        
        chart_data['Algoritma'] = chart_data['Skenario'].apply(lambda x: x.split(' ')[0])
        
        chart_f1 = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('F1-Score (Weighted Avg):Q', title='F1-Score (Weighted Average)'),
            y=alt.Y('Skenario:N', title='Skenario', sort='-x'), 
            color=alt.Color('Algoritma:N'),
            tooltip=['Skenario', 'F1-Score (Weighted Avg)', 'Akurasi']
        ).properties(
            title='Perbandingan F1-Score 9 Skenario'
        ).interactive()
        
        st.altair_chart(chart_f1, use_container_width=True)

        st.markdown("---")
        
        # 2. TABEL RINGKASAN AKHIR (Diurutkan Menurun)
        st.subheader("2. Tabel Detail Kinerja (Diurutkan Menurun)")
        
        def highlight_winner(s):
            is_winner = s['Skenario'] == 'SVM 3.2 (AWE+SMOTE)'
            return ['background-color: #f7ffdd; color: #4CAF50; font-weight: bold' if is_winner else '' for v in s]

        st.dataframe(
            df_summary_display.sort_values(by='F1-Score (Weighted Avg)', ascending=False).style.apply(highlight_winner, axis=1), 
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("3. Kesimpulan Model Pemenang:")
        
        st.success(
            f"**Model Pemenang:** **SVM LinearSVC 3.2 (AWE + SMOTE)**\n\n"
            f"- **F1-Score Tertinggi:** **{df_summary_display[df_summary_display['Skenario'] == 'SVM 3.2 (AWE+SMOTE)']['F1-Score (Weighted Avg)'].iloc[0]:.4f}**\n"
            f"- **Kunci Sukses:** Visualisasi menunjukkan bahwa **semua skenario SMOTE** mendominasi di posisi atas, membuktikan bahwa penanganan *imbalance* jauh lebih penting daripada pemilihan algoritma (LR, RF, atau SVM) itu sendiri dalam kasus ini. "
        )
