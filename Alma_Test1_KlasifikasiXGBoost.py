# ============================================================
# DASHBOARD STREAMLIT — KLASIFIKASI CUACA KOLKATA
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from xgboost import XGBClassifier, DMatrix

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Cuaca Kolkata",
    page_icon="🌦️",
    layout="wide"
)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ============================================================
# [1] JUDUL & DESKRIPSI PROYEK
# ============================================================
st.title("🌦️ Dashboard Klasifikasi Hari Hujan — Kolkata")
st.markdown("""
**Proyek:** Prediksi apakah suatu hari tergolong **hujan signifikan (≥ 5 mm)**
menggunakan **XGBoost Classifier** dari data cuaca harian Kolkata (2021–2026).

**Target:** `is_rainy` = 1 jika curah hujan ≥ 5 mm, 0 jika < 5 mm.
**Fitur:** suhu (mean/max/min), angin (speed/gust/direction), durasi sinar matahari, bulan.
""")
st.markdown("---")

# ============================================================
# LOAD DATA & TRAIN MODEL (CACHED)
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("D:\Klasifikasi Data Menggunakan XGBoost\kolkata_weather_jan_june_2026.csv")
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['is_rainy'] = (df['rain_sum'] >= 5.0).astype(int)
    return df

@st.cache_resource
def train_model(Xtr, y_train):
    m = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )
    m.fit(Xtr, y_train)
    return m

# Load & prepare
df = load_data()
fitur = [
    'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
    'wind_speed_10m_max', 'wind_gusts_10m_max',
    'wind_direction_10m_dominant', 'sunshine_duration', 'month'
]
X = df[fitur]
y = df['is_rainy']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)
model = train_model(Xtr, y_train)

y_pred  = model.predict(Xte)
y_proba = model.predict_proba(Xte)[:, 1]

# ============================================================
# [2] RINGKASAN DATASET
# ============================================================
st.header("1️⃣ Ringkasan Dataset")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah Data", f"{len(df):,}")
c2.metric("Jumlah Fitur", len(fitur))
c3.metric("Proporsi Hujan", f"{y.mean()*100:.1f}%")
c4.metric("Proporsi Tidak Hujan", f"{(1-y.mean())*100:.1f}%")

# ============================================================
# [3] EDA — DISTRIBUSI FITUR, KORELASI, PROPORSI KELAS
# ============================================================
st.header("2️⃣ Eksplorasi Data (EDA)")

# 3a. Distribusi SEMUA fitur
st.subheader("a) Distribusi Setiap Fitur")
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.ravel(), fitur):
    sns.histplot(df[col], kde=True, ax=ax, color='steelblue')
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel('Frekuensi')
plt.suptitle("Distribusi Setiap Fitur Cuaca Kolkata",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
st.pyplot(fig); plt.close()

# 3b. Proporsi kelas target
st.subheader("b) Proporsi Kelas Target")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=y, ax=ax, palette=['#4C9BE8', '#E8744C'])
    ax.set_xticklabels(["Tidak Hujan", "Hujan"])
    ax.set_title("Jumlah Hari per Kelas")
    ax.set_ylabel("Jumlah Hari")
    st.pyplot(fig); plt.close()
with col2:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(y.value_counts(),
           labels=["Tidak Hujan", "Hujan"],
           autopct='%1.1f%%',
           colors=['#4C9BE8', '#E8744C'],
           startangle=90, explode=(0, 0.05), shadow=True)
    ax.set_title("Proporsi Kelas")
    st.pyplot(fig); plt.close()

# 3c. Heatmap korelasi
st.subheader("c) Heatmap Korelasi Fitur vs Target")
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[fitur + ['is_rainy']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.5, ax=ax)
ax.set_title("Korelasi Fitur vs Target", fontweight='bold')
st.pyplot(fig); plt.close()

st.markdown("---")

# ============================================================
# [4 & 6] EVALUASI MODEL — METRIK, CONFUSION MATRIX, ROC
# ============================================================
st.header("3️⃣ Evaluasi Model XGBoost")

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)

# Metrik
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{acc:.3f}")
m2.metric("Precision", f"{prec:.3f}")
m3.metric("Recall", f"{rec:.3f}")
m4.metric("F1-Score", f"{f1:.3f}")
m5.metric("ROC-AUC", f"{auc:.3f}")

# Tabel ringkasan metrik
st.subheader("Tabel Ringkasan Metrik")
st.table(pd.DataFrame({
    "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Nilai": [acc, prec, rec, f1, auc]
}).round(4))

# Confusion Matrix & ROC
st.subheader("Confusion Matrix & ROC Curve")
col1, col2 = st.columns(2)
with col1:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Tidak Hujan", "Hujan"],
                yticklabels=["Tidak Hujan", "Hujan"],
                annot_kws={"size": 14, "fontweight": "bold"}, ax=ax)
    ax.set_xlabel("Prediksi", fontweight='bold')
    ax.set_ylabel("Aktual", fontweight='bold')
    ax.set_title("Confusion Matrix")
    st.pyplot(fig); plt.close()
with col2:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color='darkorange', lw=2)
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc='lower right')
    st.pyplot(fig); plt.close()

st.markdown("---")

# ============================================================
# [5] FEATURE IMPORTANCE
# ============================================================
st.header("4️⃣ Feature Importance")
imp = pd.Series(model.feature_importances_, index=fitur).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(imp)))
bars = ax.barh(imp.index, imp.values, color=colors,
               edgecolor='black', linewidth=0.7)
ax.set_xlabel("Importance Score (Gain)", fontweight='bold')
ax.set_title("Fitur Paling Berpengaruh terhadap Prediksi Hujan",
             fontweight='bold')
for b, v in zip(bars, imp.values):
    ax.text(v + 0.003, b.get_y() + b.get_height()/2,
            f"{v:.3f}", va='center', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
st.pyplot(fig); plt.close()

st.markdown("---")

# ============================================================
# [7] FORM PREDIKSI MANUAL + VISUALISASI HASIL
# ============================================================
st.header("5️⃣ Coba Prediksi Manual")
st.markdown("Masukkan nilai fitur cuaca di bawah, lalu klik **🔮 Prediksi**. "
            "Hasil akan menampilkan kelas prediksi, keyakinan model, "
            "dan alasan keputusan (kontribusi tiap fitur).")

with st.form("pred_form"):
    c1, c2, c3 = st.columns(3)
    tmean = c1.number_input("Suhu rata-rata (°C)", value=28.0, step=0.1)
    tmax  = c2.number_input("Suhu max (°C)",     value=32.0, step=0.1)
    tmin  = c3.number_input("Suhu min (°C)",     value=25.0, step=0.1)

    c4, c5, c6 = st.columns(3)
    ws = c4.number_input("Kecepatan angin max (km/h)", value=15.0, step=0.1)
    wg = c5.number_input("Gust max (km/h)",            value=30.0, step=0.1)
    wd = c6.number_input("Arah angin dominan (°)",     value=180,  step=1)

    c7, c8 = st.columns(2)
    sun = c7.number_input("Durasi sinar matahari (detik)",
                          value=30000, step=100,
                          help="43.200 detik = 12 jam (cerah penuh)")
    month = c8.selectbox("Bulan", list(range(1, 13)), index=5,
                         format_func=lambda m: f"{m} ({['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des'][m-1]})")

    submitted = st.form_submit_button("🔮 Prediksi", use_container_width=True)

    if submitted:
        # --- Prediksi ---
        row   = np.array([[tmean, tmax, tmin, ws, wg, wd, sun, month]])
        row_s = scaler.transform(row)
        pred  = model.predict(row_s)[0]
        proba = model.predict_proba(row_s)[0, 1]

        # --- Hasil teks ---
        st.markdown("### 🎯 Hasil Prediksi")
        if pred == 1:
            st.success(f"🌧️ **HARI HUJAN** — probabilitas {proba:.2%}")
        else:
            st.info(f"☀️ **TIDAK HUJAN** — probabilitas hujan {proba:.2%}")

        # ====================================================
        # GRAFIK 1: GAUGE PROBABILITAS
        # ====================================================
        st.markdown("#### 📊 Keyakinan Model")
        fig, ax = plt.subplots(figsize=(10, 1.8))
        ax.barh(["Hasil"], [1 - proba], color='#4C9BE8', label='Tidak Hujan')
        ax.barh(["Hasil"], [proba], left=[1 - proba], color='#E8744C', label='Hujan')
        ax.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Threshold 0,5')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilitas")
        if (1 - proba) > 0.08:
            ax.text((1 - proba) / 2, 0, f"Tidak Hujan\n{1 - proba:.1%}",
                    ha='center', va='center', color='white', fontweight='bold')
        if proba > 0.08:
            ax.text(1 - proba + proba / 2, 0, f"Hujan\n{proba:.1%}",
                    ha='center', va='center', color='white', fontweight='bold')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3)
        st.pyplot(fig); plt.close()

        # ====================================================
        # GRAFIK 2: KONTRIBUSI FITUR (SHAP-like)
        # ====================================================
        st.markdown("#### 🔍 Mengapa Model Memutuskan Demikian?")
        st.caption("Batang **merah** mendorong prediksi **HUJAN**, "
                   "batang **biru** mendorong **TIDAK HUJAN**. "
                   "Nilai = kontribusi ke log-odds.")

        contribs = model.get_booster().predict(
            DMatrix(row_s), pred_contribs=True
        )[0]
        fitur_contrib = pd.Series(contribs[:-1], index=fitur).sort_values()
        warna = ['#4C9BE8' if v < 0 else '#E8744C' for v in fitur_contrib.values]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(fitur_contrib.index, fitur_contrib.values,
                       color=warna, edgecolor='black', linewidth=0.7)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel("Kontribusi ke log-odds  "
                      "(>0 → dorong HUJAN, <0 → dorong TIDAK HUJAN)",
                      fontweight='bold')
        ax.set_title("Kontribusi Setiap Fitur terhadap Keputusan Model",
                     fontweight='bold')
        for b, v in zip(bars, fitur_contrib.values):
            x_off = 0.005 if v >= 0 else -0.005
            ha    = 'left' if v >= 0 else 'right'
            ax.text(v + x_off, b.get_y() + b.get_height() / 2,
                    f"{v:+.2f}", va='center', ha=ha,
                    fontsize=9, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig); plt.close()

        # ====================================================
        # GRAFIK 3: NILAI INPUT vs RATA-RATA DATASET
        # ====================================================
        st.markdown("#### 📈 Nilai Input vs Rata-rata Dataset")
        df_mean = df[fitur].mean()
        df_std  = df[fitur].std()
        input_vals = pd.Series(
            [tmean, tmax, tmin, ws, wg, wd, sun, month],
            index=fitur
        )
        z_input = (input_vals - df_mean) / df_std

        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(7, 5))
            colors_z = ['#E8744C' if abs(z) > 2 else '#4C9BE8' for z in z_input]
            ax.barh(z_input.index, z_input.values,
                    color=colors_z, edgecolor='black', linewidth=0.7)
            ax.axvline(0, color='black', lw=0.8)
            ax.axvline(-2, color='gray', linestyle='--', lw=1,
                       label='±2 standar deviasi (ekstrem)')
            ax.axvline( 2, color='gray', linestyle='--', lw=1)
            ax.set_xlabel("Z-score (selisih dari rata-rata dalam satuan std)",
                          fontweight='bold')
            ax.set_title("Oranye = Nilai Ekstrem, Biru = Normal",
                         fontweight='bold')
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig); plt.close()
        with col_b:
            tabel = pd.DataFrame({
                "Input Anda": input_vals.round(2),
                "Rata-rata": df_mean.round(2),
                "Std":       df_std.round(2)
            })
            st.dataframe(tabel, use_container_width=True, height=340)

        # ====================================================
        # TABEL PROBABILITAS RINGKAS
        # ====================================================
        st.markdown("#### 📋 Ringkasan Probabilitas")
        cA, cB, cC = st.columns(3)
        cA.metric("P(Hujan)", f"{proba:.2%}")
        cB.metric("P(Tidak Hujan)", f"{(1 - proba):.2%}")
        cC.metric("Prediksi Akhir", "🌧️ Hujan" if pred == 1 else "☀️ Tidak Hujan")

st.markdown("---")
st.caption("Praktikum Analisis Big Data - XGBoost Classifier | "
           "Dataset Cuaca Kolkata 2021–2026")