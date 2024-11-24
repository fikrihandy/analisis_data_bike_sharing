import os
import plotly.express as px
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "main_data.csv")

hour_df = pd.read_csv(file_path)
hour_df["dteday"] = pd.to_datetime(hour_df["dteday"])

# Sidebar filters
st.sidebar.title("Filter Data")
filter_type = st.sidebar.radio(
    "Pilih Jenis Filter:",
    options=["Semua Data", "Filter Berdasarkan"]
)

if filter_type == "Filter Berdasarkan":
    selected_year = st.sidebar.multiselect(
        "Pilih Tahun:",
        options=[2011, 2012],
        default=[2011, 2012]
    )
    selected_season = st.sidebar.multiselect(
        "Pilih Musim:",
        options=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        format_func=lambda x: {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}[x]
    )
    selected_workingday = st.sidebar.radio(
        "Hari Kerja atau Tidak?",
        options=["Semua", "Ya", "Tidak"]
    )

    # Filter data
    filtered_df = hour_df[
        (hour_df["yr"].isin([year - 2011 for year in selected_year])) &
        (hour_df["season"].isin(selected_season))
        ]

    if selected_workingday == "Ya":
        filtered_df = filtered_df[filtered_df["workingday"] == 1]
    elif selected_workingday == "Tidak":
        filtered_df = filtered_df[filtered_df["workingday"] == 0]
else:
    filtered_df = hour_df.copy()

# Display selected data (optional)
st.sidebar.write(f"Jumlah Data: {len(filtered_df)}")

# Visualisasi Feature Importance
st.title("Variabel yang Paling Mempengaruhi Bike Sharing")
st.write(
    "Visualisasi ini menunjukkan variabel apa yang memengaruhi jumlah Bike Sharing dan tingkat keterpengaruhannya.")

df_q1 = filtered_df.copy()
df_q1 = pd.get_dummies(df_q1, columns=["season", "weathersit", "weekday", "mnth", "hr"], drop_first=True)

X = df_q1.drop(columns=["cnt", "casual", "registered", "instant", "dteday"])
y = df_q1["cnt"]

# Hanya gunakan data yang difilter untuk melatih model
if len(X) > 10:  # Pastikan data cukup untuk pelatihan model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Feature importance berdasarkan data yang difilter
    feature_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig = px.bar(
        feature_importances.head(5),
        x="Importance",
        y="Feature",
        orientation="h",
        title="5 Variabel Yang Mempengaruhi Bike Sharing",
        labels={"Importance": "Importance Score", "Feature": "Features"},
        text="Importance",
        height=400
    )

    fig.update_traces(marker_color='royalblue', texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis=dict(title=""), xaxis=dict(title="Importance Score"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Data yang difilter terlalu sedikit untuk melatih model.")

# Pola Penyewaan Sepeda Berdasarkan Waktu dan Kondisi Cuaca
st.title("Pola Penyewaan Sepeda Berdasarkan Waktu dan Kondisi Cuaca")
st.write(
    "Visualisasi ini menunjukkan bagaimana jumlah penyewaan sepeda rata-rata berubah sepanjang hari untuk berbagai kondisi cuaca.")

hourly_weather = filtered_df.groupby(['hr', 'weathersit'])['cnt'].mean().reset_index()
weather_labels = {
    1: "Clear/Partly Cloudy",
    2: "Mist/Cloudy",
    3: "Light Snow/Rain",
    4: "Heavy Rain/Snow"
}
hourly_weather['weathersit'] = hourly_weather['weathersit'].map(weather_labels)

fig, ax = plt.subplots(figsize=(14, 8))
sns.lineplot(
    data=hourly_weather,
    x='hr', y='cnt', hue='weathersit', palette='tab10', ax=ax
)

ax.set_title("Pola Penyewaan Sepeda Berdasarkan Waktu dan Kondisi Cuaca", fontsize=16)
ax.set_xlabel("Jam", fontsize=12)
ax.set_ylabel("Jumlah Penyewaan (Rata-rata)", fontsize=12)
ax.set_xticks(range(0, 24))
ax.legend(title="Kondisi Cuaca", fontsize=10)
ax.grid(True)

st.pyplot(fig)
