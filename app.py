from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from inference import get_forecast, load_desa, load_ramalan, search_location
from utils import (
    format_date_id,
    format_distance,
    format_number,
    get_air_quality_label,
)

APP_ROOT = Path(__file__).resolve().parent
CSS_PATH = APP_ROOT / "assets" / "style.css"

st.set_page_config(
    page_title="ASAP Detektor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    if CSS_PATH.exists():
        st.markdown(
            f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def get_app_data():
    return load_ramalan(str(APP_ROOT / "ramalan_asap.parquet")), load_desa(
        str(APP_ROOT / "desa_kota.csv")
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-block">
                <div class="brand-icon">🌫️</div>
                <div>
                    <div class="brand-title">ASAP DETEKTOR</div>
                    <div class="brand-subtitle">Prediksi Kualitas Udara</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        page = st.radio(
            "Navigasi",
            ["Dashboard", "Tentang Model"],
            label_visibility="collapsed",
        )

        st.markdown("<div class='sidebar-spacer'></div>", unsafe_allow_html=True)
        st.caption("Data forecast diproses dari pipeline model ASAP.")
        return page


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">ASAP DETEKTOR · KALIMANTAN</div>
            <h1>Prediksi Kualitas udara<br><span>hingga 3 hari ke depan.</span></h1>
            <p>
                Cari desa atau kota untuk melihat estimasi konsentrasi udaranya
                berdasarkan grid CAMS terdekat.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_location_search(desa_df):
    st.markdown(
        '<div class="section-label">PILIH LOKASI</div>',
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Cari desa atau kota",
        value="",
        placeholder="Contoh: Ketapang, Sukamara, Pontianak...",
        label_visibility="collapsed",
    )

    if not query.strip():
        st.markdown(
            """
            <div class="empty-search">
                <div class="empty-icon">⌕</div>
                <div class="empty-title">Mulai dengan mencari wilayah</div>
                <div class="empty-text">
                    Ketik nama desa atau kota untuk melihat forecast kualitas udara.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return None

    results = search_location(query, desa_df)

    if results.empty:
        st.warning(
            f'Tidak ada wilayah yang cocok dengan "{query}". '
            "Coba gunakan nama kota/desa yang lebih pendek."
        )
        return None

    options = [
        f"{row['nama']} · {row['jenis']} · {row['provinsi']}"
        for _, row in results.iterrows()
    ]

    selected = st.selectbox(
        "Hasil pencarian",
        options,
        label_visibility="collapsed",
    )
    selected_idx = options.index(selected)
    return results.iloc[selected_idx]


def render_forecast_cards(forecast: dict) -> None:
    horizons = forecast["horizons"]

    st.markdown(
        '<div class="section-label">FORECAST kualitas udara</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3, gap="medium")

    for index, col in enumerate(cols):
        if index >= len(horizons):
            with col:
                st.markdown(
                    '<div class="forecast-card unavailable"><div class="forecast-horizon">DATA</div>'
                    '<div class="forecast-value">—</div>'
                    '<div class="forecast-unit">belum tersedia</div></div>',
                    unsafe_allow_html=True,
                )
            continue

        item = horizons[index]
        value = item["pm25_duga"]
        aq = get_air_quality_label(value)

        value_text = format_number(value, 1) if value is not None else "N/A"
        date_text = format_date_id(item["tanggal_ramal"])

        with col:
            st.markdown(
                f"""
                <div class="forecast-card" style="--aq-color:{aq['color']};">
                    <div class="forecast-top">
                        <span class="forecast-horizon">H+{item['horizon_hari']}</span>
                        <span class="forecast-dot"></span>
                    </div>
                    <div class="forecast-value">{value_text}</div>
                    <div class="forecast-unit">µg/m³ PM2.5</div>
                    <div class="forecast-date">{date_text}</div>
                    <div class="forecast-badge">
                        {aq['emoji']} {aq['label']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chart(forecast: dict) -> None:
    horizons = forecast["horizons"]
    labels = [f"H+{item['horizon_hari']}" for item in horizons]
    dates = [format_date_id(item["tanggal_ramal"]) for item in horizons]
    values = [item["pm25_duga"] for item in horizons]

    st.markdown(
        '<div class="section-label">TREN FORECAST</div>', unsafe_allow_html=True
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=values,
            customdata=[[date] for date in dates],
            mode="lines+markers+text",
            text=[format_number(v, 1) if v is not None else "N/A" for v in values],
            textposition="top center",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "%{customdata[0]}<br>"
                "PM2.5: %{y:.1f} µg/m³"
                "<extra></extra>"
            ),
            line=dict(color="#13795B", width=3),
            marker=dict(size=11, color="#13795B"),
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=15, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#50605A"),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        ),
        yaxis=dict(
            title="µg/m³",
            showgrid=True,
            gridcolor="#E9EFEC",
            zeroline=False,
            fixedrange=True,
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_location_card(desa_row, forecast: dict) -> None:
    st.markdown(
        '<div class="section-label">INFORMASI LOKASI</div>', unsafe_allow_html=True
    )

    col_a, col_b = st.columns([1.1, 1])

    with col_a:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">📍 {desa_row['nama']}</div>
                <div class="info-subtitle">{desa_row['jenis']} · {desa_row['provinsi']}</div>
                <div class="info-grid">
                    <div><span>Latitude</span><b>{float(desa_row['lat']):.4f}</b></div>
                    <div><span>Longitude</span><b>{float(desa_row['lon']):.4f}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">Grid CAMS terdekat</div>
                <div class="distance-value">{format_distance(forecast['distance_km'])}</div>
                <div class="info-grid compact">
                    <div><span>Grid latitude</span><b>{forecast['grid_lat']:.2f}</b></div>
                    <div><span>Grid longitude</span><b>{forecast['grid_lon']:.2f}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        f"Batch model terakhir: {format_date_id(forecast['tanggal_data'])}. "
        "Jarak grid ditampilkan untuk transparansi pemetaan."
    )


def render_disclaimer(forecast: dict | None = None) -> None:
    note = (
        "Nilai PM2.5 merupakan estimasi wilayah dari grid CAMS (~56 km), "
        "bukan hasil pengukuran langsung di desa/kota."
    )

    if forecast:
        horizon_notes = [
            item.get("catatan_wilayah")
            for item in forecast.get("horizons", [])
            if item.get("catatan_wilayah")
        ]
        if horizon_notes:
            note = str(horizon_notes[0])

    st.markdown(
        f"""
        <div class="disclaimer">
            <div class="disclaimer-icon">!</div>
            <div>
                <b>Interpretasi data</b>
                <p>{note}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(desa_df, ramalan_df) -> None:
    render_hero()
    st.markdown("<div class='hero-gap'></div>", unsafe_allow_html=True)

    desa_row = render_location_search(desa_df)
    if desa_row is None:
        return

    with st.spinner("Mengambil forecast wilayah..."):
        forecast = get_forecast(desa_row, ramalan_df)

    if not forecast.get("found"):
        st.error(
            "Forecast untuk grid terdekat belum tersedia. " "Silakan coba lokasi lain."
        )
        return

    st.markdown("<div class='content-gap'></div>", unsafe_allow_html=True)
    render_forecast_cards(forecast)

    st.markdown("<div class='content-gap'></div>", unsafe_allow_html=True)
    render_chart(forecast)

    st.markdown("<div class='content-gap'></div>", unsafe_allow_html=True)
    render_location_card(desa_row, forecast)

    st.markdown("<div class='content-gap-small'></div>", unsafe_allow_html=True)
    render_disclaimer(forecast)


def render_about() -> None:
    st.markdown(
        """
        <div class="hero compact">
            <div class="eyebrow">TENTANG MODEL</div>
            <h1>ASAP Detektor</h1>
            <p>
                Dashboard untuk membaca hasil forecast PM2.5 dari pipeline ASAP
                dan menyajikannya pada tingkat desa/kota secara transparan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "LightGBM")
    c2.metric("Forecast", "H+1 · H+2 · H+3")
    c3.metric("Feature version", "asap-v1")

    st.markdown("""
        ### Cara kerja

        1. Model ASAP menghasilkan forecast PM2.5 pada grid CAMS.
        2. Aplikasi mencari grid CAMS terdekat dari koordinat desa/kota.
        3. Nilai H+1, H+2, dan H+3 dari grid tersebut ditampilkan.
        4. Jarak desa ke grid selalu ditampilkan agar pengguna mengetahui konteks spasialnya.

        ### Batasan penting

        Data pada aplikasi adalah **estimasi wilayah** dari grid CAMS (~56 km).
        Nilai tersebut bukan sensor PM2.5 yang diukur langsung di titik desa/kota.
        Aplikasi juga tidak melakukan training ulang model.
        """)


inject_css()

try:
    ramalan_df, desa_df = get_app_data()
except Exception as exc:
    st.error("Data aplikasi belum siap.")
    st.code(str(exc))
    st.info(
        "Pastikan `ramalan_asap.parquet` dan `desa_kota.csv` berada di folder yang sama "
        "dengan `app.py`."
    )
    st.stop()

page = render_sidebar()

if page == "Dashboard":
    render_dashboard(desa_df, ramalan_df)
else:
    render_about()
