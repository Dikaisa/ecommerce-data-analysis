import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="E-Commerce Dashboard", page_icon="🛒", layout="wide")
sns.set_theme(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRIMARY, ACCENT, MUTED = "#1f4e79", "#c0392b", "#b0c4de"


# ----------------------------- Load data -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        os.path.join(BASE_DIR, "main_data.csv"),
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"],
    )
    geo = pd.read_csv(os.path.join(BASE_DIR, "geo_state.csv"))
    return df, geo


def create_monthly_df(df):
    monthly = (df.set_index("order_purchase_timestamp")
               .resample("MS")
               .agg({"order_id": "nunique", "price": "sum"})
               .rename(columns={"order_id": "orders", "price": "revenue"})
               .reset_index())
    return monthly


def create_category_df(df):
    return (df.groupby("product_category_name_english")
            .agg(revenue=("price", "sum"), orders=("order_id", "nunique"))
            .sort_values("revenue", ascending=False)
            .reset_index())


def create_order_level_df(df):
    return df.drop_duplicates("order_id")


def create_rfm_df(df):
    snapshot = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (df.groupby("customer_unique_id")
           .agg(last_purchase=("order_purchase_timestamp", "max"),
                frequency=("order_id", "nunique"),
                monetary=("price", "sum"))
           .reset_index())
    rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days

    # Skor recency & monetary dengan binning kuantil (sama seperti notebook)
    def qscore(series, labels):
        try:
            return pd.qcut(series, 5, labels=labels).astype(int)
        except ValueError:  # data hasil filter terlalu sedikit/banyak nilai sama
            return pd.qcut(series.rank(method="first"), 5, labels=labels).astype(int)

    rfm["R_score"] = qscore(rfm["recency"], [5, 4, 3, 2, 1])
    rfm["M_score"] = qscore(rfm["monetary"], [1, 2, 3, 4, 5])
    rfm["F_score"] = pd.cut(rfm["frequency"], bins=[0, 1, 2, np.inf], labels=[1, 3, 5]).astype(int)

    conditions = [
        (rfm["F_score"] >= 3) & (rfm["R_score"] >= 4),
        rfm["F_score"] >= 3,
        (rfm["R_score"] >= 4) & (rfm["M_score"] >= 4),
        rfm["R_score"] >= 4,
        rfm["R_score"] >= 2,
    ]
    choices = ["Loyal Customer", "Repeat - At Risk", "New High Spender", "New Customer", "Need Attention"]
    rfm["segment"] = np.select(conditions, choices, default="Lost Customer")
    return rfm


def create_state_df(df, geo):
    order_level = create_order_level_df(df)
    state = (df.groupby("customer_state")
             .agg(customers=("customer_unique_id", "nunique"), revenue=("price", "sum"))
             .join(order_level.groupby("customer_state")["is_late"].mean().rename("late_rate"))
             .reset_index()
             .merge(geo, left_on="customer_state", right_on="state", how="left"))
    return state


all_df, geo_df = load_data()

# ----------------------------- Sidebar -----------------------------
min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()

with st.sidebar:
    st.title("🛒 E-Commerce Dashboard")
    st.caption("Olist Brazilian E-Commerce, pesanan delivered")
    date_range = st.date_input("Rentang waktu", value=[min_date, max_date],
                               min_value=min_date, max_value=max_date)
    states = sorted(all_df["customer_state"].unique())
    selected_states = st.multiselect("Filter state (kosongkan = semua)", states)
    st.markdown("---")
    st.caption("Dibuat oleh Muhammad Dikaisa Ibnu Amin")

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

main_df = all_df[(all_df["order_purchase_timestamp"].dt.date >= start_date) &
                 (all_df["order_purchase_timestamp"].dt.date <= end_date)]
if selected_states:
    main_df = main_df[main_df["customer_state"].isin(selected_states)]

if main_df.empty:
    st.warning("Tidak ada data pada filter yang dipilih. Silakan ubah rentang waktu atau state.")
    st.stop()

order_level_df = create_order_level_df(main_df)

# ----------------------------- Header & KPI -----------------------------
st.title("Dashboard Analisis Penjualan E-Commerce")
st.caption(f"Periode: {start_date:%d %b %Y} – {end_date:%d %b %Y}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pesanan", f"{order_level_df['order_id'].nunique():,}")
col2.metric("Total Revenue", f"BRL {main_df['price'].sum():,.0f}")
col3.metric("Rata-rata Skor Review", f"{order_level_df['review_score'].mean():.2f}")
col4.metric("Pesanan Terlambat", f"{order_level_df['is_late'].mean():.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Tren Bulanan", "🏷️ Kategori Produk",
                                  "🚚 Pengiriman & Review", "👥 Pelanggan (RFM & Wilayah)"])

# ----------------------------- Tab 1: Monthly trend -----------------------------
with tab1:
    st.subheader("Tren Jumlah Pesanan dan Revenue per Bulan")
    monthly_df = create_monthly_df(main_df)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(monthly_df["order_purchase_timestamp"], monthly_df["orders"], marker="o", color=PRIMARY, linewidth=2)
    ax1.set_ylabel("Jumlah Pesanan", color=PRIMARY)
    ax1.set_ylim(0, monthly_df["orders"].max() * 1.15)
    ax2 = ax1.twinx()
    ax2.bar(monthly_df["order_purchase_timestamp"], monthly_df["revenue"], width=20, alpha=0.25, color="#7f8c8d")
    ax2.set_ylabel("Revenue (BRL)", color="#7f8c8d")
    ax2.set_ylim(0, monthly_df["revenue"].max() * 1.15)
    ax2.grid(False)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    st.pyplot(fig)

    peak = monthly_df.loc[monthly_df["orders"].idxmax()]
    st.info(f"Puncak pesanan terjadi pada **{peak['order_purchase_timestamp']:%B %Y}** "
            f"dengan **{int(peak['orders']):,}** pesanan.")

# ----------------------------- Tab 2: Category -----------------------------
with tab2:
    st.subheader("Kategori Produk dengan Revenue Tertinggi dan Terendah")
    n = st.slider("Jumlah kategori yang ditampilkan", 3, 10, 5)
    category_df = create_category_df(main_df)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    top = category_df.head(n)
    bottom = category_df.tail(n).sort_values("revenue")
    ax[0].barh(top["product_category_name_english"][::-1], top["revenue"][::-1],
               color=[MUTED] * (len(top) - 1) + [PRIMARY])
    ax[0].set_title(f"{n} Kategori Revenue Tertinggi")
    ax[0].set_xlabel("Revenue (BRL)")
    ax[1].barh(bottom["product_category_name_english"][::-1], bottom["revenue"][::-1],
               color=["#f5b7b1"] * (len(bottom) - 1) + [ACCENT])
    ax[1].set_title(f"{n} Kategori Revenue Terendah")
    ax[1].set_xlabel("Revenue (BRL)")
    for a in ax:
        a.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("Lihat tabel lengkap"):
        st.dataframe(category_df, width="stretch")

# ----------------------------- Tab 3: Delivery & review -----------------------------
with tab3:
    st.subheader("Pengaruh Keterlambatan Pengiriman terhadap Skor Review")
    late_review = (order_level_df.groupby("is_late")["review_score"].mean()
                   .rename(index={False: "Tepat waktu", True: "Terlambat"}))

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(late_review.index, late_review.values,
                      color=[PRIMARY if i == "Tepat waktu" else ACCENT for i in late_review.index])
        ax.bar_label(bars, fmt="%.2f")
        ax.set_ylim(0, 5)
        ax.set_ylabel("Rata-rata Skor Review (1–5)")
        ax.set_title("Rata-rata Skor Review")
        st.pyplot(fig)
    with c2:
        state_late = (order_level_df.groupby("customer_state")["is_late"].mean()
                      .sort_values(ascending=False).head(10))
        avg_late = order_level_df["is_late"].mean()
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = [ACCENT] * 3 + ["#f5b7b1"] * (len(state_late) - 3)
        ax.barh(state_late.index[::-1], state_late.values[::-1], color=colors[:len(state_late)][::-1])
        ax.axvline(avg_late, color=PRIMARY, linestyle="--", label=f"Rata-rata ({avg_late:.1%})")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title("10 State dengan Keterlambatan Tertinggi")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    if len(late_review) == 2:
        st.info(f"Pesanan terlambat mendapat skor review rata-rata **{late_review['Terlambat']:.2f}**, "
                f"lebih rendah **{late_review['Tepat waktu'] - late_review['Terlambat']:.2f} poin** "
                f"dibanding pesanan tepat waktu.")

# ----------------------------- Tab 4: RFM & geospatial -----------------------------
with tab4:
    st.subheader("Segmentasi Pelanggan (RFM Analysis)")
    rfm_df = create_rfm_df(main_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rata-rata Recency (hari)", f"{rfm_df['recency'].mean():.0f}")
    c2.metric("Rata-rata Frequency", f"{rfm_df['frequency'].mean():.2f}")
    c3.metric("Rata-rata Monetary", f"BRL {rfm_df['monetary'].mean():,.2f}")

    seg = rfm_df["segment"].value_counts().sort_values()
    highlight = {"Loyal Customer": PRIMARY, "New High Spender": "#2e86c1", "Repeat - At Risk": ACCENT}
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(seg.index, seg.values, color=[highlight.get(s, "#b0b0b0") for s in seg.index])
    ax.bar_label(bars, labels=[f" {v:,} ({v / seg.sum():.1%})" for v in seg.values])
    ax.set_xlim(0, seg.max() * 1.25)
    ax.set_xlabel("Jumlah Pelanggan")
    ax.set_title("Jumlah Pelanggan per Segmen")
    st.pyplot(fig)

    st.subheader("Sebaran Pelanggan dan Revenue per State")
    state_df = create_state_df(main_df, geo_df).dropna(subset=["lat", "lng"])
    state_df["size"] = 20000 + 230000 * np.sqrt(state_df["revenue"] / state_df["revenue"].max())
    st.map(state_df, latitude="lat", longitude="lng", size="size", color="#1f4e79aa")

    st.dataframe(
        state_df[["customer_state", "customers", "revenue", "late_rate"]]
        .sort_values("revenue", ascending=False)
        .rename(columns={"customer_state": "State", "customers": "Pelanggan",
                         "revenue": "Revenue (BRL)", "late_rate": "% Terlambat"})
        .style.format({"Revenue (BRL)": "{:,.0f}", "% Terlambat": "{:.1%}", "Pelanggan": "{:,}"}),
        width="stretch", hide_index=True,
    )

st.caption("Copyright © 2026 Muhammad Dikaisa Ibnu Amin – Proyek Akhir Dicoding: Belajar Fundamental Analisis Data")
