import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
from kaya_core import load_data, decompose_all, period_summary, CLUSTERS, COUNTRY_TO_CLUSTER, COMPONENT_COLORS, COMPONENT_LABELS

st.set_page_config(page_title="Kaya Decomposition Explorer", page_icon="🌍", layout="wide")

st.markdown("""
<style>
section[data-testid="stSidebar"] { background-color: #1A2E3B; }
section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
h1 { color: #1A2E3B !important; }
h2 { color: #0090CA !important; font-size: 1.3rem !important; }
[data-testid="metric-container"] { background: #D6EEF8; border-radius: 8px; padding: 12px; border: 1px solid #B8D8EE; }
.footer { color: #6B7F8A; font-size: 0.8rem; margin-top: 2rem; border-top: 1px solid #DEE9EE; padding-top: 0.8rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Načítám data…")
def get_data():
    df = load_data()
    decomps = decompose_all(df)
    return df, decomps

df_raw, decomps = get_data()
all_countries = sorted(decomps.keys())

with st.sidebar:
    st.markdown("## 🌍 Kaya Explorer")
    st.markdown("**Fakta o klimatu** · Petra Grešlová")
    st.markdown("---")
    view = st.radio("Pohled", ["🔍 Jednotlivé země", "🗺️ Přehled klastrů", "📊 Srovnání zemí"])
    st.markdown("---")
    y1, y2 = st.slider("Období", 1991, 2022, (1991, 2022), 1)
    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem;color:#6B9AAA'>Zdroj: Our World in Data<br>(Global Carbon Project + World Bank)<br>CC BY 4.0</div>", unsafe_allow_html=True)

def plot_kaya(ax, data, country, y1, y2):
    data = data[(data.index >= y1) & (data.index <= y2)].copy()
    if data.empty:
        ax.text(0.5, 0.5, "Nedostatek dat", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(country, fontsize=10, fontweight="bold")
        return
    x = data.index.values
    pos_b = np.zeros(len(x))
    neg_b = np.zeros(len(x))
    for col in ["pop","gdp_per_cap","energy_per_gdp","co2_per_energy","interactions"]:
        v = data[col].values
        pv, nv = np.where(v>0,v,0), np.where(v<0,v,0)
        ax.bar(x, pv, bottom=pos_b, color=COMPONENT_COLORS.get(col,"#BBBBBB"), width=0.75, linewidth=0)
        ax.bar(x, nv, bottom=neg_b, color=COMPONENT_COLORS.get(col,"#BBBBBB"), width=0.75, linewidth=0)
        pos_b += pv; neg_b += nv
    ax.scatter(x, data["co2_total"].values, color="#1A2E3B", s=18, zorder=5)
    ax.axhline(0, color="#AAAAAA", linewidth=0.8)
    cid = COUNTRY_TO_CLUSTER.get(country)
    tc = CLUSTERS[cid]["color"] if cid else "#1A2E3B"
    ax.set_title(country, fontsize=10, fontweight="bold", color=tc, pad=3)
    ax.set_xlim(y1-1, y2+1)
    vals = data[["pop","gdp_per_cap","energy_per_gdp","co2_per_energy","co2_total"]].values
    maxv = min(35, max(14, np.nanmax(np.abs(vals))*1.3))
    ax.set_ylim(-maxv, maxv)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xticks([yr for yr in range(y1, y2+1, 5)])
    ax.set_xticklabels([str(yr) for yr in range(y1, y2+1, 5)], fontsize=7)
    ax.set_ylabel("%/rok", fontsize=7, labelpad=1)
    ax.set_facecolor("#F8FAFA")

def make_legend():
    patches = [mpatches.Patch(color=COMPONENT_COLORS[k], label=COMPONENT_LABELS[k]) for k in COMPONENT_COLORS]
    patches.append(plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#1A2E3B", markersize=6, label="Celková změna CO₂"))
    return patches

if view == "🔍 Jednotlivé země":
    st.title("🌍 Kaya Decomposition Explorer")
    st.markdown("**Analýza hybatelů CO₂ emisí** · Our World in Data · 1990–2022")
    col_sel, col_info = st.columns([2,1])
    with col_sel:
        selected = st.multiselect("Vyber 1–8 zemí", all_countries, default=["Czechia","Germany","China","India"], max_selections=8)
    with col_info:
        if len(selected) == 1:
            c = selected[0]
            cid = COUNTRY_TO_CLUSTER.get(c)
            if cid: st.info(f"**Klastr:** {CLUSTERS[cid]['name']}")
            r = decomps[c]
            sub = r[(r.index>=y1)&(r.index<=y2)]
            if not sub.empty:
                st.metric("Průměrná změna CO₂/rok", f"{sub['co2_total'].mean():+.1f} %")
    if selected:
        n = len(selected)
        ncols = min(n, 4)
        nrows = (n+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 4.5*nrows))
        fig.patch.set_facecolor("white")
        axes_flat = [axes] if n==1 else list(axes) if nrows==1 else [ax for row in axes for ax in row]
        for i, country in enumerate(selected):
            if country in decomps: plot_kaya(axes_flat[i], decomps[country], country, y1, y2)
        for j in range(len(selected), len(axes_flat)): axes_flat[j].set_visible(False)
        fig.legend(handles=make_legend(), loc="lower center", ncol=6, fontsize=9, frameon=True, bbox_to_anchor=(0.5,0.0), title="Složky změny CO₂", title_fontsize=9)
        plt.tight_layout(rect=[0,0.06,1,1])
        st.pyplot(fig); plt.close()

elif view == "🗺️ Přehled klastrů":
    st.title("🗺️ Klastry zemí podle Kaya dekompozice")
    st.markdown(f"Průměrné meziroční příspěvky k změně CO₂, **{y1}–{y2}**")
    summary = period_summary(decomps, y1, y2)
    st.markdown("## Efektivita vs. dekarbonizace energetiky")
    fig, ax = plt.subplots(figsize=(12,7))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#F8FAFA")
    ax.grid(True, color="white", linewidth=1.0, zorder=0)
    ax.axhline(0, color="#CCCCCC", linewidth=1.0); ax.axvline(0, color="#CCCCCC", linewidth=1.0)
    highlight = {"United States","Germany","China","India","Russia","Brazil","Vietnam","Ethiopia","Nigeria","Poland","Ukraine","Denmark","South Korea","Indonesia","Iran","Bangladesh","Saudi Arabia", "Czechia"}
    for cid, info in CLUSTERS.items():
        sub = summary[summary["cluster"]==cid]
        ax.scatter(sub["energy_per_gdp"], sub["co2_per_energy"], color=info["color"], s=65, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.6, label=info["name"])
        for _, row in sub[sub["country"].isin(highlight)].iterrows():
            ax.annotate(row["country"], xy=(row["energy_per_gdp"], row["co2_per_energy"]), xytext=(row["energy_per_gdp"]+0.15, row["co2_per_energy"]+0.15), fontsize=7.5, color=info["color"], fontweight="bold")
    ax.set_xlabel("Průměrná změna Energie/HDP (%/rok)  ←  negativní = efektivnější ekonomika", fontsize=10, color="#444")
    ax.set_ylabel("Průměrná změna CO₂/Energie (%/rok)  ←  negativní = čistší mix", fontsize=10, color="#444")
    ax.set_title(f"Klastry zemí: efektivita vs. dekarbonizace  |  {y1}–{y2}", fontsize=12, fontweight="bold", color="#1A2E3B")
    ax.legend(fontsize=9, frameon=True, edgecolor="#DDD", loc="lower right")
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("## Kaya grafy podle klastru")
    sel_cluster = st.selectbox("Vyber klastr", list(CLUSTERS.keys()), format_func=lambda x: f"Klastr {x}: {CLUSTERS[x]['name']}")
    cluster_countries = [c for c in CLUSTERS[sel_cluster]["countries"] if c in decomps]
    ncols = 4; nrows = (len(cluster_countries)+ncols-1)//ncols
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 4.5*nrows))
    fig3.patch.set_facecolor("white")
    axes3_flat = axes3.flatten() if nrows>1 else list(axes3) if ncols>1 else [axes3]
    for i, country in enumerate(cluster_countries): plot_kaya(axes3_flat[i], decomps[country], country, y1, y2)
    for j in range(len(cluster_countries), len(axes3_flat)): axes3_flat[j].set_visible(False)
    fig3.legend(handles=make_legend(), loc="lower center", ncol=6, fontsize=9, frameon=True, bbox_to_anchor=(0.5,0.0), title="Složky změny CO₂", title_fontsize=9)
    plt.tight_layout(rect=[0,0.06,1,1]); st.pyplot(fig3); plt.close()

else:
    st.title("📊 Srovnání zemí — průměrné příspěvky")
    summary = period_summary(decomps, y1, y2)
    filter_cluster = st.multiselect("Filtrovat podle klastru", list(CLUSTERS.keys()), format_func=lambda x: f"Klastr {x}: {CLUSTERS[x]['name']}", default=list(CLUSTERS.keys()))
    if filter_cluster: summary = summary[summary["cluster"].isin(filter_cluster)]
    top_n = st.slider("Počet zemí", 10, len(summary), min(40, len(summary)), 5)
    summary_sorted = pd.concat([summary.nlargest(top_n//2,"co2_total"), summary.nsmallest(top_n//2,"co2_total")]).drop_duplicates().sort_values("co2_total", ascending=True)
    fig, ax = plt.subplots(figsize=(13, max(6, len(summary_sorted)*0.42)))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#F8FAFA")
    ax.axvline(0, color="#AAAAAA", linewidth=1.0); ax.grid(axis="x", color="white", linewidth=1.0, zorder=0)
    y_pos = range(len(summary_sorted))
    pos_b = np.zeros(len(summary_sorted)); neg_b = np.zeros(len(summary_sorted))
    for col in ["pop","gdp_per_cap","energy_per_gdp","co2_per_energy"]:
        v = summary_sorted[col].values
        pv, nv = np.where(v>0,v,0), np.where(v<0,v,0)
        ax.barh(list(y_pos), pv, left=pos_b, color=COMPONENT_COLORS[col], height=0.72, linewidth=0)
        ax.barh(list(y_pos), nv, left=neg_b, color=COMPONENT_COLORS[col], height=0.72, linewidth=0)
        pos_b += pv; neg_b += nv
    ax.scatter(summary_sorted["co2_total"].values, list(y_pos), color="#1A2E3B", s=25, zorder=5)
    ytick_colors = [CLUSTERS[int(r["cluster"])]["color"] if pd.notna(r.get("cluster")) else "#1A2E3B" for _,r in summary_sorted.iterrows()]
    ax.set_yticks(list(y_pos)); ax.set_yticklabels(summary_sorted["country"].values, fontsize=8.5)
    for tick, color in zip(ax.get_yticklabels(), ytick_colors): tick.set_color(color)
    ax.set_xlabel("Průměrná meziroční změna (%/rok)", fontsize=10, color="#444")
    ax.set_title(f"Průměrné příspěvky k CO₂ změně  |  {y1}–{y2}", fontsize=11, fontweight="bold", color="#1A2E3B")
    legend_patches = [mpatches.Patch(color=COMPONENT_COLORS[k], label=COMPONENT_LABELS[k]) for k in ["pop","gdp_per_cap","energy_per_gdp","co2_per_energy"]]
    legend_patches.append(plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#1A2E3B", markersize=6, label="Celková CO₂ změna"))
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8.5, frameon=True)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    with st.expander("📋 Zobrazit tabulku dat"):
        display = summary_sorted[["country","cluster","co2_total","pop","gdp_per_cap","energy_per_gdp","co2_per_energy"]].copy()
        display.columns = ["Země","Klastr","CO₂ celkem","Populace","HDP/os.","Energie/HDP","CO₂/Energie"]
        st.dataframe(display.round(2), use_container_width=True)

st.markdown("<div class='footer'>Zdroj: Our World in Data · Petra Grešlová · Výběrové řízení Analytik/čka, Fakta o klimatu, 2026</div>", unsafe_allow_html=True)
