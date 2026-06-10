import pandas as pd
import numpy as np

CLUSTERS = {
    1: {"name": "Decarbonizing Advanced Economies", "color": "#0090CA",
        "countries": ["United States","Canada","Germany","France","United Kingdom","Spain","Italy","Netherlands","Norway","Sweden","Denmark","Finland","Belgium","Austria","Switzerland","Portugal","Greece","Ireland","Iceland","New Zealand","Australia","Japan","South Korea"]},
    2: {"name": "Rostouci asijske ekonomiky", "color": "#E63B2E",
        "countries": ["China","India","Vietnam","Bangladesh","Indonesia","Philippines","Myanmar","Thailand","Malaysia","Sri Lanka","Cambodia","Laos","Nepal","Mongolia","Taiwan","Hong Kong","Pakistan"]},
    3: {"name": "Exporteri fosilnich paliv", "color": "#F4A11D",
        "countries": ["Saudi Arabia","Qatar","Kuwait","Oman","Iraq","Iran","Turkmenistan","Algeria","Libya","Azerbaijan","Trinidad and Tobago","Bahrain"]},
    4: {"name": "Postsovetska transformace", "color": "#6B3FA0",
        "countries": ["Russia","Ukraine","Poland","Czechia","Romania","Bulgaria","Estonia","Belarus","Hungary","Slovakia","Slovenia","Croatia","Latvia","Lithuania","North Macedonia","Albania","Kazakhstan","Uzbekistan"]},
    5: {"name": "Afrika a Latinska Amerika", "color": "#3BAE2A",
        "countries": ["Ethiopia","Nigeria","Tanzania","Ghana","Cameroon","Uganda","Zambia","Mozambique","Mali","Niger","Burkina Faso","Senegal","Kenya","Angola","Congo","Gabon","Zimbabwe","Egypt","Morocco","Tunisia","Jordan","Lebanon","Syria","Yemen","Brazil","Colombia","Chile","Argentina","Mexico","Peru","Ecuador","Bolivia","Venezuela","Uruguay","Cuba","Dominican Republic","Guatemala","Honduras","El Salvador","Panama","Nicaragua","Costa Rica","Jamaica","Paraguay","South Africa","Singapore"]},
}

COUNTRY_TO_CLUSTER = {c: cid for cid, info in CLUSTERS.items() for c in info["countries"]}

COMPONENT_COLORS = {
    "pop": "#C7E6C8", "gdp_per_cap": "#F4A5B8",
    "energy_per_gdp": "#88C9A1", "co2_per_energy": "#AAD4E8", "interactions": "#BBBBBB",
}
COMPONENT_LABELS = {
    "pop": "Populace", "gdp_per_cap": "HDP/obyvatele",
    "energy_per_gdp": "Energie/HDP", "co2_per_energy": "CO2/Energie", "interactions": "Interakce",
}

DATA_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
LOCAL_PATH = "data/owid_co2.csv"

def load_data(path=LOCAL_PATH):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, path)
        df = pd.read_csv(path)
    cols = ["country","year","co2","population","gdp","primary_energy_consumption"]
    df = df[cols].copy()
    df = df[df["year"].between(1990, 2022)]
    return df

def decompose(country_df):
    d = country_df.sort_values("year").copy().set_index("year")
    d = d.reindex(range(int(d.index.min()), int(d.index.max())+1))
    d = d.select_dtypes(include="number").interpolate(limit=2)
    d["gdp_per_cap"]     = d["gdp"] / d["population"]
    d["energy_per_gdp_"] = d["primary_energy_consumption"] / (d["gdp"] / 1e9)
    d["co2_per_energy"]  = d["co2"] / d["primary_energy_consumption"]
    def ld(s): return np.log(s).diff() * 100
    r = pd.DataFrame(index=d.index)
    r["co2_total"]      = ld(d["co2"])
    r["pop"]            = ld(d["population"])
    r["gdp_per_cap"]    = ld(d["gdp_per_cap"])
    r["energy_per_gdp"] = ld(d["energy_per_gdp_"])
    r["co2_per_energy"] = ld(d["co2_per_energy"])
    r["interactions"]   = r["co2_total"] - r["pop"] - r["gdp_per_cap"] - r["energy_per_gdp"] - r["co2_per_energy"]
    r["co2_level"]      = d["co2"]
    return r.dropna(subset=["co2_total"])

def decompose_all(df):
    results = {}
    for country, grp in df.groupby("country"):
        try:
            r = decompose(grp)
            if len(r) >= 5:
                results[country] = r
        except Exception:
            pass
    return results

def period_summary(decomps, y1, y2):
    rows = []
    for country, r in decomps.items():
        sub = r[(r.index >= y1) & (r.index <= y2)]
        if len(sub) < 3:
            continue
        rows.append({
            "country": country,
            "cluster": COUNTRY_TO_CLUSTER.get(country),
            "co2_total": sub["co2_total"].mean(),
            "pop": sub["pop"].mean(),
            "gdp_per_cap": sub["gdp_per_cap"].mean(),
            "energy_per_gdp": sub["energy_per_gdp"].mean(),
            "co2_per_energy": sub["co2_per_energy"].mean(),
        })
    return pd.DataFrame(rows)
