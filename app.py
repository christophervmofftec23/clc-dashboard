#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:39:06 2026

@author: christophervelmor
"""


import io
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import folium
from shapely.geometry import LineString
from streamlit_folium import st_folium


st.set_page_config(page_title="CL Circular | Riesgo + Farma", layout="wide")

st.title("CL Circular — Riesgo de robo a transportista (SNSP) + Capacidad farma (DENUE)")
st.caption("Dashboard estatal (2015–2025) + pronóstico 12 meses. Enfocado a operación logística pharma (SCIAN 3254).")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding-left: 4px;
        padding-right: 4px;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# RUTAS / PATHS
# =========================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MARKET_FOLDER = DATA_DIR

STATES_GEOJSON = DATA_DIR / "states.geojson"
GEOJSON_MUN = DATA_DIR / "mg_2025_integrado" / "mun_00mun_light.geojson"
MUN_CATALOGO = DATA_DIR / "mg_2025_integrado" / "mun_catalogo.csv"
MUN_CENTROIDES = DATA_DIR / "mg_2025_integrado" / "mun_centroides.csv"

MUN_RISK_12M = DATA_DIR / "snsp_rt_municipal_12m_jun2024_may2025_OKCVEGEO.csv"
SERIE_MX = DATA_DIR / "snsp_rt_mensual_mx_2015_2025.csv"
SERIE_STATE = DATA_DIR / "snsp_rt_mensual_estado_2015_2025.csv"
DENUE_CLEAN = DATA_DIR / "INEGI_DENUE_3254_LIMPIO_PRETTY.csv"



# =========================================================
# HELPERS GENERALES
# =========================================================
def norm_txt(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def fix_cdmx(s: pd.Series) -> pd.Series:
    return s.replace({
        "Distrito Federal": "Ciudad de México",
        "CDMX": "Ciudad de México",
    })


def normalize_estado_name(x: str) -> str:
    x = str(x).strip()
    mapping = {
        "Distrito Federal": "Ciudad de México",
        "CDMX": "Ciudad de México",
        "Estado de México": "México",
        "Mexico": "México",
        "Nuevo Leon": "Nuevo León",
        "Michoacan de Ocampo": "Michoacán de Ocampo",
        "Queretaro": "Querétaro",
        "San Luis Potosi": "San Luis Potosí",
        "Yucatan": "Yucatán",
        "Veracruz de Ignacio de la Llave": "Veracruz de Ignacio de la Llave",
        "Coahuila de Zaragoza": "Coahuila de Zaragoza",
    }
    return mapping.get(x, x)


def fmt_riesgo(x) -> str:
    try:
        x = float(x)
    except Exception:
        return ""
    return str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:.2f}"


def minmax_0_10(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mn, mx = x.min(), x.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0.0] * len(x), index=x.index, dtype="float")
    return 10 * (x - mn) / (mx - mn)


def pct_change(vf, va):
    vf = pd.to_numeric(pd.Series([vf]), errors="coerce").iloc[0]
    va = pd.to_numeric(pd.Series([va]), errors="coerce").iloc[0]
    if pd.isna(vf) or pd.isna(va) or va == 0:
        return None
    return (vf - va) / va * 100


def fmt_pp(x):
    if x is None or pd.isna(x):
        return "N/D"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}%"


def bounds_for_estado_states(gdf_states, estado_name):
    gsub = gdf_states[gdf_states["Estado"] == estado_name]
    if len(gsub) == 0:
        return None
    b = gsub.to_crs(epsg=4326).total_bounds
    return [[b[1], b[0]], [b[3], b[2]]]


def pick_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None


def format_compact_number(value):
    if pd.isna(value):
        return "N/A"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def format_month_label(dt_value):
    if pd.isna(dt_value):
        return "N/A"
    ts = pd.to_datetime(dt_value)
    meses = {
        1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
    }
    return f"{meses[ts.month]} {ts.year}"


def apply_boardroom_theme(fig, yaxis_money=True):
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="-apple-system, system-ui, sans-serif", color="#0f172a"),
        margin=dict(l=10, r=10, t=80, b=10),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=0,
            title=None,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(148, 163, 184, 0.0)",
            borderwidth=0,
            font=dict(size=11, color="#0f172a"),
        ),
        xaxis=dict(
            showgrid=False,
            linecolor="rgba(148, 163, 184, 0.7)",
            tickfont=dict(size=10, color="#6b7280"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.25)",
            zeroline=False,
            tickfont=dict(size=10, color="#6b7280"),
        ),
    )

    if yaxis_money:
        fig.update_yaxes(tickformat=",.0f")

    return fig


def resolve_first_match(folder: Path, patterns):
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def clean_excel_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


# =========================================================
# SCORE DE PLANTA
# =========================================================
def estrato_to_range(txt):
    txt = str(txt).strip().lower()
    if not txt or txt in {"nan", "none"}:
        return (None, None)
    if "251 y más" in txt or "251 y mas" in txt:
        return (251, None)

    import re
    nums = re.findall(r"\d+", txt)
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    if len(nums) == 1:
        n = int(nums[0])
        return (n, n)
    return (None, None)


def parse_estrato_score(df):
    out = df.copy()
    mins, maxs = [], []

    col = "Descripcion estrato personal ocupado"
    vals = out[col].fillna("") if col in out.columns else pd.Series([""] * len(out), index=out.index)

    for v in vals:
        mn, mx = estrato_to_range(v)
        mins.append(mn)
        maxs.append(mx)

    out["Estrato_min"] = mins
    out["Estrato_max"] = maxs

    def _score(row):
        mn = row["Estrato_min"]
        mx = row["Estrato_max"]

        if pd.isna(mn) and pd.isna(mx):
            return 0.0
        if mn == 251:
            return 10.0
        if mx == 5:
            return 1.0
        if mn == 6 and mx == 10:
            return 2.0
        if mn == 11 and mx == 30:
            return 4.0
        if mn == 31 and mx == 50:
            return 5.5
        if mn == 51 and mx == 100:
            return 7.0
        if mn == 101 and mx == 250:
            return 8.5
        return 3.0

    out["EstratoScore"] = out.apply(_score, axis=1)
    return out


def add_plant_score(plants_df, df_state_local):
    plants = plants_df.copy()

    if "Estado" not in plants.columns and "Entidad federativa" in plants.columns:
        plants["Estado"] = plants["Entidad federativa"].astype(str).map(normalize_estado_name)

    if "Estado" in plants.columns:
        state_aux = df_state_local[["Estado", "Riesgo_12m_0_10", "Potencial_Vulnerable_0_10", "Incidentes_12m"]].copy()
        plants = plants.merge(state_aux, on="Estado", how="left")

    plants = parse_estrato_score(plants)

    plants["Riesgo_12m_0_10"] = pd.to_numeric(plants.get("Riesgo_12m_0_10", 0), errors="coerce").fillna(0)
    plants["EstratoScore"] = pd.to_numeric(plants.get("EstratoScore", 0), errors="coerce").fillna(0)

    plants["Score_planta_0_10"] = (0.70 * plants["Riesgo_12m_0_10"] + 0.30 * plants["EstratoScore"]).round(2)
    plants["Cálculo_score_planta"] = "70% riesgo estatal + 30% estrato personal ocupado"

    sort_cols = [c for c in ["Score_planta_0_10", "EstratoScore", "Estado", "Municipio", "Nombre de la Unidad Económica"] if c in plants.columns]
    asc = [False, False, True, True, True][:len(sort_cols)]
    plants = plants.sort_values(sort_cols, ascending=asc).reset_index(drop=True)

    return plants


def build_plants_export(plants_scored):
    out = plants_scored.copy()

    if "Nombre de la vialidad" in out.columns:
        out["Dirección"] = (
            out["Nombre de la vialidad"].fillna("").astype(str).str.strip()
            + " "
            + out.get("Número exterior o kilómetro", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
            + " "
            + out.get("Número interior", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
            + ", "
            + out.get("Nombre de asentamiento humano", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
        ).str.replace(r"\s+", " ", regex=True).str.replace(r"\s+,", ",", regex=True).str.strip(" ,")

    rename_map = {
        "Descripcion estrato personal ocupado": "Estrato personal ocupado",
        "Fecha de incorporación al DENUE": "Fecha de inscripción",
        "Código Postal": "Código postal",
        "Correo electrónico": "Correo electrónico",
        "Sitio en Internet": "Sitio web",
        "Número de teléfono": "Teléfono",
        "Clee": "CLEE",
        "Score_planta_0_10": "Score planta (0-10)",
        "Riesgo_12m_0_10": "Riesgo estatal (0-10)",
        "Incidentes_12m": "Incidentes estatales 12m",
        "Cálculo_score_planta": "Cálculo score planta",
    }
    out = out.rename(columns=rename_map)

    preferred = [
        "Estado",
        "Municipio",
        "Score planta (0-10)",
        "Cálculo score planta",
        "Riesgo estatal (0-10)",
        "Incidentes estatales 12m",
        "Nombre de la Unidad Económica",
        "Razón social",
        "Estrato personal ocupado",
        "Dirección",
        "Código postal",
        "Teléfono",
        "Correo electrónico",
        "Sitio web",
        "Fecha de inscripción",
        "Latitud",
        "Longitud",
        "CLEE",
        "CVEGEO",
    ]
    preferred = [c for c in preferred if c in out.columns]
    rest = [c for c in out.columns if c not in preferred]
    out = out[preferred + rest].copy()

    return out


def download_buttons(df_export, base_filename):
    csv_bytes = df_export.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Descargar plantas (CSV)",
        data=csv_bytes,
        file_name=f"{base_filename}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_export.to_excel(writer, index=False, sheet_name="Plantas")
    st.download_button(
        "Descargar plantas (Excel)",
        data=buffer.getvalue(),
        file_name=f"{base_filename}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


# =========================================================
# CARGAS BASE DASHBOARD
# =========================================================
@st.cache_data
def load_denue_clean(clean_path):
    last_err = None
    df = None
    for enc in ["utf-8-sig", "cp1252", "latin1", "utf-8"]:
        try:
            df = pd.read_csv(clean_path, encoding=enc, low_memory=False)
            break
        except Exception as e:
            last_err = e
    if df is None:
        raise last_err

    df.columns = [str(c).strip() for c in df.columns]

    keep_cols = [
        "CVEGEO",
        "Entidad federativa",
        "Municipio",
        "Nombre de la Unidad Económica",
        "Razón social",
        "Descripcion estrato personal ocupado",
        "Número de teléfono",
        "Correo electrónico",
        "Sitio en Internet",
        "Nombre de la vialidad",
        "Número exterior o kilómetro",
        "Número interior",
        "Nombre de asentamiento humano",
        "Código Postal",
        "Latitud",
        "Longitud",
        "Fecha de incorporación al DENUE",
        "Clee",
        "DEDUP_KEY",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    if "Entidad federativa" in df.columns:
        df["Estado"] = df["Entidad federativa"].astype(str).map(normalize_estado_name)
    if "Municipio" in df.columns:
        df["Municipio"] = norm_txt(df["Municipio"])

    if "CVEGEO" in df.columns:
        df["CVEGEO"] = df["CVEGEO"].astype(str).str.zfill(5)

    return df


@st.cache_data
def load_mun_geo_and_centroids():
    cent = pd.read_csv(MUN_CENTROIDES, encoding="utf-8-sig")
    cent["CVEGEO"] = cent["CVEGEO"].astype(str).str.zfill(5)

    gdf = gpd.read_file(str(GEOJSON_MUN))
    gdf["CVEGEO"] = gdf["CVEGEO"].astype(str).str.zfill(5)
    gdf = gdf.to_crs(epsg=4326)
    return gdf, cent


@st.cache_data
def load_states_geojson():
    gdf = gpd.read_file(str(STATES_GEOJSON))
    gdf = gdf.to_crs(epsg=4326)

    if "state_name" not in gdf.columns:
        raise ValueError(f"No encontré 'state_name' en states.geojson. Columnas: {gdf.columns.tolist()}")

    gdf["Estado"] = gdf["state_name"].astype(str).str.strip().map(normalize_estado_name)
    gdf = gdf[["Estado", "geometry"]].copy()
    gdf["geometry"] = gdf["geometry"].buffer(0)
    return gdf


@st.cache_data
def load_series():
    mx = pd.read_csv(SERIE_MX, encoding="utf-8-sig")
    stt = pd.read_csv(SERIE_STATE, encoding="utf-8-sig")

    mx["fecha"] = pd.to_datetime(mx["fecha"])
    stt["fecha"] = pd.to_datetime(stt["fecha"])
    stt["Entidad"] = norm_txt(stt["Entidad"]).map(normalize_estado_name)

    return mx, stt


@st.cache_data
def compute_forecast_state_12m(_state_month):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    estados = sorted(_state_month["Entidad"].dropna().unique().tolist())
    rows = []

    for estado in estados:
        ts = (
            _state_month[_state_month["Entidad"] == estado]
            .sort_values("fecha")[["fecha", "Incidentes"]]
            .copy()
        )

        try:
            y = ts.set_index("fecha")["Incidentes"].asfreq("MS").fillna(0)

            model = SARIMAX(
                y,
                order=(1, 1, 1),
                seasonal_order=(0, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)

            fc = res.get_forecast(steps=12)
            mean = fc.predicted_mean.clip(lower=0)
            ci = fc.conf_int(alpha=0.05)
            lower = ci.iloc[:, 0].clip(lower=0)
            upper = ci.iloc[:, 1].clip(lower=0)

            rows.append({
                "Estado": estado,
                "Pron_Optimista_12m": float(lower.sum()),
                "Pron_Base_12m": float(mean.sum()),
                "Pron_Pesimista_12m": float(upper.sum()),
            })
        except Exception:
            last12 = float(ts.tail(12)["Incidentes"].sum()) if len(ts) else 0.0
            rows.append({
                "Estado": estado,
                "Pron_Optimista_12m": last12,
                "Pron_Base_12m": last12,
                "Pron_Pesimista_12m": last12,
            })

    out = pd.DataFrame(rows)
    out["Riesgo_Pron_Optimista_0_10"] = minmax_0_10(out["Pron_Optimista_12m"])
    out["Riesgo_Pron_Base_0_10"] = minmax_0_10(out["Pron_Base_12m"])
    out["Riesgo_Pron_Pesimista_0_10"] = minmax_0_10(out["Pron_Pesimista_12m"])
    return out


# =========================================================
# CARGAS MERCADO / RICARDO
# =========================================================
@st.cache_data
def resolve_market_files():
    market_final = MARKET_FOLDER / "Farma_Imports_BBDD_Final.xlsx"
    market_extended = MARKET_FOLDER / "Farma_Imports_BBDD_Extendida_2026_2027.xlsx"

    if not market_final.exists():
        market_final = None
    if not market_extended.exists():
        market_extended = None

    return market_final, market_extended

@st.cache_data
def load_market_final(final_path_str: str):
    fp = Path(final_path_str)
    df = pd.read_excel(fp, sheet_name=0)
    df = clean_excel_columns(df)

    year_col = pick_col(df, ["Año", "Ano", "Year"])
    month_col = pick_col(df, ["Mes", "Month"])
    total_col = pick_col(df, ["Imports_Total", "Imports Total", "Valor_USD_Total", "Valor_Total", "Total_Imports", "Valor"])

    if total_col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        num_cols = [c for c in num_cols if c not in [year_col, month_col]]
        if not num_cols:
            raise ValueError("No encontré una columna numérica de imports en Farma_Imports_BBDD_Final.xlsx")
        total_col = num_cols[0]

    if year_col is None or month_col is None:
        raise ValueError("No encontré columnas de Año/Mes en Farma_Imports_BBDD_Final.xlsx")

    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    out[month_col] = pd.to_numeric(out[month_col], errors="coerce")
    out[total_col] = pd.to_numeric(out[total_col], errors="coerce")
    out = out.dropna(subset=[year_col, month_col, total_col]).copy()

    out["Fecha"] = pd.to_datetime(
        dict(year=out[year_col].astype(int), month=out[month_col].astype(int), day=1),
        errors="coerce"
    )
    out = out.dropna(subset=["Fecha"]).sort_values("Fecha")
    out = out.rename(columns={total_col: "Imports_Total"})
    return out


@st.cache_data
def load_market_extended(extended_path_str: str | None):
    if extended_path_str is None:
        return None

    fp = Path(extended_path_str)
    xls = pd.ExcelFile(fp)
    sheets = {}
    for sh in xls.sheet_names:
        try:
            tmp = pd.read_excel(fp, sheet_name=sh)
            tmp = clean_excel_columns(tmp)
            sheets[sh] = tmp
        except Exception:
            pass

    if not sheets:
        return None

    best_name = max(sheets, key=lambda k: len(sheets[k]))
    df = sheets[best_name].copy()

    year_col = pick_col(df, ["Año", "Ano", "Year"])
    month_col = pick_col(df, ["Mes", "Month"])
    merc_col = pick_col(df, ["Estado_Mercancia", "Mercancia", "estado_mercancia", "Categoria", "Producto", "HS_Desc"])
    val_col = pick_col(df, ["Volumen_Imports", "Valor_USD", "Imports_Total", "Valor", "Monto_USD"])

    if year_col and month_col and merc_col and val_col:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df[month_col] = pd.to_numeric(df[month_col], errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[year_col, month_col, merc_col, val_col]).copy()
        df["Fecha"] = pd.to_datetime(
            dict(year=df[year_col].astype(int), month=df[month_col].astype(int), day=1),
            errors="coerce"
        )
        df = df.dropna(subset=["Fecha"]).copy()
        df = df.rename(columns={merc_col: "Mercancia", val_col: "Valor"})
        return df[["Fecha", "Mercancia", "Valor"]].copy()

    return None


@st.cache_data
def load_market_destinos(dest_path_str: str | None):
    if dest_path_str is None:
        return None, None, None

    fp = Path(dest_path_str)
    xls = pd.ExcelFile(fp)

    debug = {
        "sheets": xls.sheet_names,
        "columns_by_sheet": {}
    }

    best_df = None
    best_sheet = None

    for sh in xls.sheet_names:
        tmp = pd.read_excel(fp, sheet_name=sh)
        tmp = clean_excel_columns(tmp)
        debug["columns_by_sheet"][sh] = tmp.columns.tolist()

        if best_df is None and not tmp.empty:
            best_df = tmp.copy()
            best_sheet = sh

    if best_df is None:
        return None, None, debug

    df = best_df.copy()

    municipio_col = None
    estado_col = None
    share_col = None
    valor_col = None

    for c in df.columns:
        cl = str(c).strip().lower()
        if municipio_col is None and ("municip" in cl or "destino" in cl):
            municipio_col = c
        if estado_col is None and ("estado" in cl or "state" in cl or "entidad" in cl):
            estado_col = c
        if share_col is None and ("share" in cl or "particip" in cl or "%" in cl):
            share_col = c
        if valor_col is None and ("valor" in cl or "import" in cl or "usd" in cl or "monto" in cl):
            valor_col = c

    if municipio_col is None:
        return None, best_sheet, debug

    out = df.copy()

    if share_col is not None:
        out[share_col] = pd.to_numeric(out[share_col], errors="coerce")
    if valor_col is not None:
        out[valor_col] = pd.to_numeric(out[valor_col], errors="coerce")

    keep_cols = [c for c in [municipio_col, estado_col, share_col, valor_col] if c is not None]
    out = out[keep_cols].copy()

    rename_map = {municipio_col: "Municipio"}
    if estado_col is not None:
        rename_map[estado_col] = "Estado"
    if share_col is not None:
        rename_map[share_col] = "Share_Imports"
    if valor_col is not None:
        rename_map[valor_col] = "Valor_Imports"

    out = out.rename(columns=rename_map)

    return out, best_sheet, debug

@st.cache_data
def build_total_forecast_from_final(final_path_str: str):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import statsmodels.api as sm

    df_imports = load_market_final(final_path_str).copy()
    y = df_imports.set_index("Fecha")["Imports_Total"].sort_index()

    if len(y) < 36:
        hist = y.reset_index().rename(columns={"Imports_Total": "Valor"})
        hist["Tipo"] = "Histórico"
        return hist[["Fecha", "Valor", "Tipo"]], pd.DataFrame(), "N/D"

    val_start = "2023-01-01"
    val_end = "2023-12-01"
    train_end = "2022-12-01"

    y_train = y[:train_end]
    y_val = y[val_start:val_end]

    def eval_model(y_true, y_pred, nombre):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = (np.abs((y_true - y_pred) / y_true).replace([np.inf, -np.inf], np.nan).dropna().mean()) * 100
        return {"modelo": nombre, "MAE": mae, "RMSE": rmse, "MAPE": mape}

    arima_order = (1, 1, 1)
    sarima_order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    arima_mod = sm.tsa.ARIMA(y_train, order=arima_order)
    arima_res = arima_mod.fit()
    arima_fc_2023 = arima_res.forecast(steps=len(y_val))
    met_arima = eval_model(y_val, arima_fc_2023, "ARIMA")

    sarima_mod = sm.tsa.SARIMAX(
        y_train,
        order=sarima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarima_res = sarima_mod.fit(disp=False)
    sarima_fc_2023 = sarima_res.forecast(steps=len(y_val))
    met_sarima = eval_model(y_val, sarima_fc_2023, "SARIMA")

    metrics = pd.DataFrame([met_arima, met_sarima])
    best_name = metrics.sort_values("MAPE").iloc[0]["modelo"] if not metrics.empty else "SARIMA"

    final_end = "2024-12-01"
    y_final = y[:final_end]

    if best_name == "ARIMA":
        best_mod = sm.tsa.ARIMA(y_final, order=arima_order)
        best_res = best_mod.fit()
        fc_24m = best_res.forecast(steps=24)
    else:
        best_mod = sm.tsa.SARIMAX(
            y_final,
            order=sarima_order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        best_res = best_mod.fit(disp=False)
        fc_24m = best_res.forecast(steps=24)

    fechas_fc = pd.date_range("2025-01-01", periods=24, freq="MS")
    df_fc = pd.DataFrame({
        "Fecha": fechas_fc,
        "Valor": fc_24m.values,
        "Tipo": "Forecast",
    })

    hist = y.reset_index().rename(columns={"Imports_Total": "Valor"})
    hist["Tipo"] = "Histórico"

    totalcombined = pd.concat(
        [hist[["Fecha", "Valor", "Tipo"]], df_fc[["Fecha", "Valor", "Tipo"]]],
        ignore_index=True,
    ).sort_values("Fecha")

    return totalcombined, metrics, best_name


# =========================================================
# BASES DASHBOARD
# =========================================================
denue_3254 = load_denue_clean(DENUE_CLEAN)

farma_state = (
    denue_3254.groupby("Estado", as_index=False)["DEDUP_KEY"]
    .nunique()
    .rename(columns={"DEDUP_KEY": "Empresas_Farma"})
)

df_mun = pd.read_csv(MUN_RISK_12M, encoding="utf-8-sig")
df_mun["CVEGEO"] = df_mun["CVEGEO"].astype(str).str.zfill(5)

cat = pd.read_csv(MUN_CATALOGO)
geo_set = set(cat["CVEGEO"].astype(str).str.zfill(5))
df_mun = df_mun[df_mun["CVEGEO"].isin(geo_set)].copy()

df_mun["Entidad"] = norm_txt(df_mun["Entidad"]).map(normalize_estado_name)
df_mun["Municipio"] = norm_txt(df_mun["Municipio"])
df_mun["Estado"] = df_mun["Entidad"]

gdf_mun, centroids = load_mun_geo_and_centroids()
gdf_states = load_states_geojson()
mx_month, state_month = load_series()
forecast_state = compute_forecast_state_12m(state_month)

state_12m = (
    df_mun.groupby("Estado", as_index=False)["Incidentes_12m"]
    .sum()
    .sort_values("Incidentes_12m", ascending=False)
)
state_12m["Riesgo_12m_0_10"] = minmax_0_10(state_12m["Incidentes_12m"])

df_state = state_12m.merge(farma_state, on="Estado", how="left")
df_state["Empresas_Farma"] = pd.to_numeric(df_state["Empresas_Farma"], errors="coerce").fillna(0).astype(int)
df_state["Potencial_Vulnerable"] = df_state["Empresas_Farma"] * df_state["Riesgo_12m_0_10"]
df_state["Potencial_Vulnerable_0_10"] = minmax_0_10(df_state["Potencial_Vulnerable"])

hist_last12 = df_state[["Estado", "Incidentes_12m", "Riesgo_12m_0_10", "Potencial_Vulnerable_0_10"]].copy()
hist_last12 = hist_last12.rename(columns={
    "Incidentes_12m": "Hist_12m",
    "Riesgo_12m_0_10": "Riesgo_Hist_0_10",
    "Potencial_Vulnerable_0_10": "Pot_Hist_0_10"
})

forecast_state = forecast_state.merge(hist_last12, on="Estado", how="left")
forecast_state = forecast_state.merge(df_state[["Estado", "Empresas_Farma"]], on="Estado", how="left")
forecast_state["Empresas_Farma"] = pd.to_numeric(forecast_state["Empresas_Farma"], errors="coerce").fillna(0)

forecast_state["Pot_Pron_Optimista"] = forecast_state["Empresas_Farma"] * forecast_state["Riesgo_Pron_Optimista_0_10"]
forecast_state["Pot_Pron_Base"] = forecast_state["Empresas_Farma"] * forecast_state["Riesgo_Pron_Base_0_10"]
forecast_state["Pot_Pron_Pesimista"] = forecast_state["Empresas_Farma"] * forecast_state["Riesgo_Pron_Pesimista_0_10"]

forecast_state["Pot_Pron_Optimista_0_10"] = minmax_0_10(forecast_state["Pot_Pron_Optimista"])
forecast_state["Pot_Pron_Base_0_10"] = minmax_0_10(forecast_state["Pot_Pron_Base"])
forecast_state["Pot_Pron_Pesimista_0_10"] = minmax_0_10(forecast_state["Pot_Pron_Pesimista"])

forecast_state["Var_%_Inc_Optimista"] = forecast_state.apply(lambda r: pct_change(r["Pron_Optimista_12m"], r["Hist_12m"]), axis=1)
forecast_state["Var_%_Inc_Base"] = forecast_state.apply(lambda r: pct_change(r["Pron_Base_12m"], r["Hist_12m"]), axis=1)
forecast_state["Var_%_Inc_Pesimista"] = forecast_state.apply(lambda r: pct_change(r["Pron_Pesimista_12m"], r["Hist_12m"]), axis=1)

forecast_state["Var_%_Riesgo_Optimista"] = forecast_state.apply(lambda r: pct_change(r["Riesgo_Pron_Optimista_0_10"], r["Riesgo_Hist_0_10"]), axis=1)
forecast_state["Var_%_Riesgo_Base"] = forecast_state.apply(lambda r: pct_change(r["Riesgo_Pron_Base_0_10"], r["Riesgo_Hist_0_10"]), axis=1)
forecast_state["Var_%_Riesgo_Pesimista"] = forecast_state.apply(lambda r: pct_change(r["Riesgo_Pron_Pesimista_0_10"], r["Riesgo_Hist_0_10"]), axis=1)

forecast_state["Var_%_Pot_Optimista"] = forecast_state.apply(lambda r: pct_change(r["Pot_Pron_Optimista_0_10"], r["Pot_Hist_0_10"]), axis=1)
forecast_state["Var_%_Pot_Base"] = forecast_state.apply(lambda r: pct_change(r["Pot_Pron_Base_0_10"], r["Pot_Hist_0_10"]), axis=1)
forecast_state["Var_%_Pot_Pesimista"] = forecast_state.apply(lambda r: pct_change(r["Pot_Pron_Pesimista_0_10"], r["Pot_Hist_0_10"]), axis=1)


# =========================================================
# UI PRINCIPAL
# =========================================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Filtros")

    estados = sorted(df_state["Estado"].dropna().unique().tolist())
    estado_sel = st.selectbox("Estado:", estados, index=0, key="estado_sel")

    st.divider()

    map_view = st.radio(
        "Vista del mapa:",
        ["Histórico 12m", "Pronóstico 12m (SARIMA)"],
        index=0,
        key="map_view"
    )

    metric_mode = st.radio(
        "Métrica:",
        ["Riesgo", "Potencial"],
        index=0,
        key="metric_mode"
    )

    if map_view == "Pronóstico 12m (SARIMA)":
        forecast_scenario = st.radio(
            "Escenario pronóstico:",
            ["Pesimista", "Base", "Optimista"],
            horizontal=True,
            index=1,
            key="forecast_scenario"
        )
    else:
        forecast_scenario = "Base"

    st.divider()

    row_state = df_state[df_state["Estado"] == estado_sel]
    if row_state.empty:
        emp_farma_state = 0
        inc_state = 0
        risk_state = 0.0
        pot_state = 0.0
    else:
        row_state = row_state.iloc[0]
        emp_farma_state = int(row_state["Empresas_Farma"])
        inc_state = int(row_state["Incidentes_12m"])
        risk_state = float(row_state["Riesgo_12m_0_10"])
        pot_state = float(row_state["Potencial_Vulnerable_0_10"])

    k1, k2 = st.columns(2)
    with k1:
        st.metric("Capacidad farma (estado)", f"{emp_farma_state:,}")
        st.metric("Riesgo actual (0–10)", f"{fmt_riesgo(risk_state)}/10")
    with k2:
        st.metric("Incidentes actuales 12m", f"{inc_state:,}")
        st.metric("Potencial actual (0–10)", f"{fmt_riesgo(pot_state)}/10")

    if map_view == "Pronóstico 12m (SARIMA)":
        row_fc = forecast_state[forecast_state["Estado"] == estado_sel]
        if not row_fc.empty:
            row_fc = row_fc.iloc[0]

            if forecast_scenario == "Optimista":
                inc_fc = float(row_fc["Pron_Optimista_12m"])
                risk_fc = float(row_fc["Riesgo_Pron_Optimista_0_10"])
                pot_fc = float(row_fc["Pot_Pron_Optimista_0_10"])
                inc_var = row_fc["Var_%_Inc_Optimista"]
                risk_var = row_fc["Var_%_Riesgo_Optimista"]
                pot_var = row_fc["Var_%_Pot_Optimista"]
            elif forecast_scenario == "Pesimista":
                inc_fc = float(row_fc["Pron_Pesimista_12m"])
                risk_fc = float(row_fc["Riesgo_Pron_Pesimista_0_10"])
                pot_fc = float(row_fc["Pot_Pron_Pesimista_0_10"])
                inc_var = row_fc["Var_%_Inc_Pesimista"]
                risk_var = row_fc["Var_%_Riesgo_Pesimista"]
                pot_var = row_fc["Var_%_Pot_Pesimista"]
            else:
                inc_fc = float(row_fc["Pron_Base_12m"])
                risk_fc = float(row_fc["Riesgo_Pron_Base_0_10"])
                pot_fc = float(row_fc["Pot_Pron_Base_0_10"])
                inc_var = row_fc["Var_%_Inc_Base"]
                risk_var = row_fc["Var_%_Riesgo_Base"]
                pot_var = row_fc["Var_%_Pot_Base"]

            st.divider()

            f1, f2, f3 = st.columns(3)
            with f1:
                st.metric(
                    f"Pronóstico {forecast_scenario} (incidentes 12m)",
                    f"{inc_fc:,.1f}",
                    delta=f"{fmt_pp(inc_var)} vs AA"
                )
            with f2:
                st.metric(
                    "Riesgo pronosticado (0–10)",
                    f"{fmt_riesgo(risk_fc)}/10",
                    delta=f"{fmt_pp(risk_var)} vs AA"
                )
            with f3:
                st.metric(
                    "Potencial pronosticado (0–10)",
                    f"{fmt_riesgo(pot_fc)}/10",
                    delta=f"{fmt_pp(pot_var)} vs AA"
                )

            st.markdown("### Insight del escenario")

            if metric_mode == "Riesgo":
                if forecast_scenario == "Optimista":
                    sort_col = "Riesgo_Pron_Optimista_0_10"
                    delta_col = "Var_%_Riesgo_Optimista"
                elif forecast_scenario == "Pesimista":
                    sort_col = "Riesgo_Pron_Pesimista_0_10"
                    delta_col = "Var_%_Riesgo_Pesimista"
                else:
                    sort_col = "Riesgo_Pron_Base_0_10"
                    delta_col = "Var_%_Riesgo_Base"
                label_main = "riesgo"
            else:
                if forecast_scenario == "Optimista":
                    sort_col = "Pot_Pron_Optimista_0_10"
                    delta_col = "Var_%_Pot_Optimista"
                elif forecast_scenario == "Pesimista":
                    sort_col = "Pot_Pron_Pesimista_0_10"
                    delta_col = "Var_%_Pot_Pesimista"
                else:
                    sort_col = "Pot_Pron_Base_0_10"
                    delta_col = "Var_%_Pot_Base"
                label_main = "potencial"

            top3 = (
                forecast_state[["Estado", sort_col, delta_col]]
                .copy()
                .sort_values(sort_col, ascending=False)
                .head(3)
                .reset_index(drop=True)
            )

            if len(top3) == 3:
                t1 = f"{top3.loc[0, 'Estado']} ({fmt_riesgo(top3.loc[0, sort_col])}/10; {fmt_pp(top3.loc[0, delta_col])} vs AA)"
                t2 = f"{top3.loc[1, 'Estado']} ({fmt_riesgo(top3.loc[1, sort_col])}/10; {fmt_pp(top3.loc[1, delta_col])} vs AA)"
                t3 = f"{top3.loc[2, 'Estado']} ({fmt_riesgo(top3.loc[2, sort_col])}/10; {fmt_pp(top3.loc[2, delta_col])} vs AA)"

                st.caption(
                    f"En el escenario **{forecast_scenario.lower()}**, los 3 estados con mayor **{label_main}** proyectado son: "
                    f"**{t1}**, **{t2}** y **{t3}**."
                )

with col2:
    st.subheader("Mapa de calor — estatal")

    view_all = st.checkbox("Ver toda la República", value=True, key="view_all")

    if map_view == "Histórico 12m":
        map_geo = gdf_states.merge(df_state, on="Estado", how="left").copy()

        if metric_mode == "Riesgo":
            value_col = "Riesgo_12m_0_10"
            legend = "Riesgo estatal histórico 12m (0–10)"
            color_scale = "Reds"
        else:
            value_col = "Potencial_Vulnerable_0_10"
            legend = "Potencial estatal histórico 12m (0–10)"
            color_scale = "Greens"
    else:
        map_geo = gdf_states.merge(forecast_state, on="Estado", how="left").copy()

        if metric_mode == "Riesgo":
            if forecast_scenario == "Pesimista":
                value_col = "Riesgo_Pron_Pesimista_0_10"
            elif forecast_scenario == "Optimista":
                value_col = "Riesgo_Pron_Optimista_0_10"
            else:
                value_col = "Riesgo_Pron_Base_0_10"
            legend = f"Riesgo estatal pronosticado (0–10) — escenario {forecast_scenario}"
            color_scale = "Reds"
        else:
            if forecast_scenario == "Pesimista":
                value_col = "Pot_Pron_Pesimista_0_10"
            elif forecast_scenario == "Optimista":
                value_col = "Pot_Pron_Optimista_0_10"
            else:
                value_col = "Pot_Pron_Base_0_10"
            legend = f"Potencial estatal pronosticado (0–10) — escenario {forecast_scenario}"
            color_scale = "Greens"

    if value_col not in map_geo.columns:
        st.error(f"No encontré la columna del mapa: {value_col}")
    else:
        map_geo[value_col] = pd.to_numeric(map_geo[value_col], errors="coerce").fillna(0)
        map_geo = map_geo[map_geo.geometry.notna()].copy()

        m = folium.Map(location=[23.6, -102.5], zoom_start=5, tiles="cartodbpositron")

        if view_all:
            b_all = map_geo.to_crs(epsg=4326).total_bounds
            m.fit_bounds([[b_all[1], b_all[0]], [b_all[3], b_all[2]]])
        else:
            b = bounds_for_estado_states(map_geo, estado_sel)
            if b:
                m.fit_bounds(b)

        folium.Choropleth(
            geo_data=map_geo.__geo_interface__,
            data=map_geo,
            columns=["Estado", value_col],
            key_on="feature.properties.Estado",
            fill_color=color_scale,
            fill_opacity=0.82,
            line_opacity=0.9,
            line_color="#4B5563",
            legend_name=legend,
            nan_fill_color="#E5E7EB",
            nan_fill_opacity=0.65,
        ).add_to(m)

        folium.GeoJson(
            map_geo.__geo_interface__,
            tooltip=folium.GeoJsonTooltip(
                fields=["Estado", value_col],
                aliases=["Estado:", "Valor:"],
                localize=True,
                sticky=True,
                labels=True,
            ),
            style_function=lambda x: {
                "fillColor": "transparent",
                "color": "#374151",
                "weight": 1.2,
                "fillOpacity": 0,
            },
            highlight_function=lambda x: {
                "fillColor": "#000000",
                "color": "#111827",
                "weight": 2.2,
                "fillOpacity": 0.08,
            },
        ).add_to(m)

        if not view_all:
            state_sel_geo = map_geo[map_geo["Estado"] == estado_sel].copy()
            if len(state_sel_geo):
                folium.GeoJson(
                    state_sel_geo.__geo_interface__,
                    style_function=lambda x: {
                        "fillColor": "transparent",
                        "color": "#111827",
                        "weight": 3,
                        "fillOpacity": 0,
                        "dashArray": "6 6",
                    },
                ).add_to(m)

        st_folium(m, width="100%", height=520, key="map_main")


# =========================================================
# PLANTAS
# =========================================================
st.divider()
st.subheader("Plantas identificadas (DENUE 3254)")

plants_state = denue_3254[denue_3254["Estado"] == estado_sel].copy()

estratos_objetivo = [
    "51 a 100 personas",
    "101 a 250 personas",
    "251 y más personas"
]

plants_state = plants_state[plants_state["Descripcion estrato personal ocupado"].isin(estratos_objetivo)].copy()
plants_state = add_plant_score(plants_state, df_state)

if plants_state.empty:
    st.info("No se encontraron plantas DENUE 3254 para este estado.")
else:
    st.caption("Top 5 de plantas medianas y grandes. Se excluyen establecimientos menores a 51 empleados para enfocar el análisis en operaciones con mayor escala comercial.")
    cols_view = [c for c in [
        "Nombre de la Unidad Económica",
        "Razón social",
        "Descripcion estrato personal ocupado"
    ] if c in plants_state.columns]

    st.dataframe(
        plants_state[cols_view].head(5),
        use_container_width=True,
        hide_index=True
    )

    export_state = build_plants_export(plants_state)
    download_buttons(export_state, base_filename=f"denue_3254_{estado_sel}")


# =========================================================
# RUTA + SCORE
# =========================================================
st.divider()
st.subheader("Ruta + score de riesgo")

risk_mun = df_mun[["CVEGEO", "Riesgo_0_10", "Incidentes_12m", "Entidad", "Municipio"]].copy()
risk_mun["Entidad"] = fix_cdmx(risk_mun["Entidad"])
risk_mun["CVEGEO"] = risk_mun["CVEGEO"].astype(str).str.zfill(5)
risk_mun["CVE_ENT"] = risk_mun["CVEGEO"].str[:2]

tmp_state = risk_mun.groupby("CVE_ENT", as_index=False)["Entidad"].agg(lambda x: x.iloc[0])
state_map = dict(zip(tmp_state["CVE_ENT"], tmp_state["Entidad"]))

cent2 = centroids.copy()
cent2["CVEGEO"] = cent2["CVEGEO"].astype(str).str.zfill(5)
cent2["CVE_ENT"] = cent2["CVEGEO"].str[:2]

cent2 = cent2.merge(risk_mun[["CVEGEO", "Riesgo_0_10", "Incidentes_12m"]], on="CVEGEO", how="left")
cent2["Riesgo_0_10"] = cent2["Riesgo_0_10"].fillna(0.0)
cent2["Incidentes_12m"] = cent2["Incidentes_12m"].fillna(0).astype(int)
cent2["Estado"] = cent2["CVE_ENT"].map(state_map).fillna("Estado " + cent2["CVE_ENT"])

cA, cB = st.columns(2)
estados_list = sorted(cent2["Estado"].dropna().unique().tolist())

with cA:
    st.markdown("**Origen**")
    estado_o = st.selectbox("Estado (origen):", estados_list, index=0, key="estado_origen")
    mun_o_list = cent2[cent2["Estado"] == estado_o].sort_values("NOMGEO")["NOMGEO"].unique().tolist()
    mun_o = st.selectbox("Municipio (origen):", mun_o_list, index=0, key="mun_origen")

with cB:
    st.markdown("**Destino**")
    estado_d = st.selectbox("Estado (destino):", estados_list, index=min(1, len(estados_list) - 1), key="estado_destino")
    mun_d_list = cent2[cent2["Estado"] == estado_d].sort_values("NOMGEO")["NOMGEO"].unique().tolist()
    mun_d = st.selectbox("Municipio (destino):", mun_d_list, index=min(1, len(mun_d_list) - 1), key="mun_destino")

run_route = st.button("Calcular ruta y riesgo", type="primary")


def osrm_route(lat1, lon1, lat2, lon2, retries=3, timeout=120):
    url = f"https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {"overview": "full", "geometries": "geojson"}

    last_error = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()

            if not data.get("routes"):
                raise ValueError("OSRM no devolvió rutas.")

            route = data["routes"][0]
            coords = route["geometry"]["coordinates"]
            dist_km = route["distance"] / 1000
            return coords, dist_km

        except requests.exceptions.ReadTimeout as e:
            last_error = e
            if i < retries - 1:
                time.sleep(2)
            else:
                raise ValueError(
                    "El servidor público de OSRM tardó demasiado en responder. "
                    "Vuelve a intentar en unos segundos o prueba una ruta más corta."
                ) from e

        except requests.exceptions.RequestException as e:
            last_error = e
            if i < retries - 1:
                time.sleep(2)
            else:
                raise ValueError(f"Error de conexión con OSRM: {e}") from e

    raise last_error


if run_route:
    try:
        o = cent2[(cent2["Estado"] == estado_o) & (cent2["NOMGEO"] == mun_o)].iloc[0]
        d = cent2[(cent2["Estado"] == estado_d) & (cent2["NOMGEO"] == mun_d)].iloc[0]

        with st.spinner("Consultando ruta real (OSRM) y calculando riesgo municipal..."):
            coords, dist_km = osrm_route(o["lat"], o["lon"], d["lat"], d["lon"])
            line = LineString(coords)

            route_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[line], crs="EPSG:4326")
            gdf_m = gdf_mun[["CVEGEO", "NOMGEO", "geometry"]].copy().to_crs(epsg=3857)
            route_m = route_gdf.to_crs(epsg=3857)

            route_geom = route_m.geometry.iloc[0]
            cand = gdf_m[gdf_m.intersects(route_geom)].copy()

            cand["seg"] = cand.geometry.intersection(route_geom)
            cand["km"] = cand["seg"].length / 1000.0
            cand = cand[cand["km"] > 1e-6].copy()

            cand = cand.merge(risk_mun[["CVEGEO", "Riesgo_0_10"]], on="CVEGEO", how="left")
            cand["Riesgo_0_10"] = cand["Riesgo_0_10"].fillna(0.0)

            total_km = float(cand["km"].sum()) if len(cand) else 0.0
            score = float((cand["km"] * cand["Riesgo_0_10"]).sum() / total_km) if total_km > 0 else 0.0

            out = cand[["CVEGEO", "NOMGEO", "km", "Riesgo_0_10"]].sort_values("km", ascending=False).reset_index(drop=True)
            out["risk_km"] = out["km"] * out["Riesgo_0_10"]
            total_risk_km = float(out["risk_km"].sum())
            out["Contrib_%"] = (100 * out["risk_km"] / max(total_risk_km, 1e-9)).round(2)

            st.session_state["route_result"] = {
                "dist_km": dist_km,
                "score": score,
                "coords": coords,
                "origin": {"estado": estado_o, "mun": mun_o, "lat": float(o["lat"]), "lon": float(o["lon"])},
                "dest": {"estado": estado_d, "mun": mun_d, "lat": float(d["lat"]), "lon": float(d["lon"])},
                "table": out
            }

    except Exception as e:
        st.error(f"Error calculando ruta/riesgo: {e}")

resu = st.session_state.get("route_result", None)

if resu:
    k1, k2 = st.columns(2)
    k1.metric("Distancia estimada de ruta (km)", f"{resu['dist_km']:,.1f}")
    k2.metric("Riesgo ponderado de la ruta (0–10)", f"{resu['score']:,.2f}/10")
    st.caption("La distancia es estimada. El riesgo pondera el tramo recorrido dentro de cada municipio.")

    out = resu["table"].copy().sort_values("Contrib_%", ascending=False).reset_index(drop=True)
    out_show = out.head(20).copy()
    out_show["km"] = out_show["km"].round(1)
    out_show["Riesgo_0_10"] = out_show["Riesgo_0_10"].round(2)

    st.markdown("### Municipios que explican el riesgo de la ruta (Top 20)")
    st.dataframe(
        out_show.rename(columns={
            "NOMGEO": "Municipio",
            "km": "KM en ruta",
            "Riesgo_0_10": "Riesgo (0–10)",
        })[["CVEGEO", "Municipio", "KM en ruta", "Riesgo (0–10)", "Contrib_%"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "KM en ruta": st.column_config.NumberColumn(format="%.1f"),
            "Riesgo (0–10)": st.column_config.NumberColumn(format="%.2f"),
            "Contrib_%": st.column_config.ProgressColumn(
                "Contribución al riesgo (%)",
                min_value=0.0,
                max_value=100.0,
                format="%.2f",
            ),
        },
    )

    fig_contrib = px.bar(
        out_show.sort_values("Contrib_%", ascending=True),
        x="Contrib_%",
        y="NOMGEO",
        orientation="h",
        title="Contribución al riesgo — Top 20 (%)",
        labels={"Contrib_%": "Contribución (%)", "NOMGEO": "Municipio"},
    )
    fig_contrib.update_layout(height=480, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_contrib, use_container_width=True)

    st.markdown("### Mapa de la ruta estimada")

    o = resu["origin"]
    d = resu["dest"]
    coords = resu["coords"]

    route_map = folium.Map(
        location=[(o["lat"] + d["lat"]) / 2, (o["lon"] + d["lon"]) / 2],
        zoom_start=6,
        tiles="cartodbpositron"
    )

    folium.GeoJson(
        str(GEOJSON_MUN),
        style_function=lambda x: {
            "fillColor": "transparent",
            "color": "#666666",
            "weight": 0.3,
            "fillOpacity": 0
        },
        name="Municipios"
    ).add_to(route_map)

    folium.PolyLine(
        locations=[(lat, lon) for lon, lat in coords],
        weight=4,
        opacity=0.9,
        tooltip="Ruta estimada"
    ).add_to(route_map)

    folium.Marker([o["lat"], o["lon"]], tooltip=f"Origen: {o['mun']} ({o['estado']})").add_to(route_map)
    folium.Marker([d["lat"], d["lon"]], tooltip=f"Destino: {d['mun']} ({d['estado']})").add_to(route_map)

    lats = [lat for lon, lat in coords]
    lons = [lon for lon, lat in coords]
    route_map.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    st_folium(route_map, width="100%", height=520, key="route_map")


# =========================================================
# EVOLUCIÓN MENSUAL + SARIMA
# =========================================================
st.divider()
st.subheader("Evolución mensual — robo a transportista (SNSP)")

scope = st.radio(
    "Ver serie por:",
    ["Nacional (MX)", "Por estado"],
    horizontal=True,
    key="scope_ts"
)

if scope == "Por estado":
    estados_ts = sorted(state_month["Entidad"].dropna().unique().tolist())
    default_idx = estados_ts.index(estado_sel) if estado_sel in estados_ts else 0
    estado_ts = st.selectbox("Estado:", estados_ts, index=default_idx, key="estado_ts_sel")
    ts = state_month[state_month["Entidad"] == estado_ts].sort_values("fecha")[["fecha", "Incidentes"]].reset_index(drop=True)
    titulo = f"Incidentes mensuales — {estado_ts}"
else:
    ts = mx_month.sort_values("fecha")[["fecha", "Incidentes"]].reset_index(drop=True)
    titulo = "Incidentes mensuales — Nacional (MX)"

run_fc = st.checkbox(
    "Calcular pronóstico (SARIMA) y mostrarlo en la misma gráfica",
    value=False,
    key="run_fc_inline"
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ts["fecha"],
    y=ts["Incidentes"],
    mode="lines+markers",
    name="Histórico",
    line=dict(width=2, color="#1f77b4")
))

if run_fc:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = ts.set_index("fecha")["Incidentes"].asfreq("MS").fillna(0)

        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)

        fc = res.get_forecast(steps=12)
        mean = fc.predicted_mean.clip(lower=0)
        ci = fc.conf_int(alpha=0.05)
        lower = ci.iloc[:, 0].clip(lower=0)
        upper = ci.iloc[:, 1].clip(lower=0)

        cut_date = y.index.max()
        fig.add_vline(
            x=cut_date,
            line_width=1,
            line_dash="dot",
            line_color="rgba(80,80,80,0.6)"
        )

        fig.add_trace(go.Scatter(
            x=lower.index, y=lower.values,
            mode="lines",
            name="Pesimista",
            line=dict(color="#e74c3c", width=2, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=mean.index, y=mean.values,
            mode="lines",
            name="Base",
            line=dict(color="#8aa0b6", width=2, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=upper.index, y=upper.values,
            mode="lines",
            name="Optimista",
            line=dict(color="#2ecc71", width=2, dash="dash")
        ))

        fig.update_xaxes(range=[ts["fecha"].min(), upper.index.max()])

    except Exception as e:
        st.error(f"No pude correr SARIMA. Error: {e}")

fig.update_layout(
    title=titulo,
    height=420,
    margin=dict(l=10, r=10, t=60, b=10),
    xaxis_title="Fecha",
    yaxis_title="Incidentes (mensual)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
)
st.plotly_chart(fig, use_container_width=True)


# =========================================================
# RANKING
# =========================================================
st.divider()
st.subheader("Ranking de estados")

metric = st.radio(
    "Ordenar por:",
    ["Riesgo 12m (0–10)", "Incidentes 12m (conteo)", "Empresas farma (DENUE 3254)", "Potencial vulnerable (0–10)"],
    horizontal=True
)

rank_n = st.selectbox("Mostrar:", ["Top 5", "Top 10", "Top 15", "Todos"], index=1)
order = st.radio("Orden:", ["Mayor → menor", "Menor → mayor"], horizontal=True)

col_map = {
    "Riesgo 12m (0–10)": "Riesgo_12m_0_10",
    "Incidentes 12m (conteo)": "Incidentes_12m",
    "Empresas farma (DENUE 3254)": "Empresas_Farma",
    "Potencial vulnerable (0–10)": "Potencial_Vulnerable_0_10",
}

mcol = col_map[metric]
n = None if rank_n == "Todos" else int(rank_n.split()[-1])

rank = df_state.copy()
rank[mcol] = pd.to_numeric(rank[mcol], errors="coerce").fillna(0)
rank = rank.sort_values(mcol, ascending=(order == "Menor → mayor"))
if n is not None:
    rank = rank.head(n)

def fmt_val(x):
    if mcol in ["Empresas_Farma", "Incidentes_12m"]:
        return f"{int(round(x)):,}"
    return f"{x:,.2f}"

rank["label"] = rank[mcol].apply(fmt_val)

if mcol == "Riesgo_12m_0_10":
    scale = "RdYlGn_r"
elif mcol == "Incidentes_12m":
    scale = "Reds"
elif mcol == "Empresas_Farma":
    scale = "Greens"
elif mcol == "Potencial_Vulnerable_0_10":
    scale = "Greens"
else:
    scale = "Blues"

fig_rank = px.bar(
    rank,
    x=mcol,
    y="Estado",
    orientation="h",
    text="label",
    title=f"{rank_n} — {metric} ({order})",
    labels={mcol: metric, "Estado": "Estado"},
    color=mcol,
    color_continuous_scale=scale
)

if mcol in ["Riesgo_12m_0_10", "Potencial_Vulnerable_0_10"]:
    fig_rank.update_coloraxes(cmin=0, cmax=10)
else:
    fig_rank.update_coloraxes(cmin=float(rank[mcol].min()), cmax=float(rank[mcol].max()))

fig_rank.update_layout(coloraxis_colorbar=dict(title=""))
fig_rank.update_yaxes(categoryorder="array", categoryarray=rank["Estado"].tolist(), autorange="reversed")
fig_rank.update_traces(textposition="outside", cliponaxis=False)
fig_rank.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10), transition_duration=0)

max_x = rank[mcol].max()
if pd.notna(max_x) and max_x > 0:
    fig_rank.update_xaxes(range=[0, max_x * 1.15])

st.plotly_chart(fig_rank, use_container_width=True)


# =========================================================
# MÓDULO MERCADO / IMPORTS MX-USA
# =========================================================
st.divider()
st.subheader("Mercado farma — imports y outlook comercial")

market_final, market_extended = resolve_market_files()

if market_final is None:
    st.warning("No encontré Farma_Imports_BBDD_Final.xlsx. Este módulo no se puede mostrar.")
else:
    try:
        totalcombined, market_metrics, best_market_model = build_total_forecast_from_final(str(market_final))
        market_hist = load_market_final(str(market_final))
        market_ext = load_market_extended(str(market_extended)) if market_extended is not None else None


        totalcombined = totalcombined.sort_values("Fecha").copy()
        totalcombined["Año"] = totalcombined["Fecha"].dt.year

        histyearly = (
            totalcombined[totalcombined["Tipo"] == "Histórico"]
            .groupby("Año")["Valor"]
            .sum()
            .reset_index()
        )
        forecastyearly = (
            totalcombined[totalcombined["Tipo"] == "Forecast"]
            .groupby("Año")["Valor"]
            .sum()
            .reset_index()
        )

        last_month = market_hist["Fecha"].max()
        last_month_total = market_hist.loc[market_hist["Fecha"] == last_month, "Imports_Total"].sum()

        if not histyearly.empty and (histyearly["Año"] == 2024).any():
            total_hist_2024 = histyearly.loc[histyearly["Año"] == 2024, "Valor"].iloc[0]
        elif not histyearly.empty:
            total_hist_2024 = histyearly["Valor"].iloc[-1]
        else:
            total_hist_2024 = 0.0

        if len(histyearly) >= 2:
            first_hist_year = int(histyearly["Año"].min())
            last_hist_year = int(histyearly["Año"].max())
            start_val = histyearly.loc[histyearly["Año"] == first_hist_year, "Valor"].iloc[0]
            end_val = histyearly.loc[histyearly["Año"] == last_hist_year, "Valor"].iloc[0]
            n_years = max(last_hist_year - first_hist_year, 1)
            cagr_hist = (end_val / start_val) ** (1 / n_years) - 1 if start_val > 0 else 0.0
        else:
            cagr_hist = 0.0

        growth_26_vs_25 = 0.0
        val_25 = forecastyearly.loc[forecastyearly["Año"] == 2025, "Valor"]
        val_26 = forecastyearly.loc[forecastyearly["Año"] == 2026, "Valor"]
        if not val_25.empty and not val_26.empty and val_25.iloc[0] > 0:
            growth_26_vs_25 = val_26.iloc[0] / val_25.iloc[0] - 1

        market_tabs = st.tabs(["Visión total", "Categorías"])

        with market_tabs[0]:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Tamaño mercado 2024", f"{format_compact_number(total_hist_2024)} USD")
            k2.metric("CAGR histórico", f"{cagr_hist * 100:.1f}%")
            k3.metric("Impulso 2026 vs 2025", f"{growth_26_vs_25 * 100:.1f}%")
            k4.metric("Modelo forecast", best_market_model)

            st.caption(
                f"El mercado de imports farma cerró {format_month_label(last_month)} con "
                f"{format_compact_number(last_month_total)} USD. "
                f"El modelo seleccionado para forecast fue {best_market_model}."
            )

            c1, c2 = st.columns([1.45, 1.0])

            with c1:
                fig_total = px.line(
                    totalcombined,
                    x="Fecha",
                    y="Valor",
                    color="Tipo",
                    markers=True,
                    color_discrete_map={"Histórico": "#0ea5e9", "Forecast": "#22c55e"},
                    title="Serie histórica + forecast del mercado total"
                )
                fig_total = apply_boardroom_theme(fig_total, yaxis_money=True)
                fig_total.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=95, b=10),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=1.01,
                        xanchor="left",
                        x=0,
                        title=None
                    )
                )
                st.plotly_chart(fig_total, use_container_width=True, config={"displayModeBar": False})

            with c2:
                yearly_all = pd.concat(
                    [histyearly.assign(Tipo="Histórico"), forecastyearly.assign(Tipo="Forecast")],
                    ignore_index=True,
                )
                fig_year = px.bar(
                    yearly_all,
                    x="Año",
                    y="Valor",
                    color="Tipo",
                    barmode="group",
                    color_discrete_map={"Histórico": "#0ea5e9", "Forecast": "#22c55e"},
                    title="Trazo anual histórico vs forecast"
                )
                fig_year = apply_boardroom_theme(fig_year, yaxis_money=True)
                fig_year.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=95, b=10),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=1.01,
                        xanchor="left",
                        x=0,
                        title=None
                    )
                )
                st.plotly_chart(fig_year, use_container_width=True, config={"displayModeBar": False})


        with market_tabs[1]:
            if market_ext is None or market_ext.empty:
                st.info("No pude construir la vista de categorías desde el archivo extendido.")
            else:
                merc_last = (
                    market_ext.groupby(["Fecha", "Mercancia"], as_index=False)["Valor"]
                    .sum()
                    .sort_values("Fecha")
                )

                latest_m = merc_last["Fecha"].max()
                mix_last = (
                    merc_last[merc_last["Fecha"] == latest_m]
                    .groupby("Mercancia", as_index=False)["Valor"]
                    .sum()
                )
                mix_last["Share"] = mix_last["Valor"] / mix_last["Valor"].sum()

                c1, c2 = st.columns([1.35, 1.0])

                with c1:
                    fig_merc = px.line(
                        merc_last,
                        x="Fecha",
                        y="Valor",
                        color="Mercancia",
                        title="Evolución por categoría"
                    )
                    fig_merc = apply_boardroom_theme(fig_merc, yaxis_money=True)
                    fig_merc.update_layout(
                        height=470,
                        margin=dict(l=10, r=10, t=60, b=80),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.18,
                            xanchor="center",
                            x=0.5,
                            title=None,
                            font=dict(size=10),
                            bgcolor="rgba(255,255,255,0.92)"
                        )
                    )
                    st.plotly_chart(fig_merc, use_container_width=True, config={"displayModeBar": False})

                with c2:
                    fig_mix = px.bar(
                        mix_last.sort_values("Share", ascending=True),
                        x="Share",
                        y="Mercancia",
                        orientation="h",
                        title=f"Mix de categorías en {format_month_label(latest_m)}",
                        text="Share"
                    )
                    fig_mix = apply_boardroom_theme(fig_mix, yaxis_money=False)
                    fig_mix.update_xaxes(tickformat=".0%")
                    fig_mix.update_traces(texttemplate="%{x:.1%}", textposition="outside")
                    fig_mix.update_layout(height=430, margin=dict(l=10, r=30, t=55, b=10))
                    st.plotly_chart(fig_mix, use_container_width=True, config={"displayModeBar": False})

    except Exception as e:
        st.error(f"No pude integrar el módulo de mercado. Error: {e}")
