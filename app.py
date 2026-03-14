import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    APPTITLE,
    APPICON,
    LAYOUT,
    PAGES,
    COLORMAP,
    MERCANCIAORDER,
    TOPNDEFAULT,
)
from data_loader import (
    loadriskdata,
    getimportstables,
    getcompanytables,
)

# --------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILO GLOBAL
# --------------------------------------------------------------------
st.set_page_config(
    page_title=APPTITLE,
    page_icon=APPICON,
    layout=LAYOUT,
)

st.markdown(
    """
    <style>
    .main {
        background: #f5f7fb;
        padding: 0rem 2.5rem 2.5rem 2.5rem;
    }
    section[data-testid="stSidebar"] {
        background: #ffffff;
        color: #0f172a;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebarNav"]::before {
        content: "CL Circular | Farma México";
        margin-left: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #9ca3af;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    html, body, [class*="css"] {
        font-family: -apple-system, system-ui, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        color: #0f172a;
    }
    .cl-header {
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        padding: 1.3rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #0b1120 45%, #0369a1 100%);
        border: 1px solid rgba(15, 23, 42, 0.5);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        color: #e5e7eb;
    }
    .cl-header-left {
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
    }
    .cl-header-title {
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: #f9fafb;
    }
    .cl-header-subtitle {
        font-size: 0.86rem;
        color: #cbd5f5;
    }
    .cl-header-pill {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #e5e7eb;
        background: rgba(15, 118, 110, 0.12);
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
        border: 1px solid rgba(45, 212, 191, 0.7);
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .cl-header-pill-dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
        background: #22c55e;
        box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.25);
    }
    .cl-header-right {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.35rem;
    }
    .cl-header-metric-label {
        font-size: 0.75rem;
        color: #cbd5f5;
    }
    .cl-header-metric-value {
        font-size: 0.95rem;
        font-weight: 500;
        color: #e5e7eb;
    }
    .element-container:has(div[data-testid="stPlotlyChart"]) {
        padding: 0.3rem 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
def normalize_mercancia_value(value):
    if pd.isna(value):
        return value
    text = str(value).strip()
    mapping = {
        "VitaminsHerbalSupplements": "VitaminsHerbalSupplements",
        "OtrosFarmaUSA": "OtrosFarmaUSA",
        "Hormones": "Hormones",
        "VaccinesImmunological": "VaccinesImmunological",
        "NeedlesInjectionInstruments": "NeedlesInjectionInstruments",
        "AntiInfectives": "AntiInfectives",
    }
    return mapping.get(text, text)


def business_label(value):
    if pd.isna(value):
        return "N/A"
    text = str(value).strip()
    mapping = {
        "VitaminsHerbalSupplements": "Vitamins & Herbal",
        "OtrosFarmaUSA": "Otros Farma USA",
        "Hormones": "Hormones",
        "VaccinesImmunological": "Vaccines & Immunological",
        "NeedlesInjectionInstruments": "Needles & Injection",
        "AntiInfectives": "Anti-infectives",
    }
    return mapping.get(text, text)


def pick_col(df, options):
    for col in options:
        if col in df.columns:
            return col
    return None


def format_money_axis(fig, axis="y"):
    if axis == "y":
        fig.update_yaxes(tickformat=",.0f")
    elif axis == "x":
        fig.update_xaxes(tickformat=",.0f")


def build_fecha_from_year_month(df, year_col, month_col):
    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    out[month_col] = pd.to_numeric(out[month_col], errors="coerce")
    out = out.dropna(subset=[year_col, month_col]).copy()
    out["Fecha"] = pd.to_datetime(
        dict(
            year=out[year_col].astype(int),
            month=out[month_col].astype(int),
            day=1,
        ),
        errors="coerce",
    )
    return out


def filter_series_view(df, tipo_col, selected_view):
    if selected_view == "Ambos":
        return df.copy()
    return df[df[tipo_col] == selected_view].copy()


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


def format_period_label(start_year, end_year):
    return f"{int(start_year)}-{int(end_year)}"


def render_kpi_card(title, value, subtitle="", tone="neutral"):
    tones = {
        "neutral": {
            "border": "rgba(148, 163, 184, 0.55)",
            "accent": "#0f172a",
            "chip": "#0ea5e9",
        },
        "good": {
            "border": "rgba(34, 197, 94, 0.7)",
            "accent": "#15803d",
            "chip": "#22c55e",
        },
        "warn": {
            "border": "rgba(234, 88, 12, 0.7)",
            "accent": "#b45309",
            "chip": "#f97316",
        },
        "info": {
            "border": "rgba(37, 99, 235, 0.7)",
            "accent": "#1d4ed8",
            "chip": "#0ea5e9",
        },
    }
    c = tones.get(tone, tones["neutral"])

    st.markdown(
        f"""
        <div style="
            border-radius: 18px;
            padding: 1rem 1.2rem;
            background: #ffffff;
            border: 1px solid {c['border']};
            box-shadow:
                0 12px 24px rgba(15, 23, 42, 0.06);
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            min-height: 100px;
        ">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.16em;
                        color: #64748b;">
                {title}
            </div>
            <div style="font-size: 1.45rem; font-weight: 600; color: {c['accent']}; line-height: 1.1;">
                {value}
            </div>
            <div style="font-size: 0.83rem; color: #6b7280; margin-top: 0.1rem;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_boardroom_theme(fig, yaxis_money=True):
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="-apple-system, system-ui, sans-serif", color="#0f172a"),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(248, 250, 252, 0.9)",
            bordercolor="rgba(148, 163, 184, 0.5)",
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

# --------------------------------------------------------------------
# FORECAST TOTAL MERCADO (ARIMA vs SARIMA sobre Imports_Total)
# --------------------------------------------------------------------
@st.cache_data
def build_total_forecast_from_final():
    # Carga base final
    df_imports = pd.read_excel("Farma_Imports_BBDD_Final.xlsx", sheet_name="Tabla_Final")

    # Fecha desde Año / Mes, serie Imports_Total
    df = df_imports.copy()
    df["Fecha"] = pd.to_datetime(
        dict(year=df["Año"].astype(int), month=df["Mes"].astype(int), day=1)
    )
    df = df.set_index("Fecha").sort_index()
    # Para el total, suponemos que ya tienes columna Imports_Total
    # Si no existe, la construimos sumando Volumen_Imports por mes:
    if "Imports_Total" in df.columns:
        y = df["Imports_Total"]
    else:
        y = df["Volumen_Imports"].groupby(df.index).sum()

    val_start = "2023-01-01"
    val_end = "2023-12-01"
    train_end = "2022-12-01"

    y_train = y[:train_end]
    y_val = y[val_start:val_end]

    def eval_model(y_true, y_pred, nombre):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = (np.abs((y_true - y_pred) / y_true).mean()) * 100
        return {"modelo": nombre, "MAE": mae, "RMSE": rmse, "MAPE": mape}

    # ARIMA
    arima_order = (1, 1, 1)
    arima_mod = sm.tsa.ARIMA(y_train, order=arima_order)
    arima_res = arima_mod.fit()
    arima_fc_2023 = arima_res.forecast(steps=len(y_val))
    met_arima = eval_model(y_val, arima_fc_2023, "ARIMA")

    # SARIMA
    sarima_order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
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
    best_name = metrics.sort_values("MAPE").iloc[0]["modelo"]

    final_end = "2024-12-01"
    y_final = y[:final_end]

    if best_name == "ARIMA":
        best_mod = sm.tsa.ARIMA(y_final, order=arima_order)
        best_res = best_mod.fit()
        fc_24m = best_res.forecast(steps=24)
    else:  # SARIMA
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
    df_fc = pd.DataFrame(
        {
            "Fecha": fechas_fc,
            "Año": fechas_fc.year,
            "Mes": fechas_fc.month,
            "Trimestre": ((fechas_fc.month - 1) // 3) + 1,
            "Valor": fc_24m.values,
            "Tipo": "Forecast",
        }
    )

    # Histórico mensual total desde la misma base
    hist = df_imports.copy()
    hist["Fecha"] = pd.to_datetime(
        dict(year=hist["Año"].astype(int), month=hist["Mes"].astype(int), day=1)
    )
    if "Imports_Total" in hist.columns:
        hist = (
            hist.groupby("Fecha")["Imports_Total"]
            .sum()
            .reset_index()
            .sort_values("Fecha")
        )
        hist["Valor"] = hist["Imports_Total"]
    else:
        hist = (
            hist.groupby("Fecha")["Volumen_Imports"]
            .sum()
            .reset_index()
            .sort_values("Fecha")
        )
        hist["Valor"] = hist["Volumen_Imports"]
    hist["Tipo"] = "Histórico"

    totalcombined = pd.concat(
        [hist[["Fecha", "Valor", "Tipo"]], df_fc[["Fecha", "Valor", "Tipo"]]],
        ignore_index=True,
    ).sort_values("Fecha")

    return totalcombined, metrics, best_name

# --------------------------------------------------------------------
# NAVEGACIÓN PRINCIPAL
# --------------------------------------------------------------------
page = st.sidebar.radio("Navegación", list(PAGES.keys()))

# --------------------------------------------------------------------
# PÁGINA: MERCADO
# --------------------------------------------------------------------
if page == "Mercado":
    monthly, shares, destinos, tablafinal = getimportstables()

    if monthly is None or tablafinal is None:
        st.error("No se encontraron las tablas de mercado en el archivo de imports.")
    else:
        totalcombined, metrics, best_name = build_total_forecast_from_final()

        # PREPARACIÓN DE datos de monthly para algunas cosas (aunque ahora el forecast sale de la otra base)
        fecha_col = pick_col(monthly, ["Fecha"])
        total_col = pick_col(monthly, ["Imports_Total", "Valor_USD_Total", "Valor_Total"])
        year_col_m = pick_col(monthly, ["Año", "Ano", "Year"])
        month_col_m = pick_col(monthly, ["Mes", "Month"])

        if fecha_col is None and year_col_m and month_col_m:
            monthly = build_fecha_from_year_month(monthly, year_col_m, month_col_m)
            fecha_col = "Fecha"

        monthlyplot = monthly.copy()
        monthlyplot[fecha_col] = pd.to_datetime(monthlyplot[fecha_col], errors="coerce")
        monthlyplot[total_col] = pd.to_numeric(monthlyplot[total_col], errors="coerce")
        monthlyplot = monthlyplot.dropna(subset=[fecha_col, total_col])
        monthlyplot = monthlyplot.rename(columns={fecha_col: "Fecha", total_col: "Valor_USD"})
        monthlyplot = monthlyplot.sort_values("Fecha")

        year_tf = pick_col(tablafinal, ["Año", "Ano", "Year"])
        month_tf = pick_col(tablafinal, ["Mes", "Month"])
        merc_tf = pick_col(tablafinal, ["Estado_Mercancia", "Mercancia", "estado_mercancia"])
        valor_tf = pick_col(tablafinal, ["Volumen_Imports", "Valor_USD", "volumen_imports"])

        tablafinal = tablafinal.copy()
        tablafinal = build_fecha_from_year_month(tablafinal, year_tf, month_tf)
        tablafinal = tablafinal.rename(
            columns={
                "Fecha": "Fecha",
                merc_tf: "Estado_Mercancia",
                valor_tf: "Volumen_Imports",
            }
        )
        tablafinal["Estado_Mercancia"] = tablafinal["Estado_Mercancia"].apply(
            normalize_mercancia_value
        )
        tablafinal["Volumen_Imports"] = pd.to_numeric(
            tablafinal["Volumen_Imports"], errors="coerce"
        )
        tablafinal = tablafinal.dropna(subset=["Fecha", "Estado_Mercancia", "Volumen_Imports"])
        tablafinal = tablafinal.sort_values("Fecha")

        # KPIs sobre totalcombined (hist + forecast)
        last_month = totalcombined["Fecha"].max()
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
            cagrhist = (end_val / start_val) ** (1 / n_years) - 1 if start_val > 0 else 0.0
        else:
            cagrhist = 0.0

        growth_26_vs_25 = 0.0
        if not forecastyearly.empty:
            val_25 = forecastyearly.loc[forecastyearly["Año"] == 2025, "Valor"]
            val_26 = forecastyearly.loc[forecastyearly["Año"] == 2026, "Valor"]
            if not val_25.empty and not val_26.empty and val_25.iloc[0] > 0:
                growth_26_vs_25 = val_26.iloc[0] / val_25.iloc[0] - 1

        if "Estado_Mercancia" in tablafinal.columns:
            last_month_rows = tablafinal[tablafinal["Fecha"] == last_month]
            mix_last_month = (
                last_month_rows.groupby("Estado_Mercancia")["Volumen_Imports"]
                .sum()
                .reset_index()
            )
            mix_last_month["Share"] = (
                mix_last_month["Volumen_Imports"]
                / mix_last_month["Volumen_Imports"].sum()
            )
            mix_last_month_sorted = mix_last_month.sort_values("Share", ascending=False)
            top2_share = mix_last_month_sorted["Share"].head(2).sum()
            leading_mercancia = (
                mix_last_month_sorted.iloc[0]["Estado_Mercancia"]
                if not mix_last_month_sorted.empty
                else "N/A"
            )
        else:
            mix_last_month_sorted = pd.DataFrame()
            top2_share = 0.0
            leading_mercancia = "N/A"

        if destinos is not None and not destinos.empty:
            dest_share_col = pick_col(destinos, ["Share_Imports_2024", "Share_Imports"])
            dest_name_col = pick_col(destinos, ["Municipio", "Destino"])
            estado_col = pick_col(destinos, ["Estado"])
            if dest_share_col and dest_name_col:
                destinos_aux = destinos.copy()
                destinos_aux[dest_share_col] = pd.to_numeric(
                    destinos_aux[dest_share_col], errors="coerce"
                )
                destinos_aux = destinos_aux.dropna(subset=[dest_share_col])
                destinos_aux = destinos_aux.sort_values(dest_share_col, ascending=False)
                leading_destino = (
                    destinos_aux.iloc[0][dest_name_col] if not destinos_aux.empty else "N/A"
                )
            else:
                leading_destino = "N/A"
        else:
            leading_destino = "N/A"

        last_month_total = (
            totalcombined[totalcombined["Fecha"] == last_month]["Valor"].sum()
        )

        # HEADER
        st.markdown(
            f"""
            <div class="cl-header">
                <div class="cl-header-left">
                    <div class="cl-header-pill">
                        <span class="cl-header-pill-dot"></span>
                        <span>Mercado Farma | Imports</span>
                    </div>
                    <div class="cl-header-title">
                        Outlook de mercado y mix de importaciones
                    </div>
                    <div class="cl-header-subtitle">
                        Lectura ejecutiva de tamaño de mercado, impulso proyectado y concentración por categoría y destino.
                    </div>
                </div>
                <div class="cl-header-right">
                    <div>
                        <div class="cl-header-metric-label">Modelo ganador forecast</div>
                        <div class="cl-header-metric-value">
                            {best_name}
                        </div>
                    </div>
                    <div>
                        <div class="cl-header-metric-label">Horizonte de forecast</div>
                        <div class="cl-header-metric-value">
                            {format_period_label(histyearly['Año'].min(), forecastyearly['Año'].max()) if not histyearly.empty and not forecastyearly.empty else "2024-2026"}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # FILTROS (sin categoría focal)
        st.markdown(
            """
            <div style="margin-top: 0.4rem; margin-bottom: 0.6rem;">
                <div style="font-size: 0.78rem; text-transform: uppercase;
                            letter-spacing: 0.14em; color: #6b7280;
                            margin-bottom: 0.25rem;">
                    Filtros de lectura
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        f2, f3, f4 = st.columns([1.2, 1.2, 1.0])

        with f2:
            series_view = st.radio(
                "Vista de serie",
                options=["Ambos", "Histórico", "Forecast"],
                horizontal=True,
            )

        with f3:
            topn_destinos = st.slider(
                "Top destinos a mostrar",
                min_value=3,
                max_value=20,
                value=TOPNDEFAULT,
                step=1,
            )

        with f4:
            mostrar_meses = st.selectbox(
                "Ventana temporal",
                options=["Últimos 12 meses", "Últimos 24 meses", "Todo el histórico"],
                index=0,
            )

        # KPIs
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            render_kpi_card(
                title="Tamaño de mercado",
                value=f"{format_compact_number(total_hist_2024)} USD",
                subtitle="Imports acumuladas 2024 (histórico)",
                tone="info",
            )

        with kpi_col2:
            render_kpi_card(
                title="Crecimiento estructural",
                value=f"{cagrhist * 100:.1f}%",
                subtitle=f"CAGR {int(histyearly['Año'].min())}-{int(histyearly['Año'].max())}" if not histyearly.empty else "CAGR histórico",
                tone="good" if cagrhist >= 0 else "warn",
            )

        with kpi_col3:
            render_kpi_card(
                title="Impulso proyectado",
                value=f"{growth_26_vs_25 * 100:.1f}%",
                subtitle="Cambio esperado 2026 vs 2025",
                tone="good" if growth_26_vs_25 >= 0 else "warn",
            )

        with kpi_col4:
            render_kpi_card(
                title="Concentración del mix",
                value=f"{top2_share * 100:.1f}%",
                subtitle=f"Top 2 mercancías en {format_month_label(last_month)}",
                tone="neutral",
            )

        # NARRATIVA
        st.markdown(
            f"""
            <div style="margin-top: 0.8rem; margin-bottom: 0.8rem;
                        padding: 0.9rem 1rem;
                        border-radius: 16px;
                        background: #ffffff;
                        border: 1px solid rgba(148, 163, 184, 0.5);
                        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.04);">
                <div style="font-size: 0.78rem; text-transform: uppercase;
                            letter-spacing: 0.16em; color: #6b7280; margin-bottom: 0.3rem;">
                    Lectura ejecutiva
                </div>
                <div style="font-size: 0.88rem; color: #111827;">
                    El mercado de importaciones farma cerró 
                    <span style="color:#0ea5e9; font-weight:500;">
                        {format_month_label(last_month)}
                    </span>
                    con un tamaño aproximado de 
                    <span style="color:#0ea5e9; font-weight:500;">
                        {format_compact_number(last_month_total)} USD
                    </span>.
                    El modelo de forecast seleccionado es 
                    <span style="color:#0369a1; font-weight:500;">
                        {best_name}
                    </span>,
                    con un impulso proyectado de 
                    <span style="color:#f97316; font-weight:500;">
                        {growth_26_vs_25 * 100:.1f}%
                    </span> en 2026 vs 2025. La categoría líder es 
                    <span style="color:#0369a1; font-weight:500;">
                        {business_label(leading_mercancia)}
                    </span>
                    y el principal destino es 
                    <span style="color:#15803d; font-weight:500;">
                        {leading_destino}
                    </span>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # PESTAÑAS
        tab_total, tab_categorias, tab_destinos = st.tabs(
            ["Visión total", "Categorías", "Destinos"]
        )

        # Visión total
        with tab_total:
            st.markdown("### Visión total de mercado")
            df_view = filter_series_view(totalcombined, "Tipo", series_view)
            if mostrar_meses == "Últimos 12 meses":
                cutoff = df_view["Fecha"].max() - pd.DateOffset(months=12)
                df_view = df_view[df_view["Fecha"] >= cutoff]
            elif mostrar_meses == "Últimos 24 meses":
                cutoff = df_view["Fecha"].max() - pd.DateOffset(months=24)
                df_view = df_view[df_view["Fecha"] >= cutoff]

            fig_total = px.line(
                df_view,
                x="Fecha",
                y="Valor",
                color="Tipo",
                color_discrete_map={
                    "Histórico": "#0ea5e9",
                    "Forecast": "#22c55e",
                },
                markers=True,
            )
            fig_total = apply_boardroom_theme(fig_total, yaxis_money=True)
            st.plotly_chart(fig_total, use_container_width=True, config={"displayModeBar": False})

        # Categorías
        with tab_categorias:
            st.markdown("### Foco por categoría")
            col_c1, col_c2 = st.columns([1.6, 1.2])

            serie_merc = tablafinal.copy()
            serie_merc = serie_merc[serie_merc["Estado_Mercancia"].isin(MERCANCIAORDER)]
            serie_merc = (
                serie_merc.groupby(["Fecha", "Estado_Mercancia"])["Volumen_Imports"]
                .sum()
                .reset_index()
            )

            with col_c1:
                st.caption("Evolución por categoría (volumen importado)")
                fig_merc = px.line(
                    serie_merc,
                    x="Fecha",
                    y="Volumen_Imports",
                    color="Estado_Mercancia",
                    color_discrete_map=COLORMAP,
                )
                fig_merc.for_each_trace(
                    lambda t: t.update(
                        name=business_label(t.name),
                        legendgroup=business_label(t.name),
                        hovertemplate=t.hovertemplate.replace(t.name, business_label(t.name)),
                    )
                )
                fig_merc = apply_boardroom_theme(fig_merc, yaxis_money=True)
                st.plotly_chart(fig_merc, use_container_width=True, config={"displayModeBar": False})

            with col_c2:
                st.caption(f"Mix de categorías en {format_month_label(last_month)}")
                mix_plot = mix_last_month_sorted.copy()
                fig_mix = px.pie(
                    mix_plot,
                    values="Share",
                    names="Estado_Mercancia",
                    color="Estado_Mercancia",
                    color_discrete_map=COLORMAP,
                )
                fig_mix.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    hovertemplate="%{label}: %{percent:.1%}<extra></extra>",
                )
                fig_mix = apply_boardroom_theme(fig_mix, yaxis_money=False)
                st.plotly_chart(fig_mix, use_container_width=True, config={"displayModeBar": False})

        # Destinos
        with tab_destinos:
            st.markdown("### Foco destinos principales")

            if destinos is None or destinos.empty:
                st.info("Aún no hay información de destinos procesable para esta vista.")
            else:
                dest_share_col = pick_col(destinos, ["Share_Imports_2024", "Share_Imports"])
                dest_name_col = pick_col(destinos, ["Municipio", "Destino"])
                estado_col = pick_col(destinos, ["Estado"])

                if dest_share_col and dest_name_col:
                    destinos_plot = destinos.copy()
                    destinos_plot[dest_share_col] = pd.to_numeric(
                        destinos_plot[dest_share_col], errors="coerce"
                    )
                    destinos_plot = destinos_plot.dropna(subset=[dest_share_col])
                    destinos_plot = destinos_plot.sort_values(dest_share_col, ascending=False)
                    destinos_plot = destinos_plot.head(topn_destinos)

                    col_d1, col_d2 = st.columns([1.4, 1.2])

                    with col_d1:
                        st.caption(f"Top {topn_destinos} destinos por participación de imports")
                        fig_dest = px.bar(
                            destinos_plot.sort_values(dest_share_col, ascending=True),
                            x=dest_share_col,
                            y=dest_name_col,
                            orientation="h",
                            color=estado_col if estado_col else dest_name_col,
                            color_discrete_sequence=[
                                "#0ea5e9", "#0369a1", "#22c55e",
                                "#16a34a", "#f97316"
                            ],
                        )
                        fig_dest = apply_boardroom_theme(fig_dest, yaxis_money=False)
                        fig_dest.update_xaxes(tickformat=".0%")
                        st.plotly_chart(fig_dest, use_container_width=True, config={"displayModeBar": False})

                    with col_d2:
                        st.caption("Distribución relativa de destinos (pie)")
                        fig_pie = px.pie(
                            destinos_plot,
                            values=dest_share_col,
                            names=dest_name_col,
                            color=dest_name_col,
                            color_discrete_sequence=[
                                "#0ea5e9", "#0369a1", "#22c55e",
                                "#16a34a", "#f97316", "#facc15"
                            ],
                        )
                        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                        fig_pie = apply_boardroom_theme(fig_pie, yaxis_money=False)
                        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("No se identificaron columnas de share/destino compatibles en la tabla de destinos.")

# --------------------------------------------------------------------
# PÁGINA: RIESGO
# --------------------------------------------------------------------
elif page == "Riesgo":
    df_risk = loadriskdata()
    if df_risk is None or df_risk.empty:
        st.error("No se pudo cargar la base de riesgo territorial.")
    else:
        st.title("Riesgo territorial")
        st.write("Módulo en desarrollo. La base de riesgo se ha cargado correctamente.")
        st.dataframe(df_risk.head())

# --------------------------------------------------------------------
# PÁGINA: CLUSTERS
# --------------------------------------------------------------------
elif page == "Clusters":
    producto80, exportadores20, integrado = getcompanytables()

    st.title("Clusters competitivos")
    if producto80 is None and exportadores20 is None and integrado is None:
        st.error("No se encontraron las tablas de clusters en el archivo de empresas.")
    else:
        st.write("Módulo en desarrollo. Tablas identificadas:")
        if producto80 is not None:
            st.success("Tabla producto80 cargada.")
            st.dataframe(producto80.head())
        if exportadores20 is not None:
            st.success("Tabla exportadores20 cargada.")
            st.dataframe(exportadores20.head())
        if integrado is not None:
            st.success("Tabla integrado cargada.")
            st.dataframe(integrado.head())
