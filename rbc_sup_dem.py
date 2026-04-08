import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.linalg import ordqz

st.set_page_config(page_title="RBC — Correlaciones cruzadas", layout="centered")

# =========================================================
# Parámetros base
# =========================================================
BENCH = dict(
    alpha=0.36,
    beta=0.99,
    delta=0.025,
    rho=0.90,          # persistencia shock tecnológico
    rho_a=0.90,        # persistencia shock de demanda
    sig_eps=0.01,      # innovación shock tecnológico 
    sig_zeta=0.01,     # innovación shock demanda 
    l_ss=0.33,
    T=50_000,
    seed=7
)

bench_run = BENCH.copy()

# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown("**Parámetros fijos**")

if shock_type == "Shock de oferta":
    st.sidebar.markdown(
        rf"$\alpha={BENCH['alpha']}$ &nbsp;&nbsp; "
        rf"$\beta={BENCH['beta']}$ &nbsp;&nbsp; "
        rf"$\delta={BENCH['delta']}$ &nbsp;&nbsp; "
        rf"$\sigma_{{\varepsilon}}={BENCH['sig_eps']*100:.1f}\%$ &nbsp;&nbsp; "
        rf"$l_{{ss}}={BENCH['l_ss']}$",
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        rf"$\alpha={BENCH['alpha']}$ &nbsp;&nbsp; "
        rf"$\beta={BENCH['beta']}$ &nbsp;&nbsp; "
        rf"$\delta={BENCH['delta']}$ &nbsp;&nbsp; "
        rf"$\sigma_{{\zeta}}={BENCH['sig_zeta']*100:.1f}\%$ &nbsp;&nbsp; "
        rf"$l_{{ss}}={BENCH['l_ss']}$",
        unsafe_allow_html=True,
    )



st.sidebar.divider()

st.sidebar.markdown("**Elección del shock**")
shock_type = st.sidebar.radio(
    "Tipo de shock",
    ["Shock de oferta", "Shock de demanda"],
    index=0,
    key="shock_selector_main"
)


st.sidebar.divider()

st.sidebar.markdown("**Parámetros libres**")
sigma = st.sidebar.slider("σ — aversión al riesgo (consumo)", 0.5, 6.0, 2.0, 0.1)
psi   = st.sidebar.slider("ψ — curvatura del ocio", 0.5, 5.0, 1.0, 0.1)

if shock_type == "Shock de oferta":
    rho_theta = st.sidebar.slider(
        "ρ_θ — persistencia shock tecnológico",
        0.50, 0.99, BENCH["rho"], 0.01
    )
    bench_run["rho"] = rho_theta
else:
    rho_a = st.sidebar.slider(
        "ρ_a — persistencia shock de demanda",
        0.50, 0.99, BENCH["rho_a"], 0.01
    )
    bench_run["rho_a"] = rho_a

st.sidebar.divider()
max_lag = st.sidebar.slider("Lags máximos", 2, 8, 5)
run_btn = st.sidebar.button("Simular", type="primary", use_container_width=True)

# =========================================================
# Construcción del sistema linealizado con dos shocks
# Estados: s_t = [k_t, theta_t, a_t]
# =========================================================
def _build_state_space_two_shocks_common(sigma_eff, eta, sigma_for_labor, bench):
    a     = bench["alpha"]
    b     = bench["beta"]
    d     = bench["delta"]
    rho   = bench["rho"]
    rho_a = bench["rho_a"]

    phi_         = (1.0 - b * (1.0 - d)) / sigma_eff
    s_aux        = (1.0 - a) / eta
    lam          = (1.0 / a) * (1.0 / b - 1.0 + d)
    a_theta      = lam
    a_k          = 1.0 / b
    a_l          = ((1.0 - a) / a) * (1.0 / b - 1.0 + d)
    a_c          = lam - d
    kappa_c      = 1.0 + phi_ * sigma_eff * s_aux
    kappa_k      = phi_ * ((a - 1.0) + a * s_aux)
    kappa_theta  = phi_ * (1.0 + s_aux)
    b11          = a_k + (a * a_l) / eta
    b12          = -a_c - (sigma_eff * a_l) / eta
    c1           = a_theta + a_l / eta

    demand_wedge = (1.0 - rho_a) / sigma_eff

    G0 = np.array([
        [1.0,       0.0, 0.0,       0.0],
        [kappa_k,   0.0, 0.0, -kappa_c],
        [0.0,       1.0, 0.0,       0.0],
        [0.0,       0.0, 1.0,       0.0],
    ])

    G1 = np.array([
        [b11,                c1,           0.0,   b12],
        [0.0, -kappa_theta * rho, demand_wedge,  -1.0],
        [0.0,                rho,          0.0,   0.0],
        [0.0,                0.0,        rho_a,   0.0],
    ])

    _, _, al, bt, _, Z = ordqz(G1, G0, sort="iuc")
    if (np.abs(al / bt) < 1 - 1e-10).sum() != 3:
        raise ValueError("Blanchard-Kahn no satisfecho en el modelo con dos shocks.")

    Z11 = Z[:3, :3]
    Z21 = Z[3:, :3]
    phi_k, phi_theta, phi_a = (Z21 @ np.linalg.inv(Z11)).reshape(-1)

    wc = np.array([phi_k, phi_theta, phi_a])

    wl = np.array([a / eta, 1.0 / eta, 0.0]) - (sigma_for_labor / eta) * wc
    wy = np.array([a, 1.0, 0.0]) + (1.0 - a) * wl
    wi = (lam / d) * wy - ((lam - d) / d) * wc

    M = np.array([
        [b11 + b12 * phi_k, c1 + b12 * phi_theta, b12 * phi_a],
        [0.0,               rho,                  0.0],
        [0.0,               0.0,                rho_a],
    ])

    return M, wc, wl, wy, wi


def build_state_space_divisible(sigma, psi, bench):
    a = bench["alpha"]
    lss = bench["l_ss"]
    eta = a + psi * (lss / (1.0 - lss))
    return _build_state_space_two_shocks_common(
        sigma_eff=sigma,
        eta=eta,
        sigma_for_labor=sigma,
        bench=bench
    )


def build_state_space_indivisible(bench):
    a = bench["alpha"]
    eta = a
    return _build_state_space_two_shocks_common(
        sigma_eff=1.0,
        eta=eta,
        sigma_for_labor=1.0,
        bench=bench
    )

# =========================================================
# Simulación
# =========================================================
def simulate(M, wc, wl, wy, wi, bench, shock_type):
    T = bench["T"]
    seed = bench["seed"]
    sig_eps = bench["sig_eps"]
    sig_zeta = bench["sig_zeta"]

    rng = np.random.default_rng(seed)
    burn = 200

    eps = np.zeros(T + burn)
    zeta = np.zeros(T + burn)

    if shock_type == "Shock de oferta":
        eps = rng.normal(0.0, sig_eps, T + burn)
    else:
        zeta = rng.normal(0.0, sig_zeta, T + burn)

    s = np.zeros((3, T + burn + 1))   # [k, theta, a]

    for t in range(T + burn):
        s[:, t+1] = M @ s[:, t] + np.array([0.0, eps[t], zeta[t]])

    s = s[:, burn+1:]

    return dict(
        y=wy @ s,
        c=wc @ s,
        k=s[0],
        i=wi @ s,
        l=wl @ s,
        theta=s[1],
        a=s[2]
    )

# =========================================================
# Correlaciones cruzadas
# =========================================================
def xcorr(y, x, lag):
    n = len(y)
    a, b = (y[:n-lag], x[lag:]) if lag >= 0 else (y[-lag:], x[:n+lag])
    return float(np.corrcoef(a, b)[0, 1])

def build_corrs(sim, max_lag):
    lags = list(range(-max_lag, max_lag + 1))
    lny = sim["y"]
    cc = {v: [xcorr(lny, sim[v], lag) for lag in lags] for v in ["c", "k", "i", "l"]}
    return lags, cc

# =========================================================
# Tablas
# =========================================================
def safe_corr(x, y):
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    if sx < 1e-14 or sy < 1e-14:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def fmt_corr(x):
    return "—" if pd.isna(x) else f"{x:+.4f}"

def fmt_pct(x):
    return f"{x*100:.2f}%"

def build_stats(sim, cc, lags):
    sy = float(np.std(sim["y"], ddof=1))
    stheta = float(np.std(sim["theta"], ddof=1))
    sa = float(np.std(sim["a"], ddof=1))
    lag0 = lags.index(0)

    vol_df = pd.DataFrame({
        "Variable": ["σ(y)", "σ(c)/σ(y)", "σ(k)/σ(y)", "σ(i)/σ(y)", "σ(l)/σ(y)", "σ(θ)", "σ(a)"],
        "Valor": [
            fmt_pct(sy),
            f"{np.std(sim['c'], ddof=1)/sy:.3f}",
            f"{np.std(sim['k'], ddof=1)/sy:.3f}",
            f"{np.std(sim['i'], ddof=1)/sy:.3f}",
            f"{np.std(sim['l'], ddof=1)/sy:.3f}",
            fmt_pct(stheta),
            fmt_pct(sa),
        ],
    }).set_index("Variable")

    corr_df = pd.DataFrame({
        "Variable": ["corr(y, c)", "corr(y, k)", "corr(y, i)", "corr(y, l)", "corr(y, θ)", "corr(y, a)"],
        "Valor": [
            fmt_corr(cc["c"][lag0]),
            fmt_corr(cc["k"][lag0]),
            fmt_corr(cc["i"][lag0]),
            fmt_corr(cc["l"][lag0]),
            fmt_corr(safe_corr(sim["y"], sim["theta"])),
            fmt_corr(safe_corr(sim["y"], sim["a"])),
        ],
    }).set_index("Variable")

    return vol_df, corr_df

# =========================================================
# Estilo gráficos
# =========================================================
COLORS = dict(c="#60a5fa", k="#fb923c", i="#4ade80", l="#f87171")
NAMES  = dict(c="ln(c/c_ss)", k="ln(k/k_ss)", i="ln(i/i_ss)", l="ln(l/l_ss)")

def draw_model_block(title, sim, cc, lags, note=""):
    st.markdown(f"## {title}")
    if note:
        st.markdown(note)

    tabs = st.tabs(["Todas", "Consumo", "Capital", "Inversión", "Trabajo"])

    for tab, keys in zip(
        tabs,
        [["c", "k", "i", "l"], ["c"], ["k"], ["i"], ["l"]],
    ):
        with tab:
            fig = go.Figure()
            fig.add_hline(y=0, line_color="rgba(150,150,150,0.3)", line_width=1)
            fig.add_vline(x=0, line_color="rgba(150,150,150,0.3)", line_width=1, line_dash="dot")

            for v in keys:
                fig.add_trace(go.Scatter(
                    x=lags,
                    y=cc[v],
                    name=NAMES[v],
                    mode="lines+markers",
                    line=dict(color=COLORS[v], width=2.5),
                    marker=dict(size=7, color=COLORS[v]),
                    hovertemplate="lag %{x}: %{y:.4f}<extra>" + NAMES[v] + "</extra>",
                ))

            fig.update_layout(
                xaxis=dict(
                    title="Lag k · corr(y_t, x_{t+k})",
                    tickmode="array",
                    tickvals=lags,
                    gridcolor="rgba(200,200,200,0.08)"
                ),
                yaxis=dict(
                    title="Correlación",
                    range=[-0.25, 1.05],
                    gridcolor="rgba(200,200,200,0.08)"
                ),
                legend=dict(orientation="h", y=1.12, x=0),
                margin=dict(l=50, r=20, t=40, b=50),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

    vol_df, corr_df = build_stats(sim, cc, lags)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Volatilidades**")
        st.dataframe(vol_df, use_container_width=True)

    with col2:
        st.markdown("**Correlación contemporánea** · lag 0")
        st.dataframe(corr_df, use_container_width=True)

# =========================================================
# Run
# =========================================================
if run_btn or "sim_div" not in st.session_state:
    with st.spinner("Simulando…"):
        try:
            M_div, wc_div, wl_div, wy_div, wi_div = build_state_space_divisible(sigma, psi, bench_run)
            sim_div = simulate(M_div, wc_div, wl_div, wy_div, wi_div, bench_run, shock_type)
            lags_div, cc_div = build_corrs(sim_div, max_lag)

            M_ind, wc_ind, wl_ind, wy_ind, wi_ind = build_state_space_indivisible(bench_run)
            sim_ind = simulate(M_ind, wc_ind, wl_ind, wy_ind, wi_ind, bench_run, shock_type)
            lags_ind, cc_ind = build_corrs(sim_ind, max_lag)

            st.session_state.update(
                sim_div=sim_div,
                lags_div=lags_div,
                cc_div=cc_div,
                sim_ind=sim_ind,
                lags_ind=lags_ind,
                cc_ind=cc_ind,
                sigma=sigma,
                psi=psi,
                shock_type=shock_type
            )

        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()

# =========================================================
# Recuperar resultados
# =========================================================
sim_div  = st.session_state["sim_div"]
lags_div = st.session_state["lags_div"]
cc_div   = st.session_state["cc_div"]

sim_ind  = st.session_state["sim_ind"]
lags_ind = st.session_state["lags_ind"]
cc_ind   = st.session_state["cc_ind"]

sig_ = st.session_state["sigma"]
psi_ = st.session_state["psi"]

# =========================================================
# Header
# =========================================================
st.markdown("# RBC — Correlaciones cruzadas")

main_tab1, main_tab2 = st.tabs(["Trabajo divisible", "Trabajo indivisible"])

with main_tab1:
    draw_model_block(
        title="Trabajo divisible",
        sim=sim_div,
        cc=cc_div,
        lags=lags_div,
        note=f"Parámetros libres actuales: $\\sigma={sig_:.1f}$, $\\psi={psi_:.1f}$"
    )

with main_tab2:
    draw_model_block(
        title="Trabajo indivisible",
        sim=sim_ind,
        cc=cc_ind,
        lags=lags_ind,
        note="$\\sigma=1$ y sin parámetro de curvatura del ocio."
    )
