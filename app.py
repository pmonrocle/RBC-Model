import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.linalg import ordqz
 
st.set_page_config(page_title="RBC — Correlaciones cruzadas", layout="centered")
 
# ── Benchmark fijo ─────────────────────────────────────────────────────────
BENCH = dict(alpha=0.36, beta=0.99, delta=0.025, rho=0.90,
             sig_eps=0.01, l_ss=0.33, T=50_000, seed=7)
 
# ── Sidebar ────────────────────────────────────────────────────────────────
 
# Benchmark 
st.sidebar.markdown("**Benchmark fijo**")
bench_params = [
    (r"\alpha",        BENCH["alpha"]),
    (r"\beta",         BENCH["beta"]),
    (r"\delta",        BENCH["delta"]),
    (r"\rho",          BENCH["rho"]),
    (r"\sigma_{\varepsilon}", rf"{BENCH['sig_eps']*100:.0f}\%"),
    (r"l_{ss}",         BENCH["l_ss"]),
]
for sym, val in bench_params:
    st.sidebar.latex(rf"{sym} = {val}")
 
st.sidebar.divider()
 
st.sidebar.markdown("**Parámetros libres**")
sigma = st.sidebar.slider("σ — aversión al riesgo (consumo)", 0.5, 6.0, 2.0, 0.1)
psi   = st.sidebar.slider("ψ — curvatura del ocio",           0.5, 5.0, 1.0, 0.1)
 
st.sidebar.divider()
max_lag = st.sidebar.slider("Lags máximos", 2, 8, 5)
run_btn = st.sidebar.button("▶ Simular", type="primary", use_container_width=True)
 
 
# ── Model ──────────────────────────────────────────────────────────────────
def build_state_space(sigma, psi, bench):
    a, b, d, rho, lss = (bench["alpha"], bench["beta"], bench["delta"],
                         bench["rho"],   bench["l_ss"])
 
    eta         = a + psi * (lss / (1.0 - lss))
    phi_        = (1.0 - b * (1.0 - d)) / sigma
    s           = (1.0 - a) / eta
    lam         = (1.0 / a) * (1.0 / b - 1.0 + d)
    a_theta, a_k = lam, 1.0 / b
    a_l         = ((1.0 - a) / a) * (1.0 / b - 1.0 + d)
    a_c         = lam - d
    kappa_c     = 1.0 + phi_ * sigma * s
    kappa_k     = phi_ * ((a - 1.0) + a * s)
    kappa_theta = phi_ * (1.0 + s)
    b11 = a_k + (a * a_l) / eta
    b12 = -a_c - (sigma * a_l) / eta
    c1  = a_theta + a_l / eta
 
    A = np.array([[1.,       0.      ], [kappa_k, -kappa_c]])
    B = np.array([[b11,      b12     ], [0.,      -1.     ]])
    C = np.array([[c1], [-kappa_theta * rho]])
 
    Ar = np.zeros((3, 3)); Br = np.zeros((3, 3))
    Ar[0,1]=A[0,0]; Ar[0,0]=A[0,1]; Br[0,1]=B[0,0]; Br[0,0]=B[0,1]; Br[0,2]=C[0,0]
    Ar[1,1]=A[1,0]; Ar[1,0]=A[1,1]; Br[1,1]=B[1,0]; Br[1,0]=B[1,1]; Br[1,2]=C[1,0]
    Ar[2,2]=1.0;    Br[2,2]=rho
 
    p = [1, 2, 0]
    _, _, al, bt, _, Z = ordqz(Br[np.ix_(p,p)], Ar[np.ix_(p,p)], sort="iuc")
    if (np.abs(al / bt) < 1 - 1e-10).sum() != 2:
        raise ValueError("Blanchard-Kahn no satisfecho. Ajusta σ o ψ.")
 
    Z11 = Z[:2, :2]; Z21 = Z[2:, :2]
    phi_k, phi_theta = (Z21 @ np.linalg.inv(Z11)).reshape(-1)
 
    w_lk  = (a   - sigma * phi_k)    / eta
    w_lth = (1.0 - sigma * phi_theta) / eta
 
    M  = np.array([[a_k + a_l*w_lk  - a_c*phi_k,
                    a_theta + a_l*w_lth - a_c*phi_theta],
                   [0.0, rho]])
    wc = np.array([phi_k,   phi_theta])
    wl = np.array([w_lk,    w_lth   ])
    wy = np.array([a,       1.0     ]) + (1.0 - a) * wl
    wi = (lam / d) * wy - ((lam - d) / d) * wc
 
    return M, wc, wl, wy, wi
 
 
def simulate(M, wc, wl, wy, wi, bench):
    sig_eps, T, seed = bench["sig_eps"], bench["T"], bench["seed"]
    rng  = np.random.default_rng(seed)
    burn = 200
    eps  = rng.normal(0.0, sig_eps, T + burn)
    s    = np.zeros((2, T + burn + 1))
    for t in range(T + burn):
        s[:, t+1] = M @ s[:, t] + np.array([0.0, eps[t]])
    s = s[:, burn+1:]
    return dict(y=wy @ s, c=wc @ s, k=s[0], i=wi @ s, l=wl @ s)
 
 
def xcorr(y, x, lag):
    n = len(y)
    a, b = (y[:n-lag], x[lag:]) if lag >= 0 else (y[-lag:], x[:n+lag])
    return float(np.corrcoef(a, b)[0, 1])
 
 
# ── Run ────────────────────────────────────────────────────────────────────
if run_btn or "sim" not in st.session_state:
    with st.spinner("Simulando…"):
        try:
            M, wc, wl, wy, wi = build_state_space(sigma, psi, BENCH)
            sim = simulate(M, wc, wl, wy, wi, BENCH)
            st.session_state.update(dict(sim=sim, sigma=sigma, psi=psi))
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()
 
sim   = st.session_state["sim"]
sig_  = st.session_state["sigma"]
psi_  = st.session_state["psi"]
lny   = sim["y"]
 
lags  = list(range(-max_lag, max_lag + 1))
lag0  = lags.index(0)
COLORS = dict(c="#60a5fa", k="#fb923c", i="#4ade80", l="#f87171")
NAMES  = dict(c="ln(c/c_ss)", k="ln(k/k_ss)", i="ln(i/i_ss)", l="ln(l/l_ss)")
cc     = {v: [xcorr(lny, sim[v], lag) for lag in lags] for v in ["c","k","i","l"]}
sy     = float(np.std(lny, ddof=1))
 
# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    f"## Correlación cruzada con ln(y/y<sub>ss</sub>) "
    f"<small style='color:#8b90a0'>σ={sig_}, ψ={psi_}</small>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='font-family:monospace;font-size:12px;color:#8b90a0'>"
    f"α={BENCH['alpha']} · β={BENCH['beta']} · δ={BENCH['delta']} · "
    f"ρ={BENCH['rho']} · σ_ε={BENCH['sig_eps']*100:.0f}% · "
    f"T={BENCH['T']:,} · l_ss={BENCH['l_ss']}</p>",
    unsafe_allow_html=True,
)
 
# ── Tabs ───────────────────────────────────────────────────────────────────
for tab, keys in zip(
    st.tabs(["Todas", "Consumo", "Capital", "Inversión", "Horas"]),
    [["c","k","i","l"], ["c"], ["k"], ["i"], ["l"]],
):
    with tab:
        fig = go.Figure()
        fig.add_hline(y=0, line_color="rgba(150,150,150,0.3)", line_width=1)
        fig.add_vline(x=0, line_color="rgba(150,150,150,0.3)", line_width=1,
                      line_dash="dot")
        for v in keys:
            fig.add_trace(go.Scatter(
                x=lags, y=cc[v], name=NAMES[v],
                mode="lines+markers",
                line=dict(color=COLORS[v], width=2.5),
                marker=dict(size=7, color=COLORS[v]),
                hovertemplate="lag %{x}: %{y:.4f}<extra>" + NAMES[v] + "</extra>",
            ))
        fig.update_layout(
            xaxis=dict(title="Lag k  ·  corr(y_t , x_{t+k})",
                       tickmode="array", tickvals=lags,
                       gridcolor="rgba(200,200,200,0.08)"),
            yaxis=dict(title="Correlación", range=[-0.25, 1.05],
                       gridcolor="rgba(200,200,200,0.08)"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=40, b=50),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
 
# ── Stats ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
 
with col1:
    st.markdown("**Volatilidades relativas · σ(x)/σ(y)**")
    st.markdown(f"σ(y) &nbsp; **{sy*100:.2f}%**", unsafe_allow_html=True)
    for v, lbl in [("c","σ(c)/σ(y)"), ("k","σ(k)/σ(y)"),
                   ("i","σ(i)/σ(y)"), ("l","σ(l)/σ(y)")]:
        st.markdown(f"{lbl} &nbsp; **{np.std(sim[v], ddof=1)/sy:.3f}**",
                    unsafe_allow_html=True)
 
with col2:
    st.markdown("**Correlación contemporánea · lag 0**")
    for v, lbl in [("c","corr(y, c)"), ("k","corr(y, k)"),
                   ("i","corr(y, i)"), ("l","corr(y, l)")]:
        val  = cc[v][lag0]
        sign = "+" if val >= 0 else ""
        st.markdown(f"{lbl} &nbsp; **{sign}{val:.4f}**", unsafe_allow_html=True)
 
# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"σ={sig_} → mayor aversión al riesgo suaviza el consumo (σ(c)/σ(y) ↓). "
    f"ψ={psi_} → mayor curvatura amortigua la respuesta del trabajo (σ(l)/σ(y) ↓). "
    "Inversión: la más procíclica en lag 0. "
    "Capital: pico en lags positivos (variable predeterminada). "
    "lag k > 0 → x adelanta a y k periodos."
)
