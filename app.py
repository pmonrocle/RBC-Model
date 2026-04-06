import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.linalg import ordqz
 
st.set_page_config(page_title="RBC — Correlaciones cruzadas", layout="centered")
 
# =========================================================
# Parámetros fijos
# =========================================================
BENCH = dict(
    alpha=0.36,
    beta=0.99,
    delta=0.025,
    rho=0.90,
    sig_eps=0.01,
    l_ss=0.33,
    T=50_000,
    seed=7
)
 
# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown("**Parámetros fijos**")
st.sidebar.markdown(
    rf"$\alpha={BENCH['alpha']}$ &nbsp;&nbsp; $\beta={BENCH['beta']}$  "
    rf"$\delta={BENCH['delta']}$ &nbsp;&nbsp; $\rho={BENCH['rho']}$  "
    rf"$\sigma_{{\varepsilon}}={BENCH['sig_eps']*100:.0f}\%$ &nbsp;&nbsp; $l_{{ss}}={BENCH['l_ss']}$",
    unsafe_allow_html=True,
)
 
st.sidebar.divider()
 
st.sidebar.markdown("**Parámetros libres (solo trabajo divisible)**")
sigma = st.sidebar.slider("σ — aversión al riesgo (consumo)", 0.5, 6.0, 2.0, 0.1)
psi   = st.sidebar.slider("ψ — curvatura del ocio",           0.5, 5.0, 1.0, 0.1)
 
st.sidebar.divider()
max_lag = st.sidebar.slider("Lags máximos", 2, 8, 5)
run_btn = st.sidebar.button("▶ Simular", type="primary", use_container_width=True)
 
 
# =========================================================
# Modelo original: trabajo divisible
# =========================================================
def build_state_space_divisible(sigma, psi, bench):
    a, b, d, rho, lss = (
        bench["alpha"],
        bench["beta"],
        bench["delta"],
        bench["rho"],
        bench["l_ss"]
    )
 
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
 
    Z11 = Z[:2, :2]
    Z21 = Z[2:, :2]
    phi_k, phi_theta = (Z21 @ np.linalg.inv(Z11)).reshape(-1)
 
    w_lk  = (a   - sigma * phi_k)     / eta
    w_lth = (1.0 - sigma * phi_theta) / eta
 
    M  = np.array([
        [a_k + a_l*w_lk  - a_c*phi_k,
         a_theta + a_l*w_lth - a_c*phi_theta],
        [0.0, rho]
    ])
    wc = np.array([phi_k,   phi_theta])
    wl = np.array([w_lk,    w_lth   ])
    wy = np.array([a,       1.0     ]) + (1.0 - a) * wl
    wi = (lam / d) * wy - ((lam - d) / d) * wc
 
    return M, wc, wl, wy, wi


# =========================================================
# Benchmark: trabajo indivisible (tipo Hansen)
# =========================================================
def build_state_space_indivisible(bench):
    a, b, d, rho = (
        bench["alpha"],
        bench["beta"],
        bench["delta"],
        bench["rho"]
    )

    # En Hansen tipo log(c) - gamma h:
    # - fijamos sigma = 1
    # - desaparece la curvatura del ocio del caso divisible
    sigma = 1.0
    eta   = a

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
        raise ValueError("Blanchard-Kahn no satisfecho en el caso indivisible.")

    Z11 = Z[:2, :2]
    Z21 = Z[2:, :2]
    phi_k, phi_theta = (Z21 @ np.linalg.inv(Z11)).reshape(-1)

    w_lk  = (a   - sigma * phi_k)     / eta
    w_lth = (1.0 - sigma * phi_theta) / eta

    M  = np.array([
        [a_k + a_l*w_lk  - a_c*phi_k,
         a_theta + a_l*w_lth - a_c*phi_theta],
        [0.0, rho]
    ])
    wc = np.array([phi_k,   phi_theta])
    wl = np.array([w_lk,    w_lth   ])
    wy = np.array([a,       1.0     ]) + (1.0 - a) * wl
    wi = (lam / d) * wy - ((lam - d) / d) * wc

    return M, wc, wl, wy, wi
 
 
# =========================================================
# Simulación
# =========================================================
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
 
 
def build_corrs(sim, max_lag):
    lags = list(range(-max_lag, max_lag + 1))
    lny  = sim["y"]
    cc   = {v: [xcorr(lny, sim[v], lag) for lag in lags] for v in ["c", "k", "i", "l"]}
    return lags, cc


def build_stats(sim, cc, lags):
    sy = float(np.std(sim["y"], ddof=1))
    lag0 = lags.index(0)

    vol_df = pd.DataFrame({
        "Variable": ["σ(y)", "σ(c)/σ(y)", "σ(k)/σ(y)", "σ(i)/σ(y)", "σ(l)/σ(y)"],
        "Valor": [
            f"{sy*100:.2f}%",
            f"{np.std(sim['c'], ddof=1)/sy:.3f}",
            f"{np.std(sim['k'], ddof=1)/sy:.3f}",
            f"{np.std(sim['i'], ddof=1)/sy:.3f}",
            f"{np.std(sim['l'], ddof=1)/sy:.3f}",
        ],
    }).set_index("Variable")

    corr_df = pd.DataFrame({
        "Variable": ["corr(y, c)", "corr(y, k)", "corr(y, i)", "corr(y, l)"],
        "Valor": [
            f"{cc['c'][lag0]:+.4f}",
            f"{cc['k'][lag0]:+.4f}",
            f"{cc['i'][lag0]:+.4f}",
            f"{cc['l'][lag0]:+.4f}",
        ],
    }).set_index("Variable")

    return vol_df, corr_df


COLORS = dict(c="#60a5fa", k="#fb923c", i="#4ade80", l="#f87171")
NAMES  = dict(c="ln(c/c_ss)", k="ln(k/k_ss)", i="ln(i/i_ss)", l="ln(l/l_ss)")


def draw_model_block(title, sim, cc, lags, note=""):
    st.markdown(f"## {title}")
    if note:
        st.markdown(note)

    for tab, keys in zip(
        st.tabs(["Todas", "Consumo", "Capital", "Inversión", "Trabajo"]),
        [["c","k","i","l"], ["c"], ["k"], ["i"], ["l"]],
    ):
        with tab:
            fig = go.Figure()
            fig.add_hline(y=0, line_color="rgba(150,150,150,0.3)", line_width=1)
            fig.add_vline(x=0, line_color="rgba(150,150,150,0.3)", line_width=1, line_dash="dot")

            for v in keys:
                fig.add_trace(go.Scatter(
                    x=lags, y=cc[v], name=NAMES[v],
                    mode="lines+markers",
                    line=dict(color=COLORS[v], width=2.5),
                    marker=dict(size=7, color=COLORS[v]),
                    hovertemplate="lag %{x}: %{y:.4f}<extra>" + NAMES[v] + "</extra>",
                ))

            fig.update_layout(
                xaxis=dict(
                    title="Lag k  ·  corr(y_t , x_{t+k})",
                    tickmode="array", tickvals=lags,
                    gridcolor="rgba(200,200,200,0.08)"
                ),
                yaxis=dict(
                    title="Correlación", range=[-0.25, 1.05],
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
        st.markdown(r"**Volatilidades relativas** $\sigma(x)/\sigma(y)$")
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
            # Divisible
            M_div, wc_div, wl_div, wy_div, wi_div = build_state_space_divisible(sigma, psi, BENCH)
            sim_div = simulate(M_div, wc_div, wl_div, wy_div, wi_div, BENCH)
            lags_div, cc_div = build_corrs(sim_div, max_lag)

            # Indivisible
            M_ind, wc_ind, wl_ind, wy_ind, wi_ind = build_state_space_indivisible(BENCH)
            sim_ind = simulate(M_ind, wc_ind, wl_ind, wy_ind, wi_ind, BENCH)
            lags_ind, cc_ind = build_corrs(sim_ind, max_lag)

            st.session_state.update(
                sim_div=sim_div,
                lags_div=lags_div,
                cc_div=cc_div,
                sim_ind=sim_ind,
                lags_ind=lags_ind,
                cc_ind=cc_ind,
                sigma=sigma,
                psi=psi
            )
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()
 
# Recuperar
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
        note=(
            f"Parámetros libres actuales: "
            f"$\\sigma={sig_:.1f}$, $\\psi={psi_:.1f}$"
        )
    )

with main_tab2:
    draw_model_block(
        title="Trabajo indivisible",
        sim=sim_ind,
        cc=cc_ind,
        lags=lags_ind,
        note=(
            "$\\sigma=1$ y sin parámetro de curvatura del ocio."
        )
    )

# =========================================================
# Footer
# =========================================================
st.divider()
st.markdown(
    f"*Trabajo divisible:* depende de *σ={sig_:.1f}* y *ψ={psi_:.1f}*. "
    "*Trabajo indivisible:* benchmark fijo frente a esos parámetros libres. "
    "La mayor diferencia suele aparecer en trabajo, aunque también puede trasladarse a consumo e inversión."
)
