import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.linalg import ordqz
from scipy.optimize import brentq

st.set_page_config(page_title="RBC — Correlaciones cruzadas", layout="centered")

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("Parámetros")
alpha   = st.sidebar.slider("α — participación capital",  0.20, 0.50, 0.36, 0.01)
beta    = st.sidebar.slider("β — descuento",              0.90, 0.999, 0.99, 0.001, format="%.3f")
delta   = st.sidebar.slider("δ — depreciación",           0.010, 0.150, 0.025, 0.005)
sigma   = st.sidebar.slider("σ — aversión al riesgo",     0.5, 6.0, 2.0, 0.1)
gamma   = st.sidebar.slider("γ — peso del ocio",          0.1, 5.0, 1.0, 0.1)
psi     = st.sidebar.slider("ψ — curvatura ocio",         0.5, 5.0, 1.0, 0.1)
rho     = st.sidebar.slider("ρ — persistencia TFP",       0.50, 0.99, 0.90, 0.01)
sig_eps = st.sidebar.slider("σ_ε — innovación (%)",       0.5, 5.0, 1.0, 0.1) / 100
T_sim   = st.sidebar.select_slider("Periodos T", [10_000, 20_000, 50_000, 100_000], 50_000)
max_lag = st.sidebar.slider("Lags máximos", 2, 8, 5)
run_btn = st.sidebar.button("▶ Simular", type="primary", use_container_width=True)

# ── Model ─────────────────────────────────────────────────────
def solve(alpha, beta, delta, sigma, gamma, psi, rho, sig_eps, T, seed=42):
    # Steady-state labour from FOC: w·c^{-σ} = γ(1-l)^{-ψ}
    k_l     = (alpha / (1/beta - 1 + delta)) ** (1/(1-alpha))
    c_per_l = k_l**alpha - delta*k_l
    w       = (1-alpha) * k_l**alpha
    l_ss    = brentq(lambda l: w*(c_per_l*l)**(-sigma) - gamma*(1-l)**(-psi),
                     1e-4, 1-1e-4)

    psi_eff     = psi * l_ss / (1 - l_ss)
    eta         = alpha + psi_eff
    phi_        = (1 - beta*(1-delta)) / sigma
    s           = (1-alpha) / eta
    lam         = (1/alpha) * (1/beta - 1 + delta)
    a_theta     = lam;  a_k = 1/beta
    a_l         = ((1-alpha)/alpha) * (1/beta - 1 + delta);  a_c = lam - delta
    kappa_c     = 1 + phi_*sigma*s
    kappa_k     = phi_*((alpha-1) + alpha*s)
    kappa_theta = phi_*(1 + s)
    b11 = a_k + (alpha*a_l)/eta
    b12 = -a_c - (sigma*a_l)/eta
    c1  = a_theta + a_l/eta

    A = np.array([[1., 0.], [kappa_k, -kappa_c]])
    B = np.array([[b11, b12], [0., -1.]])
    C = np.array([[c1], [-kappa_theta*rho]])

    Ar = np.zeros((3,3)); Br = np.zeros((3,3))
    Ar[0,1]=A[0,0]; Ar[0,0]=A[0,1]; Br[0,1]=B[0,0]; Br[0,0]=B[0,1]; Br[0,2]=C[0,0]
    Ar[1,1]=A[1,0]; Ar[1,0]=A[1,1]; Br[1,1]=B[1,0]; Br[1,0]=B[1,1]; Br[1,2]=C[1,0]
    Ar[2,2]=1.;     Br[2,2]=rho
    p  = [1,2,0]
    S,T_,al,bt,_,Z = ordqz(Br[np.ix_(p,p)], Ar[np.ix_(p,p)], sort="iuc")
    if (np.abs(al/bt) < 1-1e-10).sum() != 2:
        raise ValueError("Blanchard-Kahn: revisa los parámetros.")

    Z11=Z[:2,:2]; Z21=Z[2:,:2]
    phi_k, phi_theta = (Z21 @ np.linalg.inv(Z11)).reshape(-1)

    w_lk  = (alpha - sigma*phi_k)   / eta
    w_lth = (1.    - sigma*phi_theta)/ eta
    M  = np.array([[a_k+a_l*w_lk-a_c*phi_k, a_theta+a_l*w_lth-a_c*phi_theta],
                   [0., rho]])
    wc = np.array([phi_k,   phi_theta])
    wl = np.array([w_lk,    w_lth])
    wy = np.array([alpha,   1.]) + (1-alpha)*wl
    wi = (lam/delta)*wy - ((lam-delta)/delta)*wc

    # Simulate
    rng   = np.random.default_rng(seed)
    burn  = 2000
    eps   = rng.normal(0, sig_eps, T+burn)
    st_   = np.zeros((2, T+burn+1))
    for t in range(T+burn):
        st_[:,t+1] = M @ st_[:,t] + np.array([0., eps[t]])
    st_ = st_[:, burn+1:]
    k, th = st_[0], st_[1]

    return dict(y=wy[0]*k+wy[1]*th, c=wc[0]*k+wc[1]*th,
                k=k, i=wi[0]*k+wi[1]*th, l=wl[0]*k+wl[1]*th,
                l_ss=l_ss)

def xcorr(y, x, lag):
    n = len(y)
    a, b = (y[:n-lag], x[lag:]) if lag >= 0 else (y[-lag:], x[:n+lag])
    return float(np.corrcoef(a, b)[0,1])

# ── Run ───────────────────────────────────────────────────────
if run_btn or "sim" not in st.session_state:
    with st.spinner("Simulando…"):
        try:
            st.session_state["sim"]    = solve(alpha, beta, delta, sigma, gamma,
                                               psi, rho, sig_eps, T_sim)
            st.session_state["params"] = dict(alpha=alpha, beta=beta, delta=delta,
                                              sigma=sigma, psi=psi, rho=rho,
                                              sig_eps=sig_eps, T=T_sim, max_lag=max_lag)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

sim    = st.session_state["sim"]
params = st.session_state["params"]
lny    = sim["y"]

lags   = list(range(-max_lag, max_lag+1))
COLORS = dict(c="#60a5fa", k="#fb923c", i="#4ade80", l="#f87171")
LABELS = dict(c="ln(c/c<sub>ss</sub>)", k="ln(k/k<sub>ss</sub>)",
              i="ln(i/i<sub>ss</sub>)", l="ln(l/l<sub>ss</sub>)")

cc = {v: [xcorr(lny, sim[v], lag) for lag in lags] for v in ["c","k","i","l"]}
sy = float(np.std(lny, ddof=1))

# ── Layout ────────────────────────────────────────────────────
p = params
st.markdown(
    f"## Correlación cruzada con ln(y/y<sub>ss</sub>) &nbsp;"
    f"<small style='color:#8b90a0'>σ={p['sigma']}, ψ={p['psi']}</small>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='font-family:monospace;font-size:12px;color:#8b90a0'>"
    f"α={p['alpha']} · β={p['beta']} · δ={p['delta']} · "
    f"ρ={p['rho']} · σ_ε={p['sig_eps']*100:.1f}% · "
    f"T={p['T']:,} · l_ss={sim['l_ss']:.2f}</p>",
    unsafe_allow_html=True
)

# ── Tab filter
tabs  = st.tabs(["Todas", "Consumo", "Capital", "Inversión", "Horas"])
keys_map = [["c","k","i","l"], ["c"], ["k"], ["i"], ["l"]]

for tab, keys in zip(tabs, keys_map):
    with tab:
        fig = go.Figure()
        fig.add_hline(y=0, line_color="rgba(150,150,150,0.3)", line_width=1)
        fig.add_vline(x=0, line_color="rgba(150,150,150,0.3)", line_width=1, line_dash="dot")
        for v in keys:
            fig.add_trace(go.Scatter(
                x=lags, y=cc[v], name=v,
                mode="lines+markers",
                line=dict(color=COLORS[v], width=2.5),
                marker=dict(size=7, color=COLORS[v]),
                hovertemplate="lag %{x}: %{y:.4f}<extra>" + v + "</extra>",
            ))
        fig.update_layout(
            xaxis=dict(title="Lag k  ·  corr(y_t , x_{t+k})",
                       tickmode="array", tickvals=lags,
                       gridcolor="rgba(200,200,200,0.1)"),
            yaxis=dict(title="Correlación", range=[-0.2, 1.05],
                       gridcolor="rgba(200,200,200,0.1)"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=40, b=50),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Stats columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Volatilidades relativas · σ(x)/σ(y)**")
    st.markdown(f"σ(y) &nbsp; **{sy*100:.2f}%**", unsafe_allow_html=True)
    for v in ["c","k","i","l"]:
        ratio = np.std(sim[v], ddof=1) / sy
        lbl   = {"c":"σ(c)/σ(y)","k":"σ(k)/σ(y)","i":"σ(i)/σ(y)","l":"σ(l)/σ(y)"}[v]
        st.markdown(f"{lbl} &nbsp; **{ratio:.3f}**", unsafe_allow_html=True)

with col2:
    lag0 = lags.index(0)
    st.markdown("**Correlación contemporánea · lag 0**")
    for v in ["c","k","i","l"]:
        val  = cc[v][lag0]
        lbl  = {"c":"corr(y, c)","k":"corr(y, k)","i":"corr(y, i)","l":"corr(y, l)"}[v]
        sign = "+" if val >= 0 else ""
        st.markdown(f"{lbl} &nbsp; **{sign}{val:.4f}**", unsafe_allow_html=True)

# ── Footnote
st.divider()
st.caption(
    f"El consumidor más averso al riesgo (σ={p['sigma']}) suaviza más el consumo. "
    f"La curvatura del ocio (ψ={p['psi']}) amortigua la respuesta del trabajo. "
    "La inversión es la variable más procíclica en lag 0. "
    "El capital tiene su pico en lags positivos: variable predeterminada. "
    "Convención: lag k > 0 → x adelanta a y k periodos."
)
