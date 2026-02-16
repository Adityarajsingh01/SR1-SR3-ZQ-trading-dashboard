import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
import calendar

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="STIR Master Pro", 
    layout="wide", 
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMetric { background-color: #121212; border: 1px solid #333; padding: 10px; border-radius: 4px; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; color: #E0E0E0; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & UTILS
# -----------------------------------------------------------------------------
# 2026 FOMC Schedule
FOMC_MEETINGS_2026 = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6), 
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 11, 4), date(2026, 12, 16)
]

MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

# DV01 Constants ($ per bp per contract)
DV01 = {
    "ZQ": 41.67,
    "SR1": 41.67,
    "SR3": 25.00
}

def get_days_in_month(dt):
    return calendar.monthrange(dt.year, dt.month)[1]

def get_fomc_weights(contract_date):
    meeting = next((m for m in FOMC_MEETINGS_2026 if m.year == contract_date.year and m.month == contract_date.month), None)
    if not meeting: return (1.0, 0.0)
    dim = get_days_in_month(contract_date)
    return (meeting.day / dim, (dim - meeting.day) / dim)

# -----------------------------------------------------------------------------
# 3. DATA ENGINE (Robust)
# -----------------------------------------------------------------------------
@st.cache_data
def get_curve_data(base_rate, shock_bps):
    today = date.today()
    data = []
    slope = -0.03
    
    for i in range(18):
        f_date = today + timedelta(days=30*i)
        if f_date < today: continue
        
        mc = MONTH_CODES[f_date.month]
        yc = str(f_date.year)[-1]
        suffix = f"{mc}{yc}"
        
        # 1. Rates Logic
        raw_rate = base_rate + (i * slope) 
        final_effr = raw_rate + (shock_bps/100) # Apply Shock
        sofr_rate = final_effr - 0.05
        
        # 2. Pricing
        zq_price = 100 - final_effr
        sr1_price = 100 - sofr_rate
        sr3_price = 100 - (sofr_rate + 0.02)

        data.append({
            "Month": f_date.strftime("%b %y"),
            "Suffix": suffix,
            "Date": f_date,
            # Prices
            "ZQ": round(zq_price, 3),
            "SR1": round(sr1_price, 3),
            "SR3": round(sr3_price, 3),
            # Rates
            "ZQ_Rate": round(100-zq_price, 3),
            "SR1_Rate": round(100-sr1_price, 3),
            "SR3_Rate": round(100-sr3_price, 3)
        })
        
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 4. SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.header("STIR Desk Controls")
mode = st.sidebar.radio("Mode", ["Market Overview", "Custom Strategy Builder", "Spread Matrix"])

st.sidebar.divider()
st.sidebar.subheader("Global Curve Inputs")
# FIX: Min value 0.0 to allow 3.64
base_effr = st.sidebar.number_input("Base EFFR (%)", 0.0, 20.0, 3.64, 0.01)

# FIX: Tooltip explanation
curve_shock = st.sidebar.slider("Parallel Shift (bps)", -50, 50, 0, 
    help="Stress Test: Shifts the entire Yield Curve up/down to simulate a market shock.")

tape_source = st.sidebar.selectbox("Tape Source", ["ZQ", "SR1", "SR3"])

# Load Data
df = get_curve_data(base_effr, curve_shock)

# -----------------------------------------------------------------------------
# 5. HEADER & TAPE
# -----------------------------------------------------------------------------
st.title("STIR Trading Dashboard")
st.caption(f"Pricing: Synthetic Live | Effr: {base_effr}% | Shift: {curve_shock}bps")

# Dynamic Tape
cols = st.columns(6)
for i in range(min(6, len(df))):
    row = df.iloc[i]
    val = row[tape_source]
    rate = row[f"{tape_source}_Rate"]
    with cols[i]:
        st.metric(f"{tape_source}{row['Suffix']}", f"{val:.3f}", f"{rate:.2f}%")

st.divider()

# -----------------------------------------------------------------------------
# MODULE: MARKET OVERVIEW
# -----------------------------------------------------------------------------
if mode == "Market Overview":
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Forward Rates Structure")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Month'], y=df['ZQ_Rate'], name='ZQ (Fed Funds)', line=dict(color='#00F0FF', width=3)))
        fig.add_trace(go.Scatter(x=df['Month'], y=df['SR1_Rate'], name='SR1 (1M SOFR)', line=dict(color='#FFE800', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Month'], y=df['SR3_Rate'], name='SR3 (3M SOFR)', line=dict(color='#FF8C00', width=2, dash='dash')))
        fig.update_layout(template="plotly_dark", height=450, xaxis_title="Contract", yaxis_title="Implied Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Live Quotes")
        st.dataframe(df[['Month', 'ZQ', 'SR1', 'SR3']], hide_index=True, use_container_width=True, height=450)

# -----------------------------------------------------------------------------
# MODULE: CUSTOM STRATEGY BUILDER (Rebuilt)
# -----------------------------------------------------------------------------
elif mode == "Custom Strategy Builder":
    st.subheader("🛠️ Custom Spread Builder")
    
    # --- LEG DEFINITION ---
    c_leg1, c_mid, c_leg2 = st.columns([1, 0.2, 1])
    
    # Helper for dropdowns
    contract_opts = [f"{r['Suffix']} ({r['Month']})" for i, r in df.iterrows()]
    
    with c_leg1:
        st.markdown("#### Leg 1")
        l1_action = st.selectbox("Action", ["BUY", "SELL"], key="l1_a")
        l1_qty = st.number_input("Qty", 1, 5000, 100, key="l1_q")
        l1_prod = st.selectbox("Product", ["ZQ", "SR1", "SR3"], key="l1_p")
        l1_contract_str = st.selectbox("Contract", contract_opts, key="l1_c")
        
        # Parse Contract
        l1_suffix = l1_contract_str.split(" ")[0]
        l1_row = df[df['Suffix'] == l1_suffix].iloc[0]
        l1_price = l1_row[l1_prod]
        l1_sign = 1 if l1_action == "BUY" else -1
        
        st.metric(f"{l1_prod}{l1_suffix}", f"{l1_price:.3f}")

    with c_mid:
        st.markdown("<h2 style='text-align: center; padding-top: 80px;'>vs</h2>", unsafe_allow_html=True)

    with c_leg2:
        st.markdown("#### Leg 2")
        l2_action = st.selectbox("Action", ["BUY", "SELL"], index=1, key="l2_a") # Default to Sell
        l2_qty = st.number_input("Qty", 1, 5000, 100, key="l2_q")
        l2_prod = st.selectbox("Product", ["ZQ", "SR1", "SR3"], key="l2_p") # Allow Cross Product!
        l2_contract_str = st.selectbox("Contract", contract_opts, index=min(2, len(contract_opts)-1), key="l2_c")
        
        # Parse Contract
        l2_suffix = l2_contract_str.split(" ")[0]
        l2_row = df[df['Suffix'] == l2_suffix].iloc[0]
        l2_price = l2_row[l2_prod]
        l2_sign = 1 if l2_action == "BUY" else -1
        
        st.metric(f"{l2_prod}{l2_suffix}", f"{l2_price:.3f}")

    st.divider()

    # --- CALCULATIONS ---
    # Net Price (Spread Price)
    # Convention: (Leg1 * Qty) + (Leg2 * Qty). 
    # Usually spreads are quoted 1:1, but here we sum the total cash value for PnL.
    
    net_dv01 = (l1_sign * l1_qty * DV01[l1_prod]) + (l2_sign * l2_qty * DV01[l2_prod])
    
    # Entry Price Diff
    spread_price = (l1_price * l1_sign) + (l2_price * l2_sign) 
    # If standard calendar spread (Buy A, Sell B), Price = A - B
    
    c_res1, c_res2 = st.columns([1, 2])
    
    with c_res1:
        st.write("### Risk Profile")
        st.metric("Net Package Price", f"{spread_price:.3f}")
        
        risk_color = "red" if abs(net_dv01) > 1000 else "green"
        st.markdown(f"**Net DV01:** <span style='color:{risk_color}'>${net_dv01:,.2f}</span> / bp", unsafe_allow_html=True)
        
        if l1_prod != l2_prod:
            st.info("⚠️ Inter-commodity Spread Detected (e.g. ZQ vs SR3)")
        elif l1_suffix != l2_suffix:
            st.info("📅 Calendar Spread Detected")

    with c_res2:
        st.write("### PnL Simulation (Curve Shift)")
        
        # Generate PnL Chart for Parallel Shifts
        shifts = np.linspace(-50, 50, 21)
        pnl_vals = []
        
        for s in shifts:
            # PnL = -1 * Shift * DV01
            # If Net DV01 is positive (Long Risk), and Rates go UP (+Shift), Price goes DOWN -> Loss.
            # Formula: PnL = NetDV01 * -1 * (Shift/100) * 100 ?? 
            # Simplified: DV01 is dollar val per 1bp change. 
            # If Rates +10bps, Price -10 ticks. Long position loses.
            # So PnL = NetDV01 * (Shift * -1)
            pnl = net_dv01 * (-s) 
            pnl_vals.append(pnl)
            
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(x=shifts, y=pnl_vals, fill='tozeroy', line=dict(color='#00F0FF')))
        fig_pnl.update_layout(
            title="PnL vs Market Rate Move (Parallel Shift)",
            xaxis_title="Rate Shift (bps)",
            yaxis_title="PnL ($)",
            template="plotly_dark",
            height=350
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

# -----------------------------------------------------------------------------
# MODULE: SPREAD MATRIX
# -----------------------------------------------------------------------------
elif mode == "Spread Matrix":
    st.subheader("Calendar Spread Matrix")
    curve = st.selectbox("Select Curve", ["ZQ", "SR1", "SR3"])
    
    spreads = []
    for i in range(len(df)-1):
        front = df.iloc[i]
        back = df.iloc[i+1]
        val = front[curve] - back[curve]
        spreads.append({
            "Spread": f"{front['Suffix']}/{back['Suffix']}",
            "Price": val,
            "Type": "Inv" if val < 0 else "Steep"
        })
    
    s_df = pd.DataFrame(spreads)
    
    st.bar_chart(s_df.set_index("Spread")['Price'])
    st.dataframe(s_df.T, use_container_width=True)
