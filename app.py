import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

# -----------------------------------------------------------------------------
# 1. PAGE & STYLE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="STIR Master Pro",
    layout="wide",
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. FINANCIAL LOGIC
# -----------------------------------------------------------------------------
MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

DV01_MAP = {"ZQ": 41.67, "SR3": 25.00}

@st.cache_data
def get_market_data(base_rate, shock_bps):
    today = date.today()
    data = []
    
    # Curve Construction:
    # 1. Slope: -3bps per month (Inverted)
    # 2. Convexity: Small quadratic term so the curve isn't a perfect straight line.
    #    This ensures Fly prices are not exactly 0.000.
    slope = -0.03
    convexity = 0.0005 
    
    for i in range(24):
        f_date = today + timedelta(days=30*i)
        if f_date.year > 2028: break
        
        # Rate = Base + Linear Slope + Quadratic Convexity + User Shock
        raw_rate = base_rate + (i * slope) + ((i**2) * convexity) + (shock_bps / 100.0)
        
        # Prices
        zq_price = 100 - raw_rate
        sr3_price = 100 - (raw_rate - 0.05)
        
        # Tickers
        suffix = f"{MONTH_CODES[f_date.month]}{str(f_date.year)[-1]}"
        
        data.append({
            "Month": f_date.strftime("%b %y"),
            "Suffix": suffix,
            "ZQ_Ticker": f"ZQ{suffix}",
            "SR3_Ticker": f"SR3{suffix}",
            "ZQ_Price": zq_price,
            "SR3_Price": sr3_price,
            "ZQ_Rate": raw_rate,
            "SR3_Rate": raw_rate - 0.05
        })
        
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.header("🏛️ STIR Desk")
view_mode = st.sidebar.radio("Workstation Mode", ["Market Overview", "Strategy Lab", "Spread Matrix"])

st.sidebar.divider()
st.sidebar.subheader("Curve Assumptions")
base_effr = st.sidebar.number_input("Base Rate (%)", 0.00, 20.00, 3.64, 0.01)
curve_shock = st.sidebar.slider("Parallel Shift (bps)", -50, 50, 0)
tape_src = st.sidebar.selectbox("Tape Instrument", ["ZQ", "SR3"])

df = get_market_data(base_effr, curve_shock)

# -----------------------------------------------------------------------------
# 4. TAPE
# -----------------------------------------------------------------------------
st.title("STIR Master Pro")
st.caption(f"Pricing Date: {date.today().strftime('%Y-%m-%d')} | Mode: Synthetic Live")

cols = st.columns(6)
for i in range(min(6, len(df))):
    row = df.iloc[i]
    ticker = row[f"{tape_src}_Ticker"]
    price = row[f"{tape_src}_Price"]
    rate = row[f"{tape_src}_Rate"]
    chg = rate - base_effr
    
    with cols[i]:
        st.metric(label=ticker, value=f"{price:.3f}", delta=f"{chg:.2f}%", delta_color="inverse")
st.divider()

# -----------------------------------------------------------------------------
# MODE 1: MARKET OVERVIEW
# -----------------------------------------------------------------------------
if view_mode == "Market Overview":
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Forward Term Structure")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Month'], y=df['ZQ_Rate'], name='ZQ (Fed Funds)', line=dict(color='#00F0FF', width=3)))
        fig.add_trace(go.Scatter(x=df['Month'], y=df['SR3_Rate'], name='SR3 (SOFR)', line=dict(color='#FFA500', width=2, dash='dash')))
        fig.update_layout(template="plotly_dark", height=450, xaxis_title="Contract", yaxis_title="Implied Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Settlement Board")
        st.dataframe(df[['Month', 'ZQ_Price', 'SR3_Price']], hide_index=True, use_container_width=True, height=450)

# -----------------------------------------------------------------------------
# MODE 2: STRATEGY LAB (UPDATED RISK ENGINE)
# -----------------------------------------------------------------------------
elif view_mode == "Strategy Lab":
    st.subheader("🛠️ Strategy Constructor")
    
    c1, c2, c3 = st.columns(3)
    strat_type = c1.selectbox("Structure", ["Calendar Spread", "Butterfly (Fly)", "Condor"])
    prod = c2.selectbox("Product", ["ZQ", "SR3"])
    lots = c3.number_input("Size (Lots)", 100, 10000, 100, 100)

    legs = []
    tickers = df[f"{prod}_Ticker"].tolist()
    prices = dict(zip(tickers, df[f"{prod}_Price"]))
    
    # Strategy Logic
    if strat_type == "Calendar Spread":
        l1, l2 = st.columns(2)[0].selectbox("Front", tickers, 0), st.columns(2)[1].selectbox("Back", tickers, 1)
        legs = [{"Side": "BUY", "Qty": 1, "Ticker": l1, "Price": prices[l1]}, {"Side": "SELL", "Qty": -1, "Ticker": l2, "Price": prices[l2]}]
        
    elif strat_type == "Butterfly (Fly)":
        cols = st.columns(3)
        w1, b, w2 = cols[0].selectbox("Wing 1", tickers, 0), cols[1].selectbox("Belly", tickers, 3), cols[2].selectbox("Wing 2", tickers, 6)
        legs = [
            {"Side": "BUY", "Qty": 1, "Ticker": w1, "Price": prices[w1]},
            {"Side": "SELL", "Qty": -2, "Ticker": b, "Price": prices[b]},
            {"Side": "BUY", "Qty": 1, "Ticker": w2, "Price": prices[w2]}
        ]

    elif strat_type == "Condor":
        cols = st.columns(4)
        legs_sel = [cols[i].selectbox(f"Leg {i+1}", tickers, i) for i in range(4)]
        qtys = [1, -1, -1, 1]
        legs = [{"Side": "BUY" if q>0 else "SELL", "Qty": q, "Ticker": t, "Price": prices[t]} for q,t in zip(qtys, legs_sel)]

    # Ticket & Risk
    st.divider()
    c_tick, c_risk = st.columns([1, 2])
    
    with c_tick:
        st.markdown("#### 🎫 Ticket")
        pkg_price = 0
        tick_data = []
        for leg in legs:
            weight = leg['Qty'] / legs[0]['Qty']
            pkg_price += (leg['Price'] * weight)
            tick_data.append({"Side": "BUY" if leg['Qty']>0 else "SELL", "Qty": abs(leg['Qty'])*lots, "Ticker": leg['Ticker'], "Price": leg['Price']})
            
        st.dataframe(pd.DataFrame(tick_data), hide_index=True, use_container_width=True)
        st.metric("Package Price", f"{pkg_price:.3f}")
        st.markdown("---")
        st.markdown("#### 📊 Risk Sensitivity")
        
        # Calculate Net DV01 (Sensitivity to 1bp parallel move)
        net_dv01 = 0
        current_dv01_per_lot = DV01_MAP[prod]
        
        for leg in legs:
            # Leg PnL = Qty * Lots * (-1bp move) * DV01_Value
            leg_risk = (leg['Qty'] * lots) * -1 * current_dv01_per_lot
            net_dv01 += leg_risk
            
        c_r1, c_r2 = st.columns(2)
        c_r1.metric("Net DV01 ($)", f"${net_dv01:,.0f}")
            
        if abs(net_dv01) < (lots * 5): 
            risk_type = "Neutral"
            r_color = "off"
        elif net_dv01 > 0:
            risk_type = "Bullish"
            r_color = "normal"
        else:
            risk_type = "Bearish"
            r_color = "inverse"
        
        c_r2.metric("Bias", risk_type, delta=f"{net_dv01:.0f}", delta_color=r_color)
    with c_risk:
        st.markdown("#### ⚠️ Risk Simulation")
        sim_mode = st.radio("Mode", ["Parallel Shift", "Curve Twist (Steepener)"], horizontal=True)
        
        moves = np.arange(-25, 26, 1)
        pnl_vals = []
        dv01 = DV01_MAP[prod]
        # 3. Calculate "Center of Gravity" for the trade
        # This fixes the skew by pivoting around the middle of the structure
        center_index = (len(legs) - 1) / 2 

        for m in moves:
            run_pnl = 0
            for i, leg in enumerate(legs):
                # Get the DV01 for this specific contract
                leg_dv01 = DV01_MAP[prod] 
                
                if sim_mode == "Parallel Shift":
                    # Every leg moves the same amount 'm'
                    shift = m
                else:
                    # Curve Twist (Centered Pivot)
                    # Front legs move DOWN, Back legs move UP (if m > 0)
                    dist_from_center = i - center_index
                    shift = m * dist_from_center
                
                # PnL Calculation:
                # If Rates go UP (Shift > 0), Price goes DOWN. 
                # We multiply by -1 to capture this inverse relationship.
                leg_pnl = -1 * shift * leg_dv01 * (leg['Qty'] * lots)
                run_pnl += leg_pnl
            
            pnl_vals.append(round(run_pnl, 2))
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=moves, y=pnl_vals, fill='tozeroy', line=dict(color='#4CAF50' if pnl_vals[-1]>=0 else '#F44336')))
        fig.update_layout(title=f"PnL vs {sim_mode}", xaxis_title="Move (bps)", yaxis_title="PnL ($)", template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# MODE 3: SPREAD MATRIX
# -----------------------------------------------------------------------------
elif view_mode == "Spread Matrix":
    st.subheader("📅 Spread Matrix")
    prod = st.selectbox("Curve", ["ZQ", "SR3"])
    tickers = df[f"{prod}_Ticker"].tolist()
    prices = df[f"{prod}_Price"].tolist()
    
    data = []
    for i in range(len(tickers)-1):
        data.append({
            "Pair": f"{tickers[i]}/{tickers[i+1]}",
            "Spread": prices[i] - prices[i+1],
            "Front": prices[i],
            "Back": prices[i+1]
        })
    
    m_df = pd.DataFrame(data)
    st.bar_chart(m_df.set_index("Pair")['Spread'])
    st.dataframe(m_df, hide_index=True, use_container_width=True)
