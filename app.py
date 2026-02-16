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
    .stAlert { border-left: 5px solid #00c805 !important; }
    /* Compact the leg selectors */
    div[data-testid="stVerticalBlock"] > div { padding-bottom: 0.5rem; }
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
    
    slope = -0.03
    convexity = 0.0005 
    
    for i in range(24):
        f_date = today + timedelta(days=30*i)
        if f_date.year > 2028: break
        
        raw_rate = base_rate + (i * slope) + ((i**2) * convexity) + (shock_bps / 100.0)
        
        zq_price = 100 - raw_rate
        sr3_price = 100 - (raw_rate - 0.05)
        
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
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_tickers(product):
    """Returns list of tickers for the selected product."""
    return df[f"{product}_Ticker"].tolist()

def get_price(ticker, product):
    """Looks up price for specific ticker and product."""
    return df.loc[df[f"{product}_Ticker"] == ticker, f"{product}_Price"].values[0]

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
# MODE 2: STRATEGY LAB (MULTI-LEG MIXED PRODUCTS)
# -----------------------------------------------------------------------------
elif view_mode == "Strategy Lab":
    st.subheader("🛠️ Strategy Constructor")
    
    # Structure Selection
    c_struct, c_lots = st.columns([2, 1])
    strat_type = c_struct.selectbox("Structure", ["Calendar Spread", "Butterfly (Fly)", "Condor"])
    lots = c_lots.number_input("Size (Lots)", 1, 10000, 100, 1)

    legs = []
    
    # --- LEG BUILDERS ---
    # We now create inputs for each leg dynamically
    
    if strat_type == "Calendar Spread":
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Leg 1 (Buy)**")
            p1 = st.selectbox("Product", ["ZQ", "SR3"], key="s_p1")
            t1 = st.selectbox("Contract", get_tickers(p1), index=0, key="s_t1")
            legs.append({"Side": "BUY", "Qty": 1, "Ticker": t1, "Type": p1, "Price": get_price(t1, p1)})
            
        with c2:
            st.markdown("**Leg 2 (Sell)**")
            p2 = st.selectbox("Product", ["ZQ", "SR3"], key="s_p2")
            t2 = st.selectbox("Contract", get_tickers(p2), index=1, key="s_t2")
            legs.append({"Side": "SELL", "Qty": -1, "Ticker": t2, "Type": p2, "Price": get_price(t2, p2)})

    elif strat_type == "Butterfly (Fly)":
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("**Wing 1 (Buy)**")
            p1 = st.selectbox("Product", ["ZQ", "SR3"], key="f_p1")
            t1 = st.selectbox("Contract", get_tickers(p1), index=0, key="f_t1")
            legs.append({"Side": "BUY", "Qty": 1, "Ticker": t1, "Type": p1, "Price": get_price(t1, p1)})
            
        with c2:
            st.markdown("**Belly (Sell 2)**")
            p2 = st.selectbox("Product", ["ZQ", "SR3"], key="f_p2")
            t2 = st.selectbox("Contract", get_tickers(p2), index=3, key="f_t2")
            legs.append({"Side": "SELL", "Qty": -2, "Ticker": t2, "Type": p2, "Price": get_price(t2, p2)})

        with c3:
            st.markdown("**Wing 2 (Buy)**")
            p3 = st.selectbox("Product", ["ZQ", "SR3"], key="f_p3")
            t3 = st.selectbox("Contract", get_tickers(p3), index=6, key="f_t3")
            legs.append({"Side": "BUY", "Qty": 1, "Ticker": t3, "Type": p3, "Price": get_price(t3, p3)})

    elif strat_type == "Condor":
        cols = st.columns(4)
        qtys = [1, -1, -1, 1]
        
        for i in range(4):
            with cols[i]:
                side = "Buy" if qtys[i] > 0 else "Sell"
                st.markdown(f"**Leg {i+1} ({side})**")
                # Default selection logic to space them out
                def_idx = i * 2 
                p = st.selectbox(f"Prod", ["ZQ", "SR3"], key=f"c_p{i}")
                t = st.selectbox(f"Cont", get_tickers(p), index=min(def_idx, 23), key=f"c_t{i}")
                
                legs.append({
                    "Side": "BUY" if qtys[i]>0 else "SELL", 
                    "Qty": qtys[i], 
                    "Ticker": t, 
                    "Type": p, 
                    "Price": get_price(t, p)
                })

    # --- TICKET & RISK ---
    st.divider()
    c_tick, c_risk = st.columns([1, 2])
    
    with c_tick:
        st.markdown("#### 🎫 Ticket")
        pkg_price = 0
        tick_data = []
        
        for leg in legs:
            eff_qty = int(leg['Qty'] * lots)
            # Weighted price isn't perfect for mixed products, but useful approx
            pkg_price += (leg['Price'] * leg['Qty']) 
            
            tick_data.append({
                "Side": leg['Side'],
                "Qty": abs(eff_qty),
                "Product": leg['Type'],
                "Ticker": leg['Ticker'],
                "Price": f"{leg['Price']:.3f}"
            })
            
        st.dataframe(pd.DataFrame(tick_data), hide_index=True, use_container_width=True)

        # Risk Metrics
        st.markdown("---")
        st.markdown("#### 📊 Sensitivity")
        
        net_dv01 = 0
        for leg in legs:
            # AUTO-DETECTS DV01 based on the leg's product choice
            leg_val = DV01_MAP[leg['Type']]
            leg_risk = (leg['Qty'] * lots) * -1 * leg_val
            net_dv01 += leg_risk
            
        c_r1, c_r2 = st.columns(2)
        c_r1.metric("Net DV01 ($)", f"${net_dv01:,.0f}")
        
        # Bias Logic
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
        sim_mode = st.radio("Mode", ["Parallel Shift", "Curve Twist (Centered)"], horizontal=True)
        
        moves = np.arange(-25, 26, 1)
        pnl_vals = []
        center_index = (len(legs) - 1) / 2
        
        for m in moves:
            run_pnl = 0
            for i, leg in enumerate(legs):
                # Critical: Uses leg-specific DV01 for mixed strategies
                leg_dv01 = DV01_MAP[leg['Type']]
                
                if sim_mode == "Parallel Shift":
                    shift = m
                else:
                    dist_from_center = i - center_index
                    shift = m * dist_from_center
                
                leg_pnl = -1 * shift * leg_dv01 * (leg['Qty'] * lots)
                run_pnl += leg_pnl
            
            pnl_vals.append(round(run_pnl, 2))
            
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=moves, y=pnl_vals, 
            fill='tozeroy', name='PnL',
            line=dict(color='#4CAF50' if pnl_vals[-1] >= 0 else '#F44336', width=3)
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig.update_layout(
            title=f"PnL Profile: {sim_mode}",
            xaxis_title="Curve Move (bps)",
            yaxis_title="Profit / Loss ($)",
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if sim_mode == "Curve Twist (Centered)" and abs(pnl_vals[-1]) <= 1.0:
             st.success("✅ Strategy is Perfectly Hedged against Curve Rotation.")

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
