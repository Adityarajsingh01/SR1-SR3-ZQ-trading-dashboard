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

# Custom CSS for Professional Data Density
st.markdown("""
<style>
    /* Tighten up metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }
    /* Remove padding from top */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. FINANCIAL LOGIC & CONSTANTS
# -----------------------------------------------------------------------------
MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

# DV01: Dollar Value of a 1bp move per contract
DV01_MAP = {"ZQ": 41.67, "SR3": 25.00}

@st.cache_data
def get_market_data(base_rate, shock_bps):
    """Generates synthetic curve data based on user inputs."""
    today = date.today()
    data = []
    
    # Curve Shape: Inverted (-3bps/month) + User Shock
    slope = -0.03
    
    for i in range(24): # 2 Years forward
        f_date = today + timedelta(days=30*i)
        if f_date.year > 2028: break
        
        # 1. Calculate Rate
        # Base + Slope + Shock(converted to %)
        raw_rate = base_rate + (i * slope) + (shock_bps / 100.0)
        
        # 2. Derive Prices
        # ZQ = 100 - Rate
        # SR3 = 100 - (Rate - spread)
        zq_price = 100 - raw_rate
        sr3_price = 100 - (raw_rate - 0.05) # SR3 usually trades higher price (lower rate) in this sim
        
        # 3. Ticker Generation
        mc = MONTH_CODES[f_date.month]
        yc = str(f_date.year)[-1]
        suffix = f"{mc}{yc}"
        
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
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("🏛️ STIR Desk")

# Navigation
view_mode = st.sidebar.radio("Workstation Mode", ["Market Overview", "Strategy Lab", "Spread Matrix"])

st.sidebar.divider()
st.sidebar.subheader("Curve Assumptions")

# Inputs
base_effr = st.sidebar.number_input("Base Rate (%)", 0.00, 20.00, 3.64, 0.01)
curve_shock = st.sidebar.slider("Parallel Shift (bps)", -50, 50, 0)
tape_src = st.sidebar.selectbox("Tape Instrument", ["ZQ", "SR3"])

# Load Data
df = get_market_data(base_effr, curve_shock)

# -----------------------------------------------------------------------------
# 4. TOP TAPE (ALWAYS VISIBLE)
# -----------------------------------------------------------------------------
st.title("STIR Master Pro")
st.caption(f"Pricing Date: {date.today().strftime('%Y-%m-%d')} | Data Mode: Synthetic Real-Time")

# Display top 6 contracts
cols = st.columns(6)
for i in range(min(6, len(df))):
    row = df.iloc[i]
    ticker = row[f"{tape_src}_Ticker"]
    price = row[f"{tape_src}_Price"]
    rate = row[f"{tape_src}_Rate"]
    
    # Change calculation (vs Base Rate)
    chg = rate - base_effr
    
    with cols[i]:
        st.metric(
            label=ticker,
            value=f"{price:.3f}",
            delta=f"{chg:.2f}%",
            delta_color="inverse" # Rate Up = Red
        )
st.divider()

# -----------------------------------------------------------------------------
# MODE 1: MARKET OVERVIEW
# -----------------------------------------------------------------------------
if view_mode == "Market Overview":
    c_chart, c_board = st.columns([2, 1])
    
    with c_chart:
        st.subheader("Forward Term Structure")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Month'], y=df['ZQ_Rate'], name='ZQ (Fed Funds)', line=dict(color='#00F0FF', width=3)))
        fig.add_trace(go.Scatter(x=df['Month'], y=df['SR3_Rate'], name='SR3 (SOFR)', line=dict(color='#FFA500', width=2, dash='dash')))
        fig.update_layout(template="plotly_dark", height=450, xaxis_title="Contract Month", yaxis_title="Implied Rate (%)", margin=dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
    with c_board:
        st.subheader("Settlement Board")
        view_df = df[['Month', 'ZQ_Price', 'SR3_Price']].copy()
        st.dataframe(
            view_df,
            column_config={
                "ZQ_Price": st.column_config.NumberColumn("ZQ", format="%.3f"),
                "SR3_Price": st.column_config.NumberColumn("SR3", format="%.3f")
            },
            hide_index=True,
            use_container_width=True,
            height=450
        )

# -----------------------------------------------------------------------------
# MODE 2: STRATEGY LAB (PRO)
# -----------------------------------------------------------------------------
elif view_mode == "Strategy Lab":
    st.subheader("🛠️ Strategy Constructor")
    
    # 1. Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        strat_type = st.selectbox("Structure", ["Calendar Spread", "Butterfly (Fly)", "Condor"])
    with col2:
        prod = st.selectbox("Product", ["ZQ", "SR3"])
    with col3:
        lots = st.number_input("Size (Lots)", 100, 10000, 100, 100)

    # 2. Logic
    legs = []
    tickers = df[f"{prod}_Ticker"].tolist()
    prices = dict(zip(tickers, df[f"{prod}_Price"]))
    
    if strat_type == "Calendar Spread":
        c_leg1, c_leg2 = st.columns(2)
        l1 = c_leg1.selectbox("Front Leg", tickers, index=0)
        l2 = c_leg2.selectbox("Back Leg", tickers, index=1)
        legs = [
            {"Side": "BUY", "Qty": 1, "Ticker": l1, "Price": prices[l1]},
            {"Side": "SELL", "Qty": -1, "Ticker": l2, "Price": prices[l2]}
        ]
        
    elif strat_type == "Butterfly (Fly)":
        c_w1, c_b, c_w2 = st.columns(3)
        w1 = c_w1.selectbox("Wing 1", tickers, index=0)
        belly = c_b.selectbox("Belly", tickers, index=3)
        w2 = c_w2.selectbox("Wing 2", tickers, index=6)
        legs = [
            {"Side": "BUY", "Qty": 1, "Ticker": w1, "Price": prices[w1]},
            {"Side": "SELL", "Qty": -2, "Ticker": belly, "Price": prices[belly]},
            {"Side": "BUY", "Qty": 1, "Ticker": w2, "Price": prices[w2]}
        ]

    elif strat_type == "Condor":
        c1, c2, c3, c4 = st.columns(4)
        l1 = c1.selectbox("Leg 1", tickers, index=0)
        l2 = c2.selectbox("Leg 2", tickers, index=1)
        l3 = c3.selectbox("Leg 3", tickers, index=2)
        l4 = c4.selectbox("Leg 4", tickers, index=3)
        legs = [
            {"Side": "BUY", "Qty": 1, "Ticker": l1, "Price": prices[l1]},
            {"Side": "SELL", "Qty": -1, "Ticker": l2, "Price": prices[l2]},
            {"Side": "SELL", "Qty": -1, "Ticker": l3, "Price": prices[l3]},
            {"Side": "BUY", "Qty": 1, "Ticker": l4, "Price": prices[l4]}
        ]

    # 3. Calculation & Display
    st.divider()
    col_ticket, col_risk = st.columns([1, 2])
    
    with col_ticket:
        st.markdown("#### 🎫 Ticket")
        
        # Calculate Package Price
        pkg_price = 0
        ticket_rows = []
        for leg in legs:
            # Spread pricing convention: Sum(Price * Weight)
            # Normalize weights to first leg for display (usually 1)
            weight = leg['Qty'] / legs[0]['Qty']
            pkg_price += (leg['Price'] * weight)
            
            ticket_rows.append({
                "Side": "BUY" if leg['Qty'] > 0 else "SELL",
                "Qty": abs(leg['Qty']) * lots,
                "Ticker": leg['Ticker'],
                "Price": leg['Price']
            })
            
        st.dataframe(
            pd.DataFrame(ticket_rows), 
            hide_index=True, 
            use_container_width=True,
            column_config={"Price": st.column_config.NumberColumn(format="%.3f")}
        )
        
        st.metric("Net Package Price", f"{pkg_price:.3f}")
        
    with col_risk:
        st.markdown("#### ⚠️ Risk Simulation")
        
        sim_mode = st.radio("Stress Test", ["Parallel Shift", "Curve Twist (Steepener)"], horizontal=True)
        
        # Simulation Data
        moves = np.arange(-25, 26, 1) # -25 to +25 bps
        pnl_values = []
        
        dv01 = DV01_MAP[prod]
        
        for m in moves:
            run_pnl = 0
            for i, leg in enumerate(legs):
                # Define Shock
                if sim_mode == "Parallel Shift":
                    shift_bps = m
                else:
                    # Twist: Pivot at Leg 0. Subsequent legs move more.
                    # e.g. Leg 0 = 0bps, Leg 1 = 1*m, Leg 2 = 2*m
                    shift_bps = m * i 
                
                # PnL Calculation:
                # Rate UP (+Shift) -> Price DOWN. 
                # Long Position loses. PnL = Qty * -Shift * DV01
                # (Qty is signed: +1 for Buy, -1 for Sell)
                leg_pnl = (leg['Qty'] * lots) * (-shift_bps) * dv01
                run_pnl += leg_pnl
            
            pnl_values.append(run_pnl)
            
        # Chart
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(
            x=moves, 
            y=pnl_values, 
            fill='tozeroy', 
            line=dict(color='#4CAF50' if pnl_values[-1] >= 0 else '#F44336'),
            name="PnL"
        ))
        fig_risk.update_layout(
            title=f"PnL vs {sim_mode}",
            xaxis_title="Move (bps)",
            yaxis_title="Profit/Loss ($)",
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        if sim_mode == "Parallel Shift" and abs(pnl_values[-1]) < 50:
             st.info("ℹ️ Strategy is Delta Neutral (Flat PnL on parallel moves). Use 'Curve Twist' to see Slope risk.")

# -----------------------------------------------------------------------------
# MODE 3: SPREAD MATRIX
# -----------------------------------------------------------------------------
elif view_mode == "Spread Matrix":
    st.subheader("📅 Calendar Spread Matrix (1-Month Rolls)")
    
    m_prod = st.selectbox("Curve", ["ZQ", "SR3"])
    
    # Prepare Data
    tickers = df[f"{m_prod}_Ticker"].tolist()
    prices = df[f"{m_prod}_Price"].tolist()
    
    matrix = []
    for i in range(len(tickers)-1):
        f_sym = tickers[i]
        b_sym = tickers[i+1]
        spread = prices[i] - prices[i+1] # Front - Back
        
        matrix.append({
            "Pair": f"{f_sym}/{b_sym}",
            "Price": spread,
            "Leg 1": f"{prices[i]:.3f}",
            "Leg 2": f"{prices[i+1]:.3f}"
        })
        
    m_df = pd.DataFrame(matrix)
    
    # Bar Chart
    st.bar_chart(m_df.set_index("Pair")['Price'])
    
    # Clean Table
    st.dataframe(
        m_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn("Spread", format="%.3f"),
            "Leg 1": st.column_config.NumberColumn(format="%.3f"),
            "Leg 2": st.column_config.NumberColumn(format="%.3f")
        }
    )
