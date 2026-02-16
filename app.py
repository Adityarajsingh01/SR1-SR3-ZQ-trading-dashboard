import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="STIR Master Pro", layout="wide", page_icon="🏛️")

st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    .stAlert { border-left: 5px solid #00c805 !important; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & DATA ENGINE
# -----------------------------------------------------------------------------
DV01_MAP = {"ZQ": 41.67, "SR3": 25.00}
MONTH_CODES = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}

@st.cache_data
def get_market_data(base_rate, shock_bps):
    today = date.today()
    data = []
    # Generate 24 months of contracts
    for i in range(24):
        f_date = today + timedelta(days=30*i)
        # Simple curve shape: steepening
        slope_factor = (i * 0.025) 
        raw_rate = base_rate + slope_factor + (shock_bps / 100.0)
        
        # Prices
        zq_px = 100 - raw_rate
        sr3_px = 100 - (raw_rate - 0.05) # 5bps spread assumption
        
        suffix = f"{MONTH_CODES[f_date.month]}{str(f_date.year)[-1]}"
        
        data.append({
            "Month": f_date.strftime("%b %y"),
            "Suffix": suffix,
            "ZQ_Price": zq_px,
            "SR3_Price": sr3_px,
            "ZQ_Rate": raw_rate,
            "SR3_Rate": raw_rate - 0.05
        })
    return pd.DataFrame(data)

def get_tickers_list(df, prod):
    return [f"{prod}{row['Suffix']}" for _, row in df.iterrows()]

def get_price(df, ticker, prod):
    suffix = ticker.replace(prod, "")
    row = df[df['Suffix'] == suffix]
    if not row.empty:
        return row.iloc[0][f"{prod}_Price"]
    return 0.0

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.header("🏛️ STIR Desk")
view_mode = st.sidebar.radio("Workstation Mode", ["Market Overview", "Strategy Lab", "Spread Matrix"])

st.sidebar.divider()
st.sidebar.subheader("Curve Assumptions")
base_effr = st.sidebar.number_input("Base Rate (%)", 0.00, 20.00, 3.64, 0.01)
curve_shock = st.sidebar.slider("Parallel Shift (bps)", -50, 50, 0)

df = get_market_data(base_effr, curve_shock)

# -----------------------------------------------------------------------------
# 4. SESSION STATE (For Neutralizer)
# -----------------------------------------------------------------------------
# We track the "Sell Quantity" in state so the button can update it.
if 'sell_qty' not in st.session_state:
    st.session_state.sell_qty = 2.0 # Default start

if 'last_struct' not in st.session_state:
    st.session_state.last_struct = "Butterfly (Fly)"

# -----------------------------------------------------------------------------
# 5. MAIN LOGIC
# -----------------------------------------------------------------------------
st.title("STIR Master Pro")

# =============================================================================
# MODE: MARKET OVERVIEW
# =============================================================================
if view_mode == "Market Overview":
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Forward Rates")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Month'], y=df['ZQ_Rate'], name='ZQ (Fed Funds)', line=dict(color='#00F0FF', width=3)))
        fig.add_trace(go.Scatter(x=df['Month'], y=df['SR3_Rate'], name='SR3 (SOFR)', line=dict(color='#FFA500', width=2, dash='dash')))
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Pricing Board")
        st.dataframe(df[['Month', 'ZQ_Price', 'SR3_Price']], hide_index=True, use_container_width=True, height=450)

# =============================================================================
# MODE: SPREAD MATRIX
# =============================================================================
elif view_mode == "Spread Matrix":
    st.subheader("📅 Spread Matrix")
    prod = st.selectbox("Curve", ["ZQ", "SR3"])
    
    tickers = get_tickers_list(df, prod)
    prices = [get_price(df, t, prod) for t in tickers]
    
    matrix_data = []
    # Calculate front-month spreads (Calendar Spreads)
    for i in range(len(tickers)-1):
        spread_val = prices[i] - prices[i+1]
        matrix_data.append({
            "Pair": f"{tickers[i]}/{tickers[i+1]}",
            "Spread": spread_val,
            "Front": prices[i],
            "Back": prices[i+1]
        })
    
    m_df = pd.DataFrame(matrix_data)
    
    # Bar Chart
    st.bar_chart(m_df.set_index("Pair")['Spread'])
    # Data Table
    st.dataframe(m_df, hide_index=True, use_container_width=True)

# =============================================================================
# MODE: STRATEGY LAB
# =============================================================================
elif view_mode == "Strategy Lab":
    st.subheader("🛠️ Strategy Constructor")
    
    # --- CONTROLS ---
    c_struct, c_size, c_btn = st.columns([2, 1, 1])
    struct = c_struct.selectbox("Structure", ["Calendar Spread", "Butterfly (Fly)", "Condor"])
    base_lots = c_size.number_input("Base Size (Lots)", 1, 10000, 100)
    
    # Reset default qty if structure changes
    if struct != st.session_state.last_struct:
        if struct == "Calendar Spread": st.session_state.sell_qty = 1.0
        elif struct == "Butterfly (Fly)": st.session_state.sell_qty = 2.0
        elif struct == "Condor": st.session_state.sell_qty = 1.0
        st.session_state.last_struct = struct

    # --- LEG GENERATION ---
    legs = []
    
    # We calculate risk to enable the neutralizer
    fixed_legs_risk = 0   # Total DV01 of fixed legs (usually Buy legs)
    variable_leg_risk_unit = 0 # DV01 of 1 unit of the variable leg (Sell leg)

    if struct == "Butterfly (Fly)":
        c1, c2, c3 = st.columns(3)
        # Wing 1
        with c1:
            st.markdown("**Wing 1 (Buy)**")
            p1 = st.selectbox("Prod", ["ZQ", "SR3"], key="f1p")
            t1 = st.selectbox("Cont", get_tickers_list(df, p1), index=0, key="f1t")
            legs.append({"Side": "BUY", "Qty": 1.0, "Prod": p1, "Ticker": t1, "Price": get_price(df, t1, p1)})
            fixed_legs_risk += (1.0 * DV01_MAP[p1])
        
        # Wing 2
        with c3:
            st.markdown("**Wing 2 (Buy)**")
            p3 = st.selectbox("Prod", ["ZQ", "SR3"], key="f3p")
            t3 = st.selectbox("Cont", get_tickers_list(df, p3), index=6, key="f3t")
            legs.append({"Side": "BUY", "Qty": 1.0, "Prod": p3, "Ticker": t3, "Price": get_price(df, t3, p3)})
            fixed_legs_risk += (1.0 * DV01_MAP[p3])

        # Belly (Variable)
        with c2:
            st.markdown("**Belly (Sell)**")
            p2 = st.selectbox("Prod", ["ZQ", "SR3"], key="f2p")
            t2 = st.selectbox("Cont", get_tickers_list(df, p2), index=3, key="f2t")
            
            # Input linked to Session State
            qty_val = st.number_input("Qty", value=float(st.session_state.sell_qty), step=0.01, format="%.2f", key="qty_input")
            st.session_state.sell_qty = qty_val # Sync back
            
            legs.append({"Side": "SELL", "Qty": qty_val, "Prod": p2, "Ticker": t2, "Price": get_price(df, t2, p2)})
            variable_leg_risk_unit = -1 * DV01_MAP[p2] # Negative because it's a sell, and we track 'unit' risk

    elif struct == "Condor":
        cols = st.columns(4)
        # Leg 1 Buy
        with cols[0]:
            p = st.selectbox("L1", ["ZQ", "SR3"], key="c1p")
            t = st.selectbox("Con", get_tickers_list(df, p), index=0, key="c1t")
            legs.append({"Side": "BUY", "Qty": 1.0, "Prod": p, "Ticker": t, "Price": get_price(df, t, p)})
            fixed_legs_risk += (1.0 * DV01_MAP[p])
        
        # Leg 4 Buy
        with cols[3]:
            p = st.selectbox("L4", ["ZQ", "SR3"], key="c4p")
            t = st.selectbox("Con", get_tickers_list(df, p), index=6, key="c4t")
            legs.append({"Side": "BUY", "Qty": 1.0, "Prod": p, "Ticker": t, "Price": get_price(df, t, p)})
            fixed_legs_risk += (1.0 * DV01_MAP[p])

        # Shared Sell Qty
        qty_val = st.number_input("Body Qty (per leg)", value=float(st.session_state.sell_qty), step=0.01, format="%.2f", key="qty_input_condor")
        st.session_state.sell_qty = qty_val

        # Leg 2 Sell
        with cols[1]:
            p = st.selectbox("L2", ["ZQ", "SR3"], key="c2p")
            t = st.selectbox("Con", get_tickers_list(df, p), index=2, key="c2t")
            legs.append({"Side": "SELL", "Qty": qty_val, "Prod": p, "Ticker": t, "Price": get_price(df, t, p)})
            variable_leg_risk_unit += (-1 * DV01_MAP[p]) # Add to variable risk pool

        # Leg 3 Sell
        with cols[2]:
            p = st.selectbox("L3", ["ZQ", "SR3"], key="c3p")
            t = st.selectbox("Con", get_tickers_list(df, p), index=4, key="c3t")
            legs.append({"Side": "SELL", "Qty": qty_val, "Prod": p, "Ticker": t, "Price": get_price(df, t, p)})
            variable_leg_risk_unit += (-1 * DV01_MAP[p])

    elif struct == "Calendar Spread":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Front (Buy)**")
            p1 = st.selectbox("Prod", ["ZQ", "SR3"], key="cp1")
            t1 = st.selectbox("Cont", get_tickers_list(df, p1), index=0, key="ct1")
            legs.append({"Side": "BUY", "Qty": 1.0, "Prod": p1, "Ticker": t1, "Price": get_price(df, t1, p1)})
            fixed_legs_risk += (1.0 * DV01_MAP[p1])
            
        with c2:
            st.markdown("**Back (Sell)**")
            p2 = st.selectbox("Prod", ["ZQ", "SR3"], key="cp2")
            t2 = st.selectbox("Cont", get_tickers_list(df, p2), index=1, key="ct2")
            
            qty_val = st.number_input("Qty", value=float(st.session_state.sell_qty), step=0.01, format="%.2f", key="qty_input_cal")
            st.session_state.sell_qty = qty_val
            
            legs.append({"Side": "SELL", "Qty": qty_val, "Prod": p2, "Ticker": t2, "Price": get_price(df, t2, p2)})
            variable_leg_risk_unit = -1 * DV01_MAP[p2]

    # --- NEUTRALIZE BUTTON LOGIC ---
    with c_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        # Math: Fixed_Risk + (Qty * Variable_Unit_Risk) = 0
        # Qty = - Fixed_Risk / Variable_Unit_Risk
        if st.button("💡 Neutralize Delta"):
            if variable_leg_risk_unit != 0:
                optimal_qty = -1 * (fixed_legs_risk / variable_leg_risk_unit)
                st.session_state.sell_qty = abs(optimal_qty) # Store as positive magnitude
                st.rerun()
            else:
                st.error("Cannot neutralize: Sell leg has 0 risk.")

    # --- OUTPUT SECTION ---
    st.divider()
    
    # 1. Calc Net DV01
    net_dv01 = 0
    ticket_data = []
    
    for leg in legs:
        # Determine sign for DV01 calc (Buy = +, Sell = -)
        direction = 1 if leg['Side'] == "BUY" else -1
        # DV01 per lot (always positive constant)
        risk_per_lot = DV01_MAP[leg['Prod']]
        
        # Leg DV01 = Direction * Qty * Risk_Constant * Base_Lots
        # Note: leg['Qty'] from inputs is usually positive magnitude, so handle carefully
        qty_magnitude = leg['Qty']
        
        leg_dv01 = direction * qty_magnitude * risk_per_lot * base_lots
        net_dv01 += leg_dv01
        
        ticket_data.append({
            "Side": leg['Side'],
            "Qty": f"{qty_magnitude * base_lots:,.0f}",
            "Ticker": leg['Ticker'],
            "Price": f"{leg['Price']:.3f}"
        })

    # 2. Display
    c_tick, c_risk = st.columns([1, 2])
    
    with c_tick:
        st.markdown("#### 🎫 Ticket")
        st.table(pd.DataFrame(ticket_data))
        
        st.markdown("#### 📊 Sensitivity")
        c_r1, c_r2 = st.columns(2)
        c_r1.metric("Net DV01 ($)", f"${net_dv01:,.2f}")
        
        if abs(net_dv01) < (base_lots * 1.0):
            status = "Neutral"
            clr = "off"
        elif net_dv01 > 0:
            status = "Bullish"
            clr = "normal" # Green
        else:
            status = "Bearish"
            clr = "inverse" # Red
        
        c_r2.metric("Bias", status, delta=f"{net_dv01:.0f}", delta_color=clr)

    with c_risk:
        st.markdown("#### ⚠️ PnL Simulation")
        mode = st.radio("Scenario", ["Parallel Shift", "Curve Twist (Steepener)"], horizontal=True)
        
        moves = np.arange(-25, 26, 1)
        pnl_vals = []
        
        center_idx = (len(legs) - 1) / 2.0
        
        for m in moves:
            run_pnl = 0
            for i, leg in enumerate(legs):
                # Calc Shift
                if mode == "Parallel Shift":
                    shift_bps = m
                else:
                    dist = i - center_idx
                    shift_bps = m * dist
                
                # Calc PnL
                # PnL = -1 * Shift(bps) * DV01($)
                # We already calculated 'leg_dv01' inside the loop above, but need to recap for loop
                direction = 1 if leg['Side'] == "BUY" else -1
                risk = DV01_MAP[leg['Prod']]
                qty = leg['Qty'] * base_lots
                
                leg_risk_dollars = direction * qty * risk
                
                # PnL formula
                leg_pnl = -1 * shift_bps * leg_risk_dollars
                run_pnl += leg_pnl
            
            pnl_vals.append(run_pnl)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=moves, y=pnl_vals, 
            fill='tozeroy', 
            name='PnL',
            line=dict(color='#00E676' if pnl_vals[-1] >= 0 else '#FF5252')
        ))
        fig.update_layout(
            title=f"PnL vs {mode}",
            xaxis_title="Move (bps)",
            yaxis_title="Profit/Loss ($)",
            template="plotly_dark",
            height=300,
            margin=dict(l=20,r=20,t=40,b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if abs(net_dv01) < (base_lots * 1.0) and mode == "Parallel Shift":
             st.success("✅ Perfectly Hedged: PnL is flat on parallel moves.")
