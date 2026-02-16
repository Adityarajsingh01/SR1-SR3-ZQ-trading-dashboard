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
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. FINANCIAL CONSTANTS & DATA
# -----------------------------------------------------------------------------
DV01_MAP = {"ZQ": 41.67, "SR3": 25.00}
MONTH_CODES = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}

@st.cache_data
def get_market_data(base_rate, shock_bps):
    today = date.today()
    data = []
    # Simple curve model: slight inversion then steepening
    for i in range(24):
        f_date = today + timedelta(days=30*i)
        slope_factor = (i * 0.02) if i < 6 else (i * 0.04) 
        raw_rate = base_rate + slope_factor + (shock_bps / 100.0)
        
        data.append({
            "Month": f_date.strftime("%b %y"),
            "Suffix": f"{MONTH_CODES[f_date.month]}{str(f_date.year)[-1]}",
            "ZQ_Price": 100 - raw_rate,
            "SR3_Price": 100 - (raw_rate - 0.05), # Spread assumption
            "ZQ_Rate": raw_rate,
            "SR3_Rate": raw_rate - 0.05
        })
    return pd.DataFrame(data)

def get_tickers_list(df, prod):
    return [f"{prod}{row['Suffix']}" for _, row in df.iterrows()]

def get_price_from_ticker(df, ticker, prod):
    # Extract suffix from ticker (e.g., "ZQH6" -> "H6")
    suffix = ticker.replace(prod, "")
    row = df[df['Suffix'] == suffix]
    if not row.empty:
        return row.iloc[0][f"{prod}_Price"]
    return 0.0

# -----------------------------------------------------------------------------
# 3. STATE MANAGEMENT
# -----------------------------------------------------------------------------
# We use session state to hold the 'Sell Quantity' so the button can update it
if 'sell_qty' not in st.session_state:
    st.session_state.sell_qty = -2.00 # Default for Butterfly

if 'last_struct' not in st.session_state:
    st.session_state.last_struct = "Butterfly (Fly)"

# -----------------------------------------------------------------------------
# 4. SIDEBAR & DATA LOADING
# -----------------------------------------------------------------------------
st.sidebar.header("🏛️ STIR Desk")
view_mode = st.sidebar.radio("Workstation Mode", ["Market Overview", "Strategy Lab", "Spread Matrix"])
st.sidebar.divider()
base_effr = st.sidebar.number_input("Base Rate (%)", 0.0, 10.0, 3.64)
curve_shock = st.sidebar.slider("Parallel Shift (bps)", -50, 50, 0)

df = get_market_data(base_effr, curve_shock)

# -----------------------------------------------------------------------------
# 5. MAIN LOGIC
# -----------------------------------------------------------------------------
st.title("STIR Master Pro")

if view_mode == "Strategy Lab":
    st.subheader("🛠️ Strategy Constructor")
    
    # --- TOP CONTROLS ---
    c1, c2, c3 = st.columns([2, 1, 1])
    struct = c1.selectbox("Structure", ["Calendar Spread", "Butterfly (Fly)", "Condor"])
    base_lots = c2.number_input("Base Size (Lots)", 1, 10000, 100)
    
    # Reset default quantities if structure changes
    if struct != st.session_state.last_struct:
        if struct == "Calendar Spread": st.session_state.sell_qty = -1.0
        elif struct == "Butterfly (Fly)": st.session_state.sell_qty = -2.0
        elif struct == "Condor": st.session_state.sell_qty = -1.0
        st.session_state.last_struct = struct

    # --- LEG DEFINITIONS ---
    legs = []
    
    # We will build "Long Risk" (to be hedged) and "Short Risk Unit" (the hedger)
    buy_legs_dv01 = 0
    sell_leg_dv01_unit = 0 

    if struct == "Butterfly (Fly)":
        col_w1, col_belly, col_w2 = st.columns(3)
        
        # Wing 1 (Fixed Buy)
        with col_w1:
            st.markdown("**(1) Wing 1 [Buy]**")
            p1 = st.selectbox("Prod", ["ZQ", "SR3"], key="p1")
            t1 = st.selectbox("Cont", get_tickers_list(df, p1), index=0, key="t1")
            price1 = get_price_from_ticker(df, t1, p1)
            legs.append({"Side": "BUY", "Qty": 1, "Prod": p1, "Ticker": t1, "Price": price1})
            buy_legs_dv01 += (1 * DV01_MAP[p1])

        # Wing 2 (Fixed Buy)
        with col_w2:
            st.markdown("**(3) Wing 2 [Buy]**")
            p3 = st.selectbox("Prod", ["ZQ", "SR3"], key="p3")
            t3 = st.selectbox("Cont", get_tickers_list(df, p3), index=6, key="t3")
            price3 = get_price_from_ticker(df, t3, p3)
            legs.append({"Side": "BUY", "Qty": 1, "Prod": p3, "Ticker": t3, "Price": price3})
            buy_legs_dv01 += (1 * DV01_MAP[p3])

        # Belly (Variable Sell)
        with col_belly:
            st.markdown("**(2) Belly [Sell]**")
            p2 = st.selectbox("Prod", ["ZQ", "SR3"], key="p2")
            t2 = st.selectbox("Cont", get_tickers_list(df, p2), index=3, key="t2")
            price2 = get_price_from_ticker(df, t2, p2)
            
            # THE INPUT BOX IS LINKED TO SESSION STATE
            qty2 = st.number_input("Belly Qty", value=st.session_state.sell_qty, step=0.01, format="%.2f", key="qty_input")
            # Update state immediately if user types manually
            st.session_state.sell_qty = qty2
            
            legs.append({"Side": "SELL", "Qty": qty2, "Prod": p2, "Ticker": t2, "Price": price2})
            sell_leg_dv01_unit = DV01_MAP[p2] # The risk of 1 unit of this leg

    elif struct == "Calendar Spread":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**(1) Leg 1 [Buy]**")
            p1 = st.selectbox("Prod", ["ZQ", "SR3"], key="cp1")
            t1 = st.selectbox("Cont", get_tickers_list(df, p1), index=0, key="ct1")
            price1 = get_price_from_ticker(df, t1, p1)
            legs.append({"Side": "BUY", "Qty": 1, "Prod": p1, "Ticker": t1, "Price": price1})
            buy_legs_dv01 += (1 * DV01_MAP[p1])
            
        with col2:
            st.markdown("**(2) Leg 2 [Sell]**")
            p2 = st.selectbox("Prod", ["ZQ", "SR3"], key="cp2")
            t2 = st.selectbox("Cont", get_tickers_list(df, p2), index=1, key="ct2")
            price2 = get_price_from_ticker(df, t2, p2)
            
            qty2 = st.number_input("Leg 2 Qty", value=st.session_state.sell_qty, step=0.01, format="%.2f", key="c_qty_input")
            st.session_state.sell_qty = qty2
            
            legs.append({"Side": "SELL", "Qty": qty2, "Prod": p2, "Ticker": t2, "Price": price2})
            sell_leg_dv01_unit = DV01_MAP[p2]
    
    # --- NEUTRALIZE LOGIC ---
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💡 Neutralize Delta"):
            # Math: Total Buy Risk + (Sell_Qty * Sell_Unit_Risk) = 0
            # Sell_Qty = - (Total Buy Risk) / Sell_Unit_Risk
            if sell_leg_dv01_unit != 0:
                optimal_qty = -1 * (buy_legs_dv01 / sell_leg_dv01_unit)
                st.session_state.sell_qty = optimal_qty
                st.rerun()

    # --- TICKET & CALCULATIONS ---
    st.divider()
    
    # Calculate Totals
    total_dv01 = 0
    ticket_rows = []
    
    for leg in legs:
        abs_qty = leg['Qty'] * base_lots
        leg_dv01 = abs_qty * -1 * DV01_MAP[leg['Prod']] # -1 because DV01 is price sensitivity
        total_dv01 += leg_dv01
        
        ticket_rows.append({
            "Side": "BUY" if leg['Qty'] > 0 else "SELL",
            "Qty": f"{abs(abs_qty):,.2f}",
            "Product": leg['Prod'],
            "Contract": leg['Ticker'],
            "Price": f"{leg['Price']:.3f}"
        })

    # Columns
    c_tick, c_risk = st.columns([1, 2])
    
    with c_tick:
        st.markdown("#### 🎫 Live Ticket")
        st.table(pd.DataFrame(ticket_rows))
        
        st.markdown("#### 📊 Risk Stats")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Net DV01", f"${total_dv01:,.2f}")
        
        # Determine Bias
        if abs(total_dv01) < 1.0: # Tight tolerance
            bias = "Neutral"
            color = "off"
        elif total_dv01 > 0:
            bias = "Bullish"
            color = "normal"
        else:
            bias = "Bearish"
            color = "inverse"
        col_m2.metric("Bias", bias, delta=f"{total_dv01:.1f}", delta_color=color)

    with c_risk:
        st.markdown("#### ⚠️ PnL Simulation")
        sim_type = st.radio("Simulation Mode", ["Parallel Shift", "Curve Twist (Steepener)"], horizontal=True)
        
        moves = range(-25, 26)
        pnl_data = []
        
        # Center index for Twist pivoting
        center_idx = (len(legs) - 1) / 2
        
        for m in moves:
            scenario_pnl = 0
            for i, leg in enumerate(legs):
                # DV01 is per lot
                risk_per_lot = DV01_MAP[leg['Prod']]
                
                # Determine Shift for this leg
                if sim_type == "Parallel Shift":
                    shift_bps = m
                else:
                    # Twist: Pivot around the middle leg
                    # If m is positive (Steepener): Front legs go down (rates up), Back legs go up (rates down)
                    dist = i - center_idx 
                    shift_bps = m * dist 
                
                # PnL = -1 * Shift * Risk * Lots
                # Note: leg['Qty'] is signed (+ for Buy, - for Sell)
                # But risk calc: Long position loses on rate hike (Shift > 0)
                # Formula: Qty * (Price_Change)
                # Price_Change approx = -1 * Shift * DV01_per_lot
                
                leg_pnl = (leg['Qty'] * base_lots) * (-1 * shift_bps/100 * risk_per_lot * 100) 
                # Simplified: Qty * -Shift * DV01
                # But wait, DV01 is usually defined for 1bp.
                # So: Qty * -1 * Shift_in_bps * DV01_value
                
                leg_pnl = (leg['Qty'] * base_lots) * (-1 * shift_bps * risk_per_lot)
                scenario_pnl += leg_pnl
            
            pnl_data.append(scenario_pnl)
            
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(moves), y=pnl_data, 
            fill='tozeroy', 
            name='PnL',
            line=dict(color='#00ff00' if pnl_data[-1] >= 0 else '#ff0000', width=2)
        ))
        
        fig.update_layout(
            title=f"PnL vs {sim_type} (bps)",
            xaxis_title="Move (bps)",
            yaxis_title="PnL ($)",
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if abs(total_dv01) < 1.0 and sim_type == "Parallel Shift":
            st.info("ℹ️ The line is flat because you are perfectly hedged against parallel shifts.")

# -----------------------------------------------------------------------------
# OTHER MODES (Simplified for brevity)
# -----------------------------------------------------------------------------
elif view_mode == "Market Overview":
    st.dataframe(df)
elif view_mode == "Spread Matrix":
    st.write("Spread Matrix Placeholder")
