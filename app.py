import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

# -----------------------------------------------------------------------------
# 1. SETUP & UTILS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="STIR Master Pro", layout="wide", page_icon="🏛️")

st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    button[kind="secondary"] { border: 1px solid #4CAF50 !important; color: #4CAF50 !important; }
</style>
""", unsafe_allow_html=True)

# CONSTANTS
DV01_MAP = {"ZQ": 41.67, "SR3": 25.00}
MONTH_CODES = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}

# DATA ENGINE
@st.cache_data
def get_market_data(base_rate, shock_bps):
    today = date.today()
    data = []
    for i in range(24):
        f_date = today + timedelta(days=30*i)
        slope = (i * 0.03) 
        rate = base_rate + slope + (shock_bps / 100.0)
        suffix = f"{MONTH_CODES[f_date.month]}{str(f_date.year)[-1]}"
        data.append({
            "Month": f_date.strftime("%b %y"), "Suffix": suffix,
            "ZQ_Price": 100 - rate, "SR3_Price": 100 - (rate - 0.05),
            "ZQ_Rate": rate, "SR3_Rate": rate - 0.05
        })
    return pd.DataFrame(data)

def get_tickers(df, prod): return [f"{prod}{r['Suffix']}" for _, r in df.iterrows()]
def get_px(df, t, p): 
    r = df[df['Suffix'] == t.replace(p,"")]
    return r.iloc[0][f"{p}_Price"] if not r.empty else 0.0

# -----------------------------------------------------------------------------
# 2. SESSION STATE & CALLBACKS
# -----------------------------------------------------------------------------
if 'sell_qty' not in st.session_state: st.session_state.sell_qty = 2.0
if 'last_struct' not in st.session_state: st.session_state.last_struct = "Butterfly (Fly)"

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.header("🏛️ STIR Desk")
mode = st.sidebar.radio("Mode", ["Market Overview", "Strategy Lab", "Spread Matrix"])
st.sidebar.divider()
base_rate = st.sidebar.number_input("Base Rate", 0.0, 20.0, 3.64)
shock = st.sidebar.slider("Shift (bps)", -50, 50, 0)
df = get_market_data(base_rate, shock)

# -----------------------------------------------------------------------------
# 4. STRATEGY LAB (The Fix)
# -----------------------------------------------------------------------------
if mode == "Strategy Lab":
    st.title("STIR Master Pro")
    st.subheader("🛠️ Strategy Constructor")

    # --- INPUTS ---
    c_str, c_lot = st.columns([2, 2])
    struct = c_str.selectbox("Structure", ["Calendar Spread", "Butterfly (Fly)", "Condor"])
    base_lots = c_lot.number_input("Base Lots", 1, 10000, 100)

    # Reset Logic
    if struct != st.session_state.last_struct:
        st.session_state.sell_qty = 1.0 if struct == "Calendar Spread" else 2.0
        st.session_state.last_struct = struct

    # --- DEFINE LEGS & CALCULATE RISKS ---
    legs = []
    buy_risk_total = 0   # Total DV01 of all Buy Legs
    sell_risk_unit = 0   # DV01 of 1 Unit of the Sell Leg(s)

    # 1. BUILD LEGS (Logic Only)
    if struct == "Butterfly (Fly)":
        # WINGS (Buy)
        for i, idx in enumerate([0, 6]):
            p = "ZQ" # Default
            t = get_tickers(df, p)[idx]
            legs.append({"Side": "BUY", "Qty": 1.0, "Prod": p, "Ticker": t, "Price": get_px(df,t,p)})
            buy_risk_total += DV01_MAP[p]
        
        # BELLY (Sell)
        p_sell = "SR3"
        t_sell = get_tickers(df, p_sell)[3]
        sell_risk_unit = DV01_MAP[p_sell] # Risk of 1 contract
        
        # We add the sell leg AFTER we determine the quantity below

    elif struct == "Condor":
        # OUTERS (Buy)
        for i, idx in enumerate([0, 6]):
            p = "ZQ"
            t = get_tickers(df, p)[idx]
            legs.append({"Side": "BUY", "Qty": 1.0, "Prod": p, "Ticker": t, "Price": get_px(df,t,p)})
            buy_risk_total += DV01_MAP[p]
            
        # INNERS (Sell)
        p_sell = "SR3"
        sell_risk_unit = DV01_MAP[p_sell] * 2 # We sell 2 different contracts (Body)
        # We track "sell_qty" as the amount per leg. So total risk multiplier is 2x.

    elif struct == "Calendar Spread":
        p = "ZQ"
        t = get_tickers(df, p)[0]
        legs.append({"Side": "BUY", "Qty": 1.0, "Prod": p, "Ticker": t, "Price": get_px(df,t,p)})
        buy_risk_total += DV01_MAP[p]
        
        p_sell = "SR3"
        sell_risk_unit = DV01_MAP[p_sell]

    # --- THE NEUTRALIZER (SOLVER) ---
    # We solve: Buy_Risk - (Sell_Qty * Sell_Risk_Unit) = 0
    # Sell_Qty = Buy_Risk / Sell_Risk_Unit
    
    optimal_qty = 0.0
    if sell_risk_unit > 0:
        optimal_qty = buy_risk_total / sell_risk_unit

    c_qty, c_btn = st.columns([1, 1])
    
    with c_qty:
        # The Input Box is linked to session_state.sell_qty
        # Use on_change to keep manual edits
        def update_qty(): st.session_state.sell_qty = st.session_state.u_qty
        
        qty_val = st.number_input("Sell Leg Qty", value=st.session_state.sell_qty, step=0.01, key="u_qty", on_change=update_qty)

    with c_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        # THE FIX: Button callback updates state DIRECTLY
        def neutralize():
            st.session_state.sell_qty = optimal_qty
            st.session_state.u_qty = optimal_qty # Force UI update
        
        if st.button("💡 Neutralize Delta", on_click=neutralize):
            st.rerun() # Force reload to show new number

    # --- ADD SELL LEGS TO LIST ---
    # Now we use the final (possibly updated) qty to finish the leg list
    final_qty = st.session_state.sell_qty
    
    if struct == "Butterfly (Fly)":
        legs.append({"Side": "SELL", "Qty": final_qty, "Prod": "SR3", "Ticker": get_tickers(df, "SR3")[3], "Price": get_px(df, get_tickers(df, "SR3")[3], "SR3")})
    elif struct == "Condor":
        # Add 2 middle legs
        for idx in [2, 4]:
            legs.append({"Side": "SELL", "Qty": final_qty, "Prod": "SR3", "Ticker": get_tickers(df, "SR3")[idx], "Price": get_px(df, get_tickers(df, "SR3")[idx], "SR3")})
    elif struct == "Calendar Spread":
        legs.append({"Side": "SELL", "Qty": final_qty, "Prod": "SR3", "Ticker": get_tickers(df, "SR3")[1], "Price": get_px(df, get_tickers(df, "SR3")[1], "SR3")})

    # --- TICKET & PNL ---
    st.divider()
    
    # Calc Net DV01
    net_dv01 = 0
    tick_data = []
    
    for l in legs:
        d = 1 if l['Side'] == "BUY" else -1
        risk = DV01_MAP[l['Prod']]
        # Risk = Direction * Qty * Lots * DV01
        leg_dv01 = d * l['Qty'] * base_lots * risk
        net_dv01 += leg_dv01
        
        tick_data.append({
            "Side": l['Side'], "Qty": f"{l['Qty']*base_lots:,.0f}", 
            "Prod": l['Prod'], "Ticker": l['Ticker'], "Price": f"{l['Price']:.3f}"
        })
        
    c1, c2 = st.columns([1, 2])
    with c1:
        st.table(pd.DataFrame(tick_data))
        st.metric("Net DV01", f"${net_dv01:,.2f}", delta="Neutral" if abs(net_dv01) < 10 else "Hedge Needed")
        
    with c2:
        # PnL Chart
        moves = np.arange(-25, 26, 1)
        pnl = []
        for m in moves: # Parallel Shift
            run = 0
            for l in legs:
                d = 1 if l['Side'] == "BUY" else -1
                risk = DV01_MAP[l['Prod']]
                # PnL = -1 * Shift(bps) * Direction * Qty * Lots * Risk
                run += (-1 * m * d * l['Qty'] * base_lots * risk)
            pnl.append(run)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=moves, y=pnl, fill='tozeroy', name='PnL', line=dict(color='#00E676' if pnl[-1]>=0 else '#FF5252')))
        fig.update_layout(title="PnL (Parallel Shift)", height=300, margin=dict(l=20,r=20,t=40,b=20), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. OTHER MODES
# -----------------------------------------------------------------------------
elif mode == "Market Overview":
    st.title("Market Overview")
    st.dataframe(df)
elif mode == "Spread Matrix":
    st.title("Spread Matrix")
    p = st.selectbox("Product", ["ZQ", "SR3"])
    ts = get_tickers(df, p)
    pxs = [get_px(df, t, p) for t in ts]
    # Simple adjacent spreads
    data = [{"Pair": f"{ts[i]}/{ts[i+1]}", "Spread": pxs[i]-pxs[i+1]} for i in range(len(ts)-1)]
    st.bar_chart(pd.DataFrame(data).set_index("Pair"))
