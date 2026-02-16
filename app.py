import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
import calendar

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="STIR Master Pro", 
    layout="wide", 
    page_icon="📉",
    initial_sidebar_state="expanded"
)

# Custom CSS for a tighter, trader-like feel
st.markdown("""
<style>
    .stMetric {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #333;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem; 
        color: #00F0FF; 
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Projected FOMC Dates for 2026 (Standard Cycle)
FOMC_MEETINGS = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6), 
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 11, 4), date(2026, 12, 16)
]

# Futures Ticker Codes
MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

# DV01 Constants
DV01_ZQ = 41.67
DV01_SR3 = 25.00

# -----------------------------------------------------------------------------
# 2. CORE LOGIC
# -----------------------------------------------------------------------------

def get_days_in_month(dt):
    return calendar.monthrange(dt.year, dt.month)[1]

def get_fomc_impact_ratio(contract_date):
    """
    Returns weights (w_old, w_new) for ZQ arithmetic averaging.
    """
    meeting = next((m for m in FOMC_MEETINGS if m.year == contract_date.year and m.month == contract_date.month), None)
    
    if not meeting:
        return (1.0, 0.0) 
    
    days_in_month = get_days_in_month(contract_date)
    days_before = meeting.day 
    days_after = days_in_month - days_before
    
    return (days_before / days_in_month, days_after / days_in_month)

@st.cache_data
def fetch_stir_data(base_effr, shock_bps=0):
    """
    Generates a Pro-Grade Synthetic Curve.
    """
    today = date.today()
    contracts = []
    
    # Curve Shape: Inverted initially (-4bps/month slope)
    curve_slope = -0.04 
    
    for i in range(18): 
        future_date = today + timedelta(days=30*i)
        if future_date < today: continue
            
        month_code = MONTH_CODES[future_date.month]
        year_code = str(future_date.year)[-1] 
        ticker_suffix = f"{month_code}{year_code}"
        
        # 1. Base Market Rate (with noise + shock)
        raw_rate = base_effr + (i * curve_slope) + (np.random.normal(0, 0.01))
        market_rate = raw_rate + (shock_bps / 100)

        # 2. ZQ Pricing (Arithmetic)
        zq_price = 100 - market_rate
        
        # 3. SR3 Pricing (Geometric + Spread)
        sr3_rate = market_rate - 0.08 
        sr3_price = 100 - sr3_rate
        
        # 4. Fair Value Calculation (Arb Logic)
        w_old, w_new = get_fomc_impact_ratio(future_date)
        # Assume market implies a cut path
        implied_cut_path = base_effr - (0.02 * i) 
        fair_rate = (w_old * implied_cut_path) + (w_new * (implied_cut_path - 0.25)) 
        fair_price_zq = 100 - fair_rate
        
        arb_signal = fair_price_zq - zq_price # Positive = Market is Cheap (Buy)

        contracts.append({
            "Expiry": future_date.strftime("%b %Y"),
            "MonthCode": ticker_suffix,
            "ZQ_Ticker": f"ZQ{ticker_suffix}",
            "ZQ_Price": round(zq_price, 3),
            "ZQ_Rate": round(100 - zq_price, 3),
            "Fair_Val_ZQ": round(fair_price_zq, 3),
            "Arb_Basis": round(arb_signal * 100, 1), # in bps
            "SR3_Ticker": f"SR3{ticker_suffix}",
            "SR3_Price": round(sr3_price, 3),
            "SR3_Rate": round(100 - sr3_price, 3),
            "Date_Obj": future_date
        })
        
    return pd.DataFrame(contracts)

# -----------------------------------------------------------------------------
# 3. UI LAYOUT
# -----------------------------------------------------------------------------

# --- Sidebar ---
st.sidebar.title("STIR Desk 🏛️")
mode = st.sidebar.radio("View", ["Term Structure", "Strategy Lab", "Spread Monitor"])

st.sidebar.markdown("---")
st.sidebar.header("Curve Assumptions")
base_rate = st.sidebar.number_input("Base EFFR (%)", 5.00, 6.00, 5.33)
shock = st.sidebar.slider("Curve Shock (bps)", -50, 50, 0)
st.sidebar.caption("Adjust 'Shock' to stress-test your Flys.")

# --- Load Data ---
df = fetch_stir_data(base_rate, shock)

st.title("STIR Trading Dashboard")
st.caption(f"Pricing: Synthetic Live | Shock Scenario: {shock} bps")

# --- Top Tape ---
cols = st.columns(6)
for i in range(min(6, len(df))):
    row = df.iloc[i]
    with cols[i]:
        st.metric(
            row['ZQ_Ticker'], 
            f"{row['ZQ_Price']:.3f}", 
            delta=f"{row['Arb_Basis']} bp Arb" 
        )

# -----------------------------------------------------------------------------
# MODULE 1: TERM STRUCTURE
# -----------------------------------------------------------------------------
if mode == "Term Structure":
    
    col_main, col_tbl = st.columns([2, 1])
    
    with col_main:
        st.subheader("Implied Rates Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Expiry'], y=df['ZQ_Rate'], name='ZQ (Fed Funds)', line=dict(color='#00F0FF', width=3)))
        fig.add_trace(go.Scatter(x=df['Expiry'], y=df['SR3_Rate'], name='SR3 (SOFR)', line=dict(color='orange', width=2, dash='dash')))
        
        fig.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_tbl:
        st.subheader("Market Data")
        # FIXED: Using column_config to prevent crashing on string columns
        st.dataframe(
            df[['Expiry', 'ZQ_Price', 'SR3_Price', 'Arb_Basis']],
            column_config={
                "Expiry": st.column_config.TextColumn("Contract"),
                "ZQ_Price": st.column_config.NumberColumn("ZQ Price", format="%.3f"),
                "SR3_Price": st.column_config.NumberColumn("SR3 Price", format="%.3f"),
                "Arb_Basis": st.column_config.NumberColumn("Arb (bps)", format="%.1f"),
            },
            hide_index=True,
            height=450
        )

# -----------------------------------------------------------------------------
# MODULE 2: STRATEGY LAB
# -----------------------------------------------------------------------------
elif mode == "Strategy Lab":
    st.subheader("🧪 Strategy Constructor")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.info("Construct Structure")
        strat = st.selectbox("Structure", ["Butterfly (Fly)", "Calendar Spread", "Condor"])
        root = st.selectbox("Root", ["ZQ", "SR3"])
        
        tickers = df[f'{root}_Ticker'].tolist()
        belly = st.selectbox("Center / Front Leg", tickers, index=2)
        
        # Logic for Legs
        legs = []
        try:
            idx = tickers.index(belly)
            if strat == "Butterfly (Fly)":
                width = st.slider("Fly Width (Months)", 1, 3, 1)
                w1 = tickers[idx - width]
                w2 = tickers[idx + width]
                legs = [
                    (1, df.loc[df[f'{root}_Ticker']==w1, f'{root}_Price'].values[0], w1),
                    (-2, df.loc[df[f'{root}_Ticker']==belly, f'{root}_Price'].values[0], belly),
                    (1, df.loc[df[f'{root}_Ticker']==w2, f'{root}_Price'].values[0], w2)
                ]
            elif strat == "Calendar Spread":
                back = tickers[idx + 1]
                legs = [
                    (1, df.loc[df[f'{root}_Ticker']==belly, f'{root}_Price'].values[0], belly),
                    (-1, df.loc[df[f'{root}_Ticker']==back, f'{root}_Price'].values[0], back)
                ]
        except IndexError:
            st.error("Structure out of bounds of the curve.")

        # Calc Price
        if legs:
            price = sum([q*p for q,p,n in legs])
            st.metric(f"Net Price", f"{price:.3f}")
            
            qty = st.number_input("Size (Lots)", 1, 1000, 100)
            
            # DV01
            unit_dv01 = DV01_ZQ if root == "ZQ" else DV01_SR3
            net_dv01 = sum([q*unit_dv01 for q,p,n in legs]) * qty
            st.write(f"**Net Risk:** ${net_dv01:.2f} / bp")

    with c2:
        if legs:
            st.markdown("### Payoff & Curve Visual")
            
            # Payoff Chart logic
            shifts = np.linspace(-50, 50, 21) # bps
            pnl_vals = []
            for s in shifts:
                # Price change approx = -1 * Shift * 100
                # PnL = Sum(Qty * PriceChange * UnitDV01)
                pnl = 0
                for q, p, n in legs:
                    pnl += (q * -(s/100) * 100 * unit_dv01) * qty
                pnl_vals.append(pnl)
                
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(x=shifts, y=pnl_vals, fill='tozeroy', line=dict(color='green')))
            fig_pnl.update_layout(title="PnL at Expiry vs Rate Shift", xaxis_title="Shift (bps)", yaxis_title="PnL ($)")
            st.plotly_chart(fig_pnl, use_container_width=True)
            
            # Execution Ticket
            st.markdown("#### Execution Ticket")
            exec_data = [{"Side": "BUY" if q>0 else "SELL", "Qty": abs(q*qty), "Contract": n, "Price": f"{p:.3f}"} for q,p,n in legs]
            st.table(pd.DataFrame(exec_data))

# -----------------------------------------------------------------------------
# MODULE 3: SPREAD MONITOR
# -----------------------------------------------------------------------------
elif mode == "Spread Monitor":
    st.subheader("🔥 Hot Spreads")
    
    # Auto-generate 1-month calendar spreads
    spreads = []
    for i in range(len(df)-1):
        front = df.iloc[i]
        back = df.iloc[i+1]
        
        spread_price = front['ZQ_Price'] - back['ZQ_Price']
        spread_name = f"{front['MonthCode']}/{back['MonthCode']}"
        
        spreads.append({
            "Pair": spread_name,
            "Price": spread_price,
            "Implied": f"{front['ZQ_Rate']:.2f}% -> {back['ZQ_Rate']:.2f}%",
            "Slope": "Inverted" if spread_price < 0 else "Steep"
        })
        
    sp_df = pd.DataFrame(spreads)
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.dataframe(
            sp_df,
            column_config={
                "Price": st.column_config.NumberColumn("Spread Price", format="%.3f"),
            },
            hide_index=True,
            height=600
        )
    with col_s2:
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Bar(x=sp_df['Pair'], y=sp_df['Price'], name='Spread'))
        fig_sp.update_layout(title="ZQ Calendar Spreads (1-Month Rolls)", template="plotly_dark")
        st.plotly_chart(fig_sp, use_container_width=True)
