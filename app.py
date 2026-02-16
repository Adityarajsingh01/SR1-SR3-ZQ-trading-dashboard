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
    /* Pro Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important; 
        font-weight: 700;
        color: #E0E0E0;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
    }
    div[data-testid="stDataFrame"] {
        width: 100%;
    }
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
DV01_ZQ = 41.67
DV01_SR1 = 41.67 # 1M SOFR is also monthly accrual
DV01_SR3 = 25.00 # 3M SOFR is quarterly

def get_days_in_month(dt):
    return calendar.monthrange(dt.year, dt.month)[1]

def get_fomc_weights(contract_date):
    """Returns (weight_old, weight_new) based on meeting day."""
    meeting = next((m for m in FOMC_MEETINGS_2026 if m.year == contract_date.year and m.month == contract_date.month), None)
    if not meeting: return (1.0, 0.0)
    
    dim = get_days_in_month(contract_date)
    d_old = meeting.day 
    return (d_old / dim, (dim - d_old) / dim)

# -----------------------------------------------------------------------------
# 3. DATA ENGINE (Now with SR1)
# -----------------------------------------------------------------------------
@st.cache_data
def get_curve_data(base_rate, shock_bps):
    today = date.today()
    data = []
    
    # Curve Shape: Inverted initially (-3bps/month slope)
    slope = -0.03
    
    for i in range(18):
        f_date = today + timedelta(days=30*i)
        if f_date < today: continue
        
        mc = MONTH_CODES[f_date.month]
        yc = str(f_date.year)[-1]
        ticker_suffix = f"{mc}{yc}"
        
        # 1. Underlying Rates
        # Base Curve
        raw_rate = base_rate + (i * slope) 
        
        # Apply User Shock
        final_rate_effr = raw_rate + (shock_bps/100)
        
        # SOFR Basis (Usually trades below EFFR, e.g., -5bps)
        sofr_rate = final_rate_effr - 0.05
        
        # 2. Contract Pricing
        # ZQ (Fed Funds)
        zq_price = 100 - final_rate_effr
        
        # SR1 (1-Month SOFR)
        # Trades very close to Monthly Average SOFR
        sr1_price = 100 - sofr_rate
        
        # SR3 (3-Month SOFR)
        # 3M Quarterly compounding usually adds slight term premium/convexity
        # We approximate it as the 1M rate + small steepener
        sr3_rate = sofr_rate + 0.02 
        sr3_price = 100 - sr3_rate

        # 3. Arb Logic (ZQ Fair Value)
        w_old, w_new = get_fomc_weights(f_date)
        # Simple arb model: assumes rate cut cycle pricing
        fair_rate = (w_old * final_rate_effr) + (w_new * (final_rate_effr - 0.25)) 
        fair_price_zq = 100 - fair_rate
        arb = fair_price_zq - zq_price

        data.append({
            "Expiry": f_date.strftime("%b %y"),
            "Suffix": ticker_suffix,
            
            # Tickers
            "ZQ_Ticker": f"ZQ{ticker_suffix}",
            "SR1_Ticker": f"SR1{ticker_suffix}",
            "SR3_Ticker": f"SR3{ticker_suffix}",
            
            # Prices
            "ZQ": round(zq_price, 3),
            "SR1": round(sr1_price, 3),
            "SR3": round(sr3_price, 3),
            
            # Rates
            "ZQ_Rate": round(100 - zq_price, 3),
            "SR1_Rate": round(100 - sr1_price, 3),
            "SR3_Rate": round(100 - sr3_price, 3),
            
            "Arb_bps": round(arb * 100, 1),
            "Date": f_date
        })
        
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("STIR Desk Controls")
mode = st.sidebar.radio("Workstation Mode", ["Market Overview", "Strategy Lab", "Spread Matrix"])

st.sidebar.divider()
st.sidebar.subheader("Curve Inputs")
# FIX: Min value 0.0 allows inputs like 3.64
base_effr = st.sidebar.number_input("Base EFFR (%)", min_value=0.0, max_value=20.0, value=3.64, step=0.01)
curve_shock = st.sidebar.slider("Parallel Shock (bps)", -50, 50, 0)

st.sidebar.divider()
st.sidebar.subheader("Tape Settings")
# FIX: Added Selector for Ticker Tape
tape_source = st.sidebar.selectbox("Tape Instrument", ["ZQ (Fed Funds)", "SR1 (1M SOFR)", "SR3 (3M SOFR)"])

# Load Data
df = get_curve_data(base_effr, curve_shock)

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD
# -----------------------------------------------------------------------------
st.title(f"STIR Trading Dashboard")
st.caption(f"Pricing Date: {date.today()} | Simulation Mode: Active")

# --- Top Tape (Dynamic) ---
cols = st.columns(6)
for i in range(min(6, len(df))):
    row = df.iloc[i]
    
    # Determine what to show based on sidebar selection
    if "ZQ" in tape_source:
        lbl = row['ZQ_Ticker']
        val = row['ZQ']
        chg = row['Arb_bps'] # using Arb as 'change' proxy for demo
        delta_lbl = "bp Arb"
    elif "SR1" in tape_source:
        lbl = row['SR1_Ticker']
        val = row['SR1']
        chg = row['SR1_Rate'] - base_effr # spread to effr
        delta_lbl = "Sprd"
    else: # SR3
        lbl = row['SR3_Ticker']
        val = row['SR3']
        chg = row['SR3_Rate'] - base_effr
        delta_lbl = "Sprd"

    with cols[i]:
        st.metric(
            label=lbl, 
            value=f"{val:.3f}", 
            delta=f"{chg:.2f} {delta_lbl}",
            delta_color="normal"
        )

st.divider()

# -----------------------------------------------------------------------------
# VIEW: MARKET OVERVIEW
# -----------------------------------------------------------------------------
if mode == "Market Overview":
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Forward Rates Structure")
        fig = go.Figure()
        # Plot all 3 curves
        fig.add_trace(go.Scatter(x=df['Expiry'], y=df['ZQ_Rate'], name='ZQ (Fed Funds)', line=dict(color='#00F0FF', width=3)))
        fig.add_trace(go.Scatter(x=df['Expiry'], y=df['SR1_Rate'], name='SR1 (1M SOFR)', line=dict(color='#FFE800', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Expiry'], y=df['SR3_Rate'], name='SR3 (3M SOFR)', line=dict(color='#FF8C00', width=2, dash='dash')))
        
        fig.update_layout(template="plotly_dark", height=450, xaxis_title="Contract", yaxis_title="Implied Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Live Quotes")
        st.dataframe(
            df[['Expiry', 'ZQ', 'SR1', 'SR3']], 
            column_config={
                "Expiry": "Mth",
                "ZQ": st.column_config.NumberColumn("ZQ", format="%.3f"),
                "SR1": st.column_config.NumberColumn("SR1", format="%.3f"),
                "SR3": st.column_config.NumberColumn("SR3", format="%.3f"),
            },
            hide_index=True,
            use_container_width=True,
            height=450
        )

# -----------------------------------------------------------------------------
# VIEW: STRATEGY LAB
# -----------------------------------------------------------------------------
elif mode == "Strategy Lab":
    st.subheader("🧪 Strategy Constructor")
    
    col_builder, col_viz = st.columns([1, 2])
    
    with col_builder:
        st.info("Define Structure")
        strat_type = st.selectbox("Type", ["Butterfly (Fly)", "Calendar Spread"])
        # FIX: Added SR1 to selector
        root = st.selectbox("Product", ["ZQ", "SR1", "SR3"])
        
        # Helper to get correct ticker col
        ticker_col = f"{root}_Ticker"
        price_col = root # 'ZQ', 'SR1', or 'SR3'
        
        tickers = df[ticker_col].tolist()
        belly_ticker = st.selectbox("Center / Front Leg", tickers, index=2)
        
        legs = []
        try:
            # Find index in dataframe
            row_idx = df.index[df[ticker_col] == belly_ticker][0]
            
            if strat_type == "Butterfly (Fly)":
                width = st.slider("Wing Width", 1, 3, 1)
                if row_idx - width >= 0 and row_idx + width < len(df):
                    w1 = df.iloc[row_idx - width]
                    belly = df.iloc[row_idx]
                    w2 = df.iloc[row_idx + width]
                    
                    legs = [
                        (1, w1[price_col], w1[ticker_col]), 
                        (-2, belly[price_col], belly[ticker_col]), 
                        (1, w2[price_col], w2[ticker_col])
                    ]
            elif strat_type == "Calendar Spread":
                if row_idx + 1 < len(df):
                    front = df.iloc[row_idx]
                    back = df.iloc[row_idx+1]
                    legs = [(1, front[price_col], front[ticker_col]), (-1, back[price_col], back[ticker_col])]
        except Exception as e:
            st.error(f"Invalid Structure: {e}")

        if legs:
            entry_price = sum([q*p for q,p,n in legs])
            st.metric("Net Package Price", f"{entry_price:.3f}")
            
            qty = st.number_input("Quantity (Lots)", 100, 5000, 100, step=100)
            
            # Determine DV01
            if root == "ZQ" or root == "SR1":
                dv01 = DV01_ZQ 
            else:
                dv01 = DV01_SR3
            
            st.markdown("---")
            st.write(" **Risk Analysis**")
            
            sim_curve = st.slider("Curve Twist (bps)", -10, 10, 0, help="Moves belly relative to wings")
            
            # PnL Calc
            pnl_est = 0
            for q, p, n in legs:
                shift = sim_curve if abs(q) == 2 else 0
                price_delta = -(shift / 100)
                pnl_est += (q * qty * price_delta * dv01)
                
            st.metric("Simulated PnL", f"${pnl_est:,.0f}", delta=f"{sim_curve} bps Twist")

    with col_viz:
        if legs:
            st.subheader("Payoff Profile")
            twists = np.linspace(-15, 15, 31) 
            pnl_curve = []
            
            for t in twists:
                daily_pnl = 0
                for q, p, n in legs:
                    s = t if abs(q) == 2 else 0 
                    p_delta = -(s/100)
                    daily_pnl += (q * qty * p_delta * dv01)
                pnl_curve.append(daily_pnl)

            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=twists, y=pnl_curve, 
                fill='tozeroy', 
                line=dict(color='#00FF00' if pnl_curve[-1]>0 else '#FF0000'),
                name="PnL"
            ))
            fig_pnl.update_layout(
                title="PnL vs Belly Rate Shift (Curvature Risk)",
                xaxis_title="Belly Rate Move (bps)",
                yaxis_title="PnL ($)",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
            
            st.write("#### Execution Ticket")
            ex_df = pd.DataFrame(legs, columns=["Qty", "Price", "Contract"])
            ex_df['Side'] = ex_df['Qty'].apply(lambda x: "BUY" if x > 0 else "SELL")
            ex_df['Abs Qty'] = ex_df['Qty'].abs() * qty
            st.dataframe(ex_df[['Side', 'Abs Qty', 'Contract', 'Price']], hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# VIEW: SPREAD MATRIX
# -----------------------------------------------------------------------------
elif mode == "Spread Matrix":
    st.subheader("Calendar Spread Monitor (1-Month Rolls)")
    
    # Allow user to pick which curve to analyze
    curve_choice = st.radio("Select Curve", ["ZQ", "SR1", "SR3"], horizontal=True)
    
    spreads = []
    for i in range(len(df)-1):
        front = df.iloc[i]
        back = df.iloc[i+1]
        
        # Use curve_choice to select column
        val = front[curve_choice] - back[curve_choice]
        
        spreads.append({
            "Pair": f"{front['Suffix']}/{back['Suffix']}",
            "Spread": val,
            "Front": front[curve_choice],
            "Back": back[curve_choice]
        })
    s_df = pd.DataFrame(spreads)

    c_mat, c_chart = st.columns([1, 2])
    
    with c_mat:
        st.dataframe(
            s_df[['Pair', 'Spread']],
            column_config={
                "Spread": st.column_config.NumberColumn("Price Diff", format="%.3f")
            },
            hide_index=True,
            height=600,
            use_container_width=True
        )
        
    with c_chart:
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            x=s_df['Pair'], y=s_df['Spread'],
            marker_color=s_df['Spread'].apply(lambda x: '#FF4B4B' if x < 0 else '#00FF00')
        ))
        fig_s.update_layout(
            title=f"{curve_choice} Roll Cost",
            template="plotly_dark",
            yaxis_title="Price Spread",
            height=600
        )
        st.plotly_chart(fig_s, use_container_width=True)
