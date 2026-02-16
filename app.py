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

# precise CSS for compact metrics and cleaner tables
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem !important; 
        font-weight: 700;
        color: #E0E0E0;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
    }
    /* Compact Tables */
    div[data-testid="stDataFrame"] {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS & UTILS
# -----------------------------------------------------------------------------
FOMC_MEETINGS_2026 = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6), 
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 11, 4), date(2026, 12, 16)
]

MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

DV01_ZQ = 41.67
DV01_SR3 = 25.00

def get_days_in_month(dt):
    return calendar.monthrange(dt.year, dt.month)[1]

def get_fomc_weights(contract_date):
    """Returns (weight_old, weight_new) based on meeting day."""
    meeting = next((m for m in FOMC_MEETINGS_2026 if m.year == contract_date.year and m.month == contract_date.month), None)
    if not meeting: return (1.0, 0.0)
    
    dim = get_days_in_month(contract_date)
    # ZQ is arithmetic avg. If mtg is day 20, 19 days old, rest new.
    # Usually effective next day.
    d_old = meeting.day 
    return (d_old / dim, (dim - d_old) / dim)

# -----------------------------------------------------------------------------
# 3. DATA ENGINE (Synthetic Live)
# -----------------------------------------------------------------------------
@st.cache_data
def get_curve_data(base_rate, shock_bps):
    today = date.today()
    data = []
    
    # Inverted curve slope (-3bps/month) + random noise
    slope = -0.03
    
    for i in range(18):
        f_date = today + timedelta(days=30*i)
        if f_date < today: continue
        
        mc = MONTH_CODES[f_date.month]
        yc = str(f_date.year)[-1]
        ticker = f"{mc}{yc}"
        
        # 1. Theoretical Rate
        raw_rate = base_rate + (i * slope) 
        # Add "Market Noise" so curve isn't perfectly smooth (creates arb opps)
        noise = np.random.normal(0, 0.015) 
        final_rate = raw_rate + noise + (shock_bps/100)
        
        # 2. Prices
        zq_price = 100 - final_rate
        sr3_price = 100 - (final_rate - 0.12) # SOFR spread
        
        # 3. Arb Logic (Fair Value)
        # We assume "Fair Value" is the raw_rate without the noise
        w_old, w_new = get_fomc_weights(f_date)
        fair_rate = raw_rate # Simplified "Fair" model
        fair_price = 100 - fair_rate
        arb = fair_price - zq_price # + means Market is Cheap (Buy)

        data.append({
            "Expiry": f_date.strftime("%b %y"),
            "Code": ticker,
            "ZQ": round(zq_price, 3),
            "ZQ_Rate": round(100-zq_price, 3),
            "SR3": round(sr3_price, 3),
            "Arb_bps": round(arb * 100, 1),
            "Date": f_date
        })
        
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 4. UI LAYOUT
# -----------------------------------------------------------------------------

# Sidebar
st.sidebar.header("STIR Desk Controls")
mode = st.sidebar.radio("Workstation Mode", ["Market Overview", "Strategy Lab", "Spread Matrix"])

st.sidebar.divider()
st.sidebar.subheader("Curve Simulation")
base_effr = st.sidebar.number_input("Base EFFR (%)", 4.0, 6.0, 5.33, step=0.01)
curve_shock = st.sidebar.slider("Parallel Shock (bps)", -50, 50, 0)

# Load Data
df = get_curve_data(base_effr, curve_shock)

st.title(f"STIR Trading Dashboard")
st.caption(f"Pricing Date: {date.today()} | Simulation Mode: Active")

# --- Top Tape (First 6 Contracts) ---
cols = st.columns(6)
for i in range(min(6, len(df))):
    row = df.iloc[i]
    with cols[i]:
        color = "normal"
        if row['Arb_bps'] > 3.0: color = "off" # Streamlit delta color logic inverse?
        st.metric(
            label=f"ZQ{row['Code']}", 
            value=f"{row['ZQ']:.3f}", 
            delta=f"{row['Arb_bps']} bp",
            delta_color="normal" # Green = Positive Arb (Buy)
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
        fig.add_trace(go.Scatter(x=df['Expiry'], y=df['ZQ_Rate'], name='ZQ (Fed Funds)', line=dict(color='#00F0FF', width=3)))
        fig.add_trace(go.Scatter(x=df['Expiry'], y=100-df['SR3'], name='SR3 (SOFR)', line=dict(color='orange', dash='dash')))
        fig.update_layout(template="plotly_dark", height=450, xaxis_title="Contract", yaxis_title="Implied Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Live Quotes")
        # FIX: proper column config prevents crashes
        st.dataframe(
            df[['Expiry', 'ZQ', 'SR3', 'Arb_bps']], 
            column_config={
                "Expiry": "Mth",
                "ZQ": st.column_config.NumberColumn("ZQ Price", format="%.3f"),
                "SR3": st.column_config.NumberColumn("SR3 Price", format="%.3f"),
                "Arb_bps": st.column_config.NumberColumn("Fair Val (bps)", format="%.1f")
            },
            hide_index=True,
            use_container_width=True,
            height=450
        )

# -----------------------------------------------------------------------------
# VIEW: STRATEGY LAB (Fixed Payoff)
# -----------------------------------------------------------------------------
elif mode == "Strategy Lab":
    st.subheader("🧪 Strategy Constructor")
    
    col_builder, col_viz = st.columns([1, 2])
    
    with col_builder:
        st.info("Define Structure")
        strat_type = st.selectbox("Type", ["Butterfly (Fly)", "Calendar Spread"])
        root = st.selectbox("Product", ["ZQ", "SR3"])
        
        tickers = df['Code'].tolist()
        belly_code = st.selectbox("Center / Front Leg", tickers, index=2)
        
        legs = []
        try:
            idx = df[df['Code'] == belly_code].index[0]
            if strat_type == "Butterfly (Fly)":
                width = st.slider("Wing Width", 1, 3, 1)
                if idx - width >= 0 and idx + width < len(df):
                    w1 = df.iloc[idx - width]
                    belly = df.iloc[idx]
                    w2 = df.iloc[idx + width]
                    # Structure: +1 Wing, -2 Belly, +1 Wing
                    legs = [
                        (1, w1[root], w1['Code']), 
                        (-2, belly[root], belly['Code']), 
                        (1, w2[root], w2['Code'])
                    ]
            elif strat_type == "Calendar Spread":
                if idx + 1 < len(df):
                    front = df.iloc[idx]
                    back = df.iloc[idx+1]
                    legs = [(1, front[root], front['Code']), (-1, back[root], back['Code'])]
        except Exception as e:
            st.error(f"Invalid Structure: {e}")

        if legs:
            entry_price = sum([q*p for q,p,n in legs])
            st.metric("Net Package Price", f"{entry_price:.3f}")
            
            qty = st.number_input("Quantity (Lots)", 100, 5000, 100, step=100)
            dv01 = DV01_ZQ if root == "ZQ" else DV01_SR3
            
            st.markdown("---")
            st.write(" **Risk Analysis**")
            
            # Scenario Sliders
            st.write("Simulate Curve Moves:")
            sim_parallel = st.slider("Parallel Shift (bps)", -20, 20, 0)
            sim_curve = st.slider("Belly/Curve Twist (bps)", -10, 10, 0, help="Moves the belly relative to wings")
            
            # Calculate PnL based on sliders
            pnl_est = 0
            for q, p, n in legs:
                # Logic: PnL = Qty * PriceChange * DV01
                # PriceChange approx = -1 * (Shift + Twist)
                # Twist applies only to Belly leg for simplicity (or inversely to wings)
                
                shift = sim_parallel
                if abs(q) == 2: # Identifying belly by qty for demo
                    shift += sim_curve
                
                price_delta = -(shift / 100)
                pnl_est += (q * qty * price_delta * dv01)
                
            st.metric("Simulated PnL", f"${pnl_est:,.0f}", delta=f"{sim_curve} bps Twist")

    with col_viz:
        if legs:
            st.subheader("Payoff Profile")
            
            # 1. Generate PnL Curve for CURVATURE (Belly Shift)
            # Traders care about curvature for Flys, not parallel shift.
            twists = np.linspace(-15, 15, 31) # +/- 15 bps belly move
            pnl_curve = []
            
            for t in twists:
                # PnL of Fly = (BellyShift) * -2 * DV01 * Qty (Rough approx)
                # If belly yields go UP 1bp (Price DOWN 0.01), Short leg makes money? 
                # Short (-2) * PriceDown (-0.01) = +0.02 PnL.
                # So Higher Yields in Belly = Profit for Short Belly Fly? Yes.
                
                # Let's do exact calc:
                # Wing1 (0 shift), Belly (t shift), Wing2 (0 shift)
                daily_pnl = 0
                for q, p, n in legs:
                    s = t if abs(q) == 2 else 0 # Only shift belly
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
                xaxis_title="Belly Rate Move (bps) relative to Wings",
                yaxis_title="PnL ($)",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
            
            # 2. Execution Table
            st.write("#### Execution Ticket")
            ex_df = pd.DataFrame(legs, columns=["Qty", "Price", "Contract"])
            ex_df['Side'] = ex_df['Qty'].apply(lambda x: "BUY" if x > 0 else "SELL")
            ex_df['Abs Qty'] = ex_df['Qty'].abs() * qty
            st.dataframe(ex_df[['Side', 'Abs Qty', 'Contract', 'Price']], hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# VIEW: SPREAD MATRIX (Improved)
# -----------------------------------------------------------------------------
elif mode == "Spread Matrix":
    st.subheader("Calendar Spread Monitor (1-Month Rolls)")
    
    # Create Spread Data
    spreads = []
    for i in range(len(df)-1):
        front = df.iloc[i]
        back = df.iloc[i+1]
        val = front['ZQ'] - back['ZQ']
        spreads.append({
            "Pair": f"{front['Code']}/{back['Code']}",
            "Spread": val,
            "Type": "Inv" if val < 0 else "Steep",
            "Front": front['ZQ'],
            "Back": back['ZQ']
        })
    s_df = pd.DataFrame(spreads)

    c_mat, c_chart = st.columns([1, 2])
    
    with c_mat:
        # Heatmap style table
        st.dataframe(
            s_df[['Pair', 'Spread']],
            column_config={
                "Spread": st.column_config.NumberColumn(
                    "Price Diff", 
                    format="%.3f",
                )
            },
            hide_index=True,
            height=600,
            use_container_width=True
        )
        
    with c_chart:
        # Cleaner Bar Chart
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            x=s_df['Pair'], y=s_df['Spread'],
            marker_color=s_df['Spread'].apply(lambda x: '#FF4B4B' if x < 0 else '#00FF00')
        ))
        fig_s.update_layout(
            title="Roll Cost (Negative = Inverted/Carry)",
            template="plotly_dark",
            yaxis_title="Price Spread",
            height=600
        )
        st.plotly_chart(fig_s, use_container_width=True)
