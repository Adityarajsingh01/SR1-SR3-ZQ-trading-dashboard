import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
import calendar

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="STIR Master Pro", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Projected FOMC Dates for 2026 (Standard Cycle: Jan, Mar, May, Jun, Jul, Sep, Nov, Dec)
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

# DV01 Constants ($ value per 1bp move per contract)
DV01_ZQ = 41.67
DV01_SR3 = 25.00

# -----------------------------------------------------------------------------
# 2. QUANT LOGIC & CALCULATIONS
# -----------------------------------------------------------------------------

def get_days_in_month(dt):
    """Returns number of days in the month of the given date."""
    return calendar.monthrange(dt.year, dt.month)[1]

def get_fomc_impact_ratio(contract_date):
    """
    Calculates the weight of the NEW rate vs OLD rate for a ZQ contract
    based on FOMC meeting date within that month.
    
    Logic: ZQ is Arithmetic Average of EFFR.
    If meeting is on day 20, we have 19 days of Old Rate, and (Total - 19) days of New Rate.
    """
    # Find if there is a meeting in this contract's month
    meeting = next((m for m in FOMC_MEETINGS if m.year == contract_date.year and m.month == contract_date.month), None)
    
    if not meeting:
        # If no meeting, it trades on the rate established by the LAST meeting
        # For simplicity in this demo, we return 1.0 (Old Rate)
        return (1.0, 0.0) 
    
    days_in_month = get_days_in_month(contract_date)
    
    # Days priced at OLD rate (before meeting)
    # Usually effective date is Meeting Date + 1 or same day depending on announcement timing.
    # We assume rate change is effective next day for conservative pricing.
    days_before = meeting.day 
    days_after = days_in_month - days_before
    
    return (days_before / days_in_month, days_after / days_in_month)

# -----------------------------------------------------------------------------
# 3. DATA INGESTION (Robust Engine)
# -----------------------------------------------------------------------------

@st.cache_data
def fetch_stir_data(base_effr, shock_bps=0):
    """
    Generates the STIR curve. 
    Since free APIs (Yahoo) rarely support granular futures chains (ZQM6 vs ZQU6) reliable enough for 
    demoing math, we generate a 'Pro-Grade Synthetic Curve' that reacts to your inputs.
    """
    today = date.today()
    contracts = []
    
    # Curve Shape parameters (Simulating an inverted curve common in high rate environments)
    # Front end is higher (tight policy), back end is lower (cuts priced in)
    curve_slope = -0.04  # -4 bps per month
    
    for i in range(18): # 18 Month view
        future_date = today + timedelta(days=30*i)
        
        # Skip if date is in the past
        if future_date < today:
            continue
            
        month_code = MONTH_CODES[future_date.month]
        year_code = str(future_date.year)[-1] # '6' for 2026
        ticker_suffix = f"{month_code}{year_code}"
        
        # 1. Calculate Base Rate for this month (Inverted Curve Logic)
        # We add some random noise to simulate market inefficiencies
        market_rate = base_effr + (i * curve_slope) + (np.random.normal(0, 0.015))
        
        # Apply the user's "Shock" scenario if any
        market_rate += (shock_bps / 100)

        # 2. ZQ Pricing (Arithmetic)
        # ZQ prices exactly to 100 - Rate
        zq_price = 100 - market_rate
        
        # 3. SR3 Pricing (Geometric + Credit Spread)
        # SOFR usually trades slightly below EFFR (e.g. -5 to -10 bps)
        # SR3 has convexity (it's better to be long SR3 than ZQ in volatile cuts)
        sr3_rate = market_rate - 0.08 # Spread to EFFR
        sr3_price = 100 - sr3_rate

        contracts.append({
            "Expiry": future_date.strftime("%b %Y"),
            "MonthCode": ticker_suffix,
            "ZQ_Ticker": f"ZQ{ticker_suffix}",
            "ZQ_Price": round(zq_price, 3),
            "ZQ_Rate": round(100 - zq_price, 3),
            "SR3_Ticker": f"SR3{ticker_suffix}",
            "SR3_Price": round(sr3_price, 3),
            "SR3_Rate": round(100 - sr3_price, 3),
            "Date_Obj": future_date
        })
        
    return pd.DataFrame(contracts)

# -----------------------------------------------------------------------------
# 4. UI & STATE
# -----------------------------------------------------------------------------

# Initialize Session State
if 'notional' not in st.session_state:
    st.session_state['notional'] = 100

# --- Sidebar ---
st.sidebar.title("STIR Master Pro 📊")
view_mode = st.sidebar.radio("Module", ["Curve Analyzer", "Strategy Lab", "Arb Scanner"])

st.sidebar.markdown("---")
st.sidebar.header("Global Parameters")
base_rate_input = st.sidebar.number_input("Current EFFR (%)", value=5.33, step=0.01)
curve_shock = st.sidebar.slider("Market Shock Scenario (bps)", -100, 100, 0)

# Load Data
df = fetch_stir_data(base_rate_input, curve_shock)

# --- Main Dashboard ---

st.title(f"STIR Trading Desk")
st.caption(f"Pricing Date: {date.today().strftime('%Y-%m-%d')} | Data Mode: Synthetic Real-Time")

# Ticker Tape (Top)
tape_cols = st.columns(6)
for i in range(min(6, len(df))):
    row = df.iloc[i]
    with tape_cols[i]:
        st.metric(
            label=row['ZQ_Ticker'], 
            value=f"{row['ZQ_Price']:.3f}", 
            delta=f"{(row['ZQ_Rate'] - base_rate_input):.2f}%"
        )

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. MODULE: CURVE ANALYZER
# -----------------------------------------------------------------------------
if view_mode == "Curve Analyzer":
    col_chart, col_data = st.columns([2, 1])
    
    with col_chart:
        st.subheader("Forward Term Structure")
        
        fig = go.Figure()
        
        # ZQ Curve
        fig.add_trace(go.Scatter(
            x=df['Expiry'], y=df['ZQ_Rate'], 
            mode='lines+markers', name='ZQ (Fed Funds)',
            line=dict(color='#00F0FF', width=3)
        ))
        
        # SR3 Curve
        fig.add_trace(go.Scatter(
            x=df['Expiry'], y=df['SR3_Rate'], 
            mode='lines+markers', name='SR3 (SOFR)',
            line=dict(color='#FFA500', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title="Implied Rates (%)",
            xaxis_title="Contract Month",
            yaxis_title="Rate",
            template="plotly_dark",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_data:
        st.subheader("Curve Data")
        st.dataframe(
            df[['Expiry', 'ZQ_Price', 'SR3_Price', 'ZQ_Rate']].style.format("{:.3f}"),
            height=500
        )

# -----------------------------------------------------------------------------
# 6. MODULE: STRATEGY LAB (FLYS, SPREADS)
# -----------------------------------------------------------------------------
elif view_mode == "Strategy Lab":
    st.subheader("Strategy Constructor")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("### Build Structure")
        strat_type = st.selectbox("Type", ["Calendar Spread", "Butterfly (Fly)", "Condor"])
        instrument = st.selectbox("Instrument", ["ZQ", "SR3"])
        
        # Dynamic Contract Selection
        tickers = df[f'{instrument}_Ticker'].tolist()
        
        legs = []
        
        if strat_type == "Calendar Spread":
            leg1 = st.selectbox("Front Leg (Buy)", tickers, index=0)
            leg2 = st.selectbox("Back Leg (Sell)", tickers, index=2)
            
            p1 = df.loc[df[f'{instrument}_Ticker'] == leg1, f'{instrument}_Price'].values[0]
            p2 = df.loc[df[f'{instrument}_Ticker'] == leg2, f'{instrument}_Price'].values[0]
            
            legs = [(1, p1, leg1), (-1, p2, leg2)]
            
        elif strat_type == "Butterfly (Fly)":
            # Fly = Long Wing, Short 2x Belly, Long Wing
            belly = st.selectbox("Belly (Center)", tickers, index=2)
            width = st.slider("Wing Width (Months)", 1, 3, 1)
            
            try:
                idx = tickers.index(belly)
                wing1 = tickers[idx - width]
                wing2 = tickers[idx + width]
                
                p_belly = df.loc[df[f'{instrument}_Ticker'] == belly, f'{instrument}_Price'].values[0]
                p_w1 = df.loc[df[f'{instrument}_Ticker'] == wing1, f'{instrument}_Price'].values[0]
                p_w2 = df.loc[df[f'{instrument}_Ticker'] == wing2, f'{instrument}_Price'].values[0]
                
                legs = [(1, p_w1, wing1), (-2, p_belly, belly), (1, p_w2, wing2)]
                
            except IndexError:
                st.error("Structure out of bounds. Move Belly to center of curve.")

        # Calculate Price
        net_price = sum([q * p for q, p, n in legs])
        st.metric(f"{strat_type} Price", f"{net_price:.3f}")
        
        # Position Sizing
        st.markdown("### Sizing")
        qty = st.number_input("Quantity (Lots)", value=100)
        dv01_val = DV01_ZQ if instrument == "ZQ" else DV01_SR3
        
        # Net DV01
        # Sum of abs(qty * leg_qty) is margin risk, but net DV01 is directional risk
        net_dv01 = sum([q * dv01_val for q, p, n in legs]) * qty
        st.write(f"**Net DV01:** ${net_dv01:.2f} / bp")

    with c2:
        st.markdown("### Payoff Analysis")
        
        # Visualize "Belly" Cheapness
        if strat_type == "Butterfly (Fly)":
            # Extract just the relevant segment of the curve to visualize the "kink"
            relevant_indices = [tickers.index(l[2]) for l in legs]
            relevant_indices.sort()
            start_i, end_i = relevant_indices[0], relevant_indices[-1]
            
            segment = df.iloc[start_i:end_i+1]
            
            fig_fly = go.Figure()
            fig_fly.add_trace(go.Scatter(
                x=segment['Expiry'], y=segment[f'{instrument}_Price'],
                mode='lines+markers', name='Market Curve',
                line=dict(color='yellow')
            ))
            
            # Draw the Linear Interpolation (The "Fair" Value if no convexity)
            fig_fly.add_trace(go.Scatter(
                x=[segment.iloc[0]['Expiry'], segment.iloc[-1]['Expiry']],
                y=[segment.iloc[0][f'{instrument}_Price'], segment.iloc[-1][f'{instrument}_Price']],
                mode='lines', name='Linear Fair Value',
                line=dict(color='white', dash='dot')
            ))
            
            fig_fly.update_layout(title=f"Visualizing the Belly: {belly}", template="plotly_dark")
            st.plotly_chart(fig_fly, use_container_width=True)

        st.write("#### Leg Execution")
        exec_df = pd.DataFrame(legs, columns=["Ratio", "Ref Price", "Ticker"])
        exec_df['Total Lots'] = exec_df['Ratio'] * qty
        st.table(exec_df)

# -----------------------------------------------------------------------------
# 7. MODULE: ARB SCANNER
# -----------------------------------------------------------------------------
elif view_mode == "Arb Scanner":
    st.subheader("ZQ vs FOMC Meeting Dates")
    st.info("This tool strips the ZQ price to find the implied rate between meetings.")
    
    col_arb1, col_arb2 = st.columns(2)
    
    with col_arb1:
        arb_ticker = st.selectbox("Select ZQ Contract", df['ZQ_Ticker'].unique())
        row = df[df['ZQ_Ticker'] == arb_ticker].iloc[0]
        
        contract_date = row['Date_Obj']
        # Calculate weights
        w_old, w_new = get_fomc_impact_ratio(contract_date)
        
        st.metric("Current Market Price", f"{row['ZQ_Price']:.3f}")
        st.write(f"**FOMC Weighting Logic:**")
        st.write(f"- Days at Old Rate: {w_old*100:.1f}%")
        st.write(f"- Days at New Rate: {w_new*100:.1f}%")
        
        if w_new == 0:
            st.warning("No FOMC meeting scheduled in this contract month.")
    
    with col_arb2:
        st.write("### Theoretical Pricing")
        
        user_hike = st.select_slider("FOMC Move (bps)", options=[-50, -25, 0, 25, 50], value=-25)
        
        # Calculate what the price SHOULD be if that hike happens
        # Logic: (OldRate * w_old) + ((OldRate + Hike) * w_new)
        
        # We assume the "Old Rate" is the current EFFR input from sidebar
        old_rate = base_rate_input
        new_rate = old_rate + (user_hike/100)
        
        fair_rate = (old_rate * w_old) + (new_rate * w_new)
        fair_price = 100 - fair_rate
        
        st.metric(f"Fair Value (if {user_hike}bps)", f"{fair_price:.3f}", delta=f"{fair_price - row['ZQ_Price']:.3f}")
        
        st.caption("If Delta is POSITIVE (Green), Market is CHEAP (Buy). If NEGATIVE (Red), Market is RICH (Sell).")
