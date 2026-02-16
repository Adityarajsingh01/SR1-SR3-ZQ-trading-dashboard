import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
import calendar

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="STIR Master Pro",
    layout="wide",
    page_icon="📉",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. FINANCIAL CONSTANTS & UTILS
# -----------------------------------------------------------------------------
# Ticker generation helpers
MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

# DV01 ($ value of 1bp move per contract)
DV01_MAP = {
    "ZQ": 41.67,
    "SR1": 41.67,
    "SR3": 25.00
}

def get_contract_expiry(start_date, months_forward):
    """Calculates future contract date."""
    target_date = start_date + timedelta(days=30 * months_forward)
    return target_date

def generate_ticker(product, dt):
    """Generates ZQM6 format ticker."""
    mc = MONTH_CODES[dt.month]
    yc = str(dt.year)[-1]
    return f"{product}{mc}{yc}"

# -----------------------------------------------------------------------------
# 3. MARKET DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def get_market_data(base_rate, shock_bps):
    """
    Generates a synthetic yield curve and prices contracts.
    """
    today = date.today()
    curve_data = []
    
    # Base Curve Shape (Inverted: -3bps per month)
    slope = -0.03 
    
    for i in range(24): # 2 Years out
        f_date = today + timedelta(days=30*i)
        
        # 1. Rate Construction
        # Natural rate + Slope + User Shock
        raw_rate = base_rate + (i * slope) + (shock_bps / 100.0)
        
        # Product Spreads
        effr_rate = raw_rate
        sofr_1m_rate = raw_rate - 0.05
        sofr_3m_rate = raw_rate - 0.03 # 3M usually higher than 1M in normal times, but tracking closely here
        
        # 2. Price Construction (100 - Rate)
        # We round to 3 decimals to mimic market precision
        row = {
            "Date": f_date,
            "Month_Code": f"{MONTH_CODES[f_date.month]}{str(f_date.year)[-1]}",
            "Month_Str": f_date.strftime("%b %y"),
            
            # ZQ (Fed Funds)
            "ZQ_Rate": round(effr_rate, 3),
            "ZQ_Price": round(100 - effr_rate, 3),
            
            # SR1 (1M SOFR)
            "SR1_Rate": round(sofr_1m_rate, 3),
            "SR1_Price": round(100 - sofr_1m_rate, 3),
            
            # SR3 (3M SOFR)
            "SR3_Rate": round(sofr_3m_rate, 3),
            "SR3_Price": round(100 - sofr_3m_rate, 3),
        }
        
        # Tickers for lookup
        row["ZQ_Ticker"] = f"ZQ{row['Month_Code']}"
        row["SR1_Ticker"] = f"SR1{row['Month_Code']}"
        row["SR3_Ticker"] = f"SR3{row['Month_Code']}"
        
        curve_data.append(row)
        
    return pd.DataFrame(curve_data)

# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🏛️ Desk Controls")
view_mode = st.sidebar.radio("Workstation", ["Market Overview", "Strategy Lab", "Spread Matrix"])

st.sidebar.divider()
st.sidebar.markdown("### 📊 Curve Assumptions")
base_effr = st.sidebar.number_input("Base EFFR (%)", 0.00, 20.00, 3.64, 0.01)
curve_shock = st.sidebar.slider("Parallel Shift (bps)", -50, 50, 0, help="Shifts the entire yield curve up/down.")

# Load Data
df = get_market_data(base_effr, curve_shock)

# -----------------------------------------------------------------------------
# 5. DASHBOARD HEADER (TAPE)
# -----------------------------------------------------------------------------
st.title("STIR Master Pro")
st.markdown(f"**Pricing Date:** {date.today().strftime('%Y-%m-%d')} | **Mode:** Synthetic Live")

# Tape Selection
tape_asset = st.sidebar.selectbox("Tape Asset", ["ZQ", "SR1", "SR3"], index=0)

# Render Tape (Top 6 Contracts)
tape_cols = st.columns(6)
for i in range(min(6, len(df))):
    row = df.iloc[i]
    ticker = row[f"{tape_asset}_Ticker"]
    price = row[f"{tape_asset}_Price"]
    rate = row[f"{tape_asset}_Rate"]
    
    # Calculate day-over-day change (simulated as spread from base)
    change = rate - base_effr
    
    with tape_cols[i]:
        st.metric(
            label=ticker,
            value=f"{price:.3f}",
            delta=f"{change:.2f}%",
            delta_color="inverse" # Red if rate higher (price lower)
        )

st.divider()

# -----------------------------------------------------------------------------
# VIEW 1: MARKET OVERVIEW
# -----------------------------------------------------------------------------
if view_mode == "Market Overview":
    col_chart, col_quotes = st.columns([2, 1])
    
    with col_chart:
        st.subheader("Forward Term Structure")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=df['Month_Str'], y=df['ZQ_Rate'], name='ZQ (FF)', line=dict(color='#00F0FF', width=3)))
        fig.add_trace(go.Scatter(x=df['Month_Str'], y=df['SR1_Rate'], name='SR1 (1M)', line=dict(color='#FFE800', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Month_Str'], y=df['SR3_Rate'], name='SR3 (3M)', line=dict(color='#FF8C00', width=2, dash='dash')))
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Contract Month",
            yaxis_title="Implied Rate (%)",
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_quotes:
        st.subheader("Settlement Board")
        # Clean Table
        display_df = df[['Month_Str', 'ZQ_Price', 'SR3_Price']].copy()
        display_df['Basis (bps)'] = (display_df['ZQ_Price'] - display_df['SR3_Price']) * 100
        
        st.dataframe(
            display_df,
            column_config={
                "Month_Str": "Month",
                "ZQ_Price": st.column_config.NumberColumn("ZQ", format="%.3f"),
                "SR3_Price": st.column_config.NumberColumn("SR3", format="%.3f"),
                "Basis (bps)": st.column_config.NumberColumn("Basis", format="%.1f")
            },
            hide_index=True,
            use_container_width=True,
            height=500
        )

# -----------------------------------------------------------------------------
# VIEW 2: STRATEGY LAB (REBUILT)
# -----------------------------------------------------------------------------
elif view_mode == "Strategy Lab":
    st.subheader("🛠️ Strategy Constructor")
    
    # 1. CONFIGURATION
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        strat_type = st.selectbox("Structure Type", ["Calendar Spread", "Butterfly (Fly)", "Condor", "Custom"])
    with c2:
        product = st.selectbox("Product Class", ["ZQ", "SR3"])
    with c3:
        lot_size = st.number_input("Size (Lots)", 100, 10000, 100, step=100)

    # 2. LEG BUILDER LOGIC
    legs = [] # List of dicts: {qty, ticker, price}
    
    contract_opts = df[f"{product}_Ticker"].tolist()
    prices_map = dict(zip(df[f"{product}_Ticker"], df[f"{product}_Price"]))
    
    if strat_type == "Calendar Spread":
        cols = st.columns(2)
        front = cols[0].selectbox("Front Leg", contract_opts, index=0)
        back = cols[1].selectbox("Back Leg", contract_opts, index=1)
        
        legs.append({"Side": "BUY", "Qty": 1, "Ticker": front, "Price": prices_map[front]})
        legs.append({"Side": "SELL", "Qty": -1, "Ticker": back, "Price": prices_map[back]})
        
    elif strat_type == "Butterfly (Fly)":
        cols = st.columns(3)
        wing1 = cols[0].selectbox("Wing 1", contract_opts, index=0)
        belly = cols[1].selectbox("Belly", contract_opts, index=3)
        wing2 = cols[2].selectbox("Wing 2", contract_opts, index=6)
        
        legs.append({"Side": "BUY", "Qty": 1, "Ticker": wing1, "Price": prices_map[wing1]})
        legs.append({"Side": "SELL", "Qty": -2, "Ticker": belly, "Price": prices_map[belly]})
        legs.append({"Side": "BUY", "Qty": 1, "Ticker": wing2, "Price": prices_map[wing2]})

    elif strat_type == "Condor":
        cols = st.columns(4)
        l1 = cols[0].selectbox("Leg 1", contract_opts, index=0)
        l2 = cols[1].selectbox("Leg 2", contract_opts, index=1)
        l3 = cols[2].selectbox("Leg 3", contract_opts, index=2)
        l4 = cols[3].selectbox("Leg 4", contract_opts, index=3)
        
        legs.append({"Side": "BUY", "Qty": 1, "Ticker": l1, "Price": prices_map[l1]})
        legs.append({"Side": "SELL", "Qty": -1, "Ticker": l2, "Price": prices_map[l2]})
        legs.append({"Side": "SELL", "Qty": -1, "Ticker": l3, "Price": prices_map[l3]})
        legs.append({"Side": "BUY", "Qty": 1, "Ticker": l4, "Price": prices_map[l4]})

    # 3. TICKET GENERATION
    st.divider()
    t_col, r_col = st.columns([1, 2])
    
    with t_col:
        st.markdown("##### 🎫 Deal Ticket")
        
        # Calculate Net Price
        net_price = 0
        ticket_data = []
        
        for leg in legs:
            leg_sign = 1 if leg['Qty'] > 0 else -1
            # For pricing display, we usually show spread price.
            # Spread Price = Sum(Price * Weight). 
            # E.g. Fly = P1 - 2P2 + P3
            weight = leg['Qty'] / abs(legs[0]['Qty']) # Normalize to first leg 
            net_price += (leg['Price'] * weight)
            
            ticket_data.append({
                "Side": "BUY" if leg['Qty'] > 0 else "SELL",
                "Qty": abs(leg['Qty']) * lot_size,
                "Ticker": leg['Ticker'],
                "Price": leg['Price']
            })
            
        st.dataframe(pd.DataFrame(ticket_data), hide_index=True, use_container_width=True)
        
        # Highlight Net Price
        st.metric("Package Price", f"{net_price:.3f}")
        
        # Net DV01 Calculation
        net_dv01 = 0
        current_dv01 = DV01_MAP.get(product[:2] if "SR" in product else "ZQ", 25)
        for leg in legs:
            # DV01 is risk. Long position loses money if rates go UP (Price Down).
            # So Long DV01 is negative PnL correlation to rate up?
            # Standard convention: DV01 is PnL for 1bp decline in rates (Price Up).
            # Long 1 Lot: +$25. Short 1 Lot: -$25.
            net_dv01 += (leg['Qty'] * lot_size * current_dv01)
            
        risk_color = "red" if abs(net_dv01) > 500 else "green"
        st.markdown(f"**Net DV01:** <span style='color:{risk_color}'>${net_dv01:,.2f}</span> / bp", unsafe_allow_html=True)
        st.caption("Delta Neutral if close to $0")

    with r_col:
        st.markdown("##### ⚠️ Risk Simulation")
        
        risk_mode = st.radio("Simulation Mode", ["Parallel Shift", "Curve Twist (Slope)"], horizontal=True)
        
        # Simulation Logic
        sim_range = np.linspace(-25, 25, 51) # -25bps to +25bps
        pnl_results = []
        
        for move in sim_range:
            sim_pnl = 0
            for i, leg in enumerate(legs):
                # Determine rate shift for this leg based on mode
                if risk_mode == "Parallel Shift":
                    # All rates move by 'move'
                    shift = move
                else:
                    # Curve Twist: Front leg anchored, back legs move more/less
                    # Simple steepener logic: Leg index * move
                    # e.g. Leg 0 moves 0, Leg 1 moves 1*move, Leg 2 moves 2*move
                    shift = move * i 
                
                # Price Change approx = -1 * Shift(bps) / 100
                # But careful: Price is 100-Rate. Rate up (+Shift) -> Price Down.
                # Price Delta = -Shift/100
                # PnL = Qty * LotSize * DV01_Per_BP * Shift
                # actually: PnL = Qty * PriceDelta * ValueOfPoint
                
                # Easier: PnL = Qty * LotSize * (-Shift * current_dv01) ?? No
                # DV01 is value of 1bp.
                # If Rate +1bp. Long Position PnL = -DV01.
                # PnL = Qty(signed) * LotSize * (Shift * -1) * (current_dv01 / 100? No DV01 is dollar value)
                
                # Correct: PnL = SignedQty * LotSize * (Shift * -1) * (current_dv01/LotSize?? No DV01 is per contract)
                # DV01 is usually defined as PnL per 1bp move.
                # PnL = SignedQty * LotSize * DV01 * (Shift * -1) ??
                
                # Let's simplify: 
                # Long 1 lot ZQ. Rate +1bp. Price -0.01. PnL = -$41.67.
                # Formula: Signedimport streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
import calendar

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="STIR Master Pro",
    layout="wide",
    page_icon="📉",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. FINANCIAL CONSTANTS & UTILS
# -----------------------------------------------------------------------------
# Ticker generation helpers
MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

# DV01 ($ value of 1bp move per contract)
DV01_MAP = {
    "ZQ": 41.67,
    "SR1": 41.67,
    "SR3": 25.00
}

def get_contract_expiry(start_date, months_forward):
    """Calculates future contract date."""
    target_date = start_date + timedelta(days=30 * months_forward)
    return target_date

def generate_ticker(product, dt):
    """Generates ZQM6 format ticker."""
    mc = MONTH_CODES[dt.month]
    yc = str(dt.year)[-1]
    return f"{product}{mc}{yc}"

# -----------------------------------------------------------------------------
# 3. MARKET DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def get_market_data(base_rate, shock_bps):
    """
    Generates a synthetic yield curve and prices contracts.
    """
    today = date.today()
    curve_data = []
    
    # Base Curve Shape (Inverted: -3bps per month)
    slope = -0.03 
    
    for i in range(24): # 2 Years out
        f_date = today + timedelta(days=30*i)
        
        # 1. Rate Construction
        # Natural rate + Slope + User Shock
        raw_rate = base_rate + (i * slope) + (shock_bps / 100.0)
        
        # Product Spreads
        effr_rate = raw_rate
        sofr_1m_rate = raw_rate - 0.05
        sofr_3m_rate = raw_rate - 0.03 # 3M usually higher than 1M in normal times, but tracking closely here
        
        # 2. Price Construction (100 - Rate)
        # We round to 3 decimals to mimic market precision
        row = {
            "Date": f_date,
            "Month_Code": f"{MONTH_CODES[f_date.month]}{str(f_date.year)[-1]}",
            "Month_Str": f_date.strftime("%b %y"),
            
            # ZQ (Fed Funds)
            "ZQ_Rate": round(effr_rate, 3),
            "ZQ_Price": round(100 - effr_rate, 3),
            
            # SR1 (1M SOFR)
            "SR1_Rate": round(sofr_1m_rate, 3),
            "SR1_Price": round(100 - sofr_1m_rate, 3),
            
            # SR3 (3M SOFR)
            "SR3_Rate": round(sofr_3m_rate, 3),
            "SR3_Price": round(100 - sofr_3m_rate, 3),
        }
        
        # Tickers for lookup
        row["ZQ_Ticker"] = f"ZQ{row['Month_Code']}"
        row["SR1_Ticker"] = f"SR1{row['Month_Code']}"
        row["SR3_Ticker"] = f"SR3{row['Month_Code']}"
        
        curve_data.append(row)
        
    return pd.DataFrame(curve_data)

# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🏛️ Desk Controls")
view_mode = st.sidebar.radio("Workstation", ["Market Overview", "Strategy Lab", "Spread Matrix"])

st.sidebar.divider()
st.sidebar.markdown("### 📊 Curve Assumptions")
base_effr = st.sidebar.number_input("Base EFFR (%)", 0.00, 20.00, 3.64, 0.01)
curve_shock = st.sidebar.slider("Parallel Shift (bps)", -50, 50, 0, help="Shifts the entire yield curve up/down.")

# Load Data
df = get_market_data(base_effr, curve_shock)

# -----------------------------------------------------------------------------
# 5. DASHBOARD HEADER (TAPE)
# -----------------------------------------------------------------------------
st.title("STIR Master Pro")
st.markdown(f"**Pricing Date:** {date.today().strftime('%Y-%m-%d')} | **Mode:** Synthetic Live")

# Tape Selection
tape_asset = st.sidebar.selectbox("Tape Asset", ["ZQ", "SR1", "SR3"], index=0)

# Render Tape (Top 6 Contracts)
tape_cols = st.columns(6)
for i in range(min(6, len(df))):
    row = df.iloc[i]
    ticker = row[f"{tape_asset}_Ticker"]
    price = row[f"{tape_asset}_Price"]
    rate = row[f"{tape_asset}_Rate"]
    
    # Calculate day-over-day change (simulated as spread from base)
    change = rate - base_effr
    
    with tape_cols[i]:
        st.metric(
            label=ticker,
            value=f"{price:.3f}",
            delta=f"{change:.2f}%",
            delta_color="inverse" # Red if rate higher (price lower)
        )

st.divider()

# -----------------------------------------------------------------------------
# VIEW 1: MARKET OVERVIEW
# -----------------------------------------------------------------------------
if view_mode == "Market Overview":
    col_chart, col_quotes = st.columns([2, 1])
    
    with col_chart:
        st.subheader("Forward Term Structure")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=df['Month_Str'], y=df['ZQ_Rate'], name='ZQ (FF)', line=dict(color='#00F0FF', width=3)))
        fig.add_trace(go.Scatter(x=df['Month_Str'], y=df['SR1_Rate'], name='SR1 (1M)', line=dict(color='#FFE800', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Month_Str'], y=df['SR3_Rate'], name='SR3 (3M)', line=dict(color='#FF8C00', width=2, dash='dash')))
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Contract Month",
            yaxis_title="Implied Rate (%)",
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_quotes:
        st.subheader("Settlement Board")
        # Clean Table
        display_df = df[['Month_Str', 'ZQ_Price', 'SR3_Price']].copy()
        display_df['Basis (bps)'] = (display_df['ZQ_Price'] - display_df['SR3_Price']) * 100
        
        st.dataframe(
            display_df,
            column_config={
                "Month_Str": "Month",
                "ZQ_Price": st.column_config.NumberColumn("ZQ", format="%.3f"),
                "SR3_Price": st.column_config.NumberColumn("SR3", format="%.3f"),
                "Basis (bps)": st.column_config.NumberColumn("Basis", format="%.1f")
            },
            hide_index=True,
            use_container_width=True,
            height=500
        )

# -----------------------------------------------------------------------------
# VIEW 2: STRATEGY LAB (REBUILT)
# -----------------------------------------------------------------------------
elif view_mode == "Strategy Lab":
    st.subheader("🛠️ Strategy Constructor")
    
    # 1. CONFIGURATION
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        strat_type = st.selectbox("Structure Type", ["Calendar Spread", "Butterfly (Fly)", "Condor", "Custom"])
    with c2:
        product = st.selectbox("Product Class", ["ZQ", "SR3"])
    with c3:
        lot_size = st.number_input("Size (Lots)", 100, 10000, 100, step=100)

    # 2. LEG BUILDER LOGIC
    legs = [] # List of dicts: {qty, ticker, price}
    
    contract_opts = df[f"{product}_Ticker"].tolist()
    prices_map = dict(zip(df[f"{product}_Ticker"], df[f"{product}_Price"]))
    
    if strat_type == "Calendar Spread":
        cols = st.columns(2)
        front = cols[0].selectbox("Front Leg", contract_opts, index=0)
        back = cols[1].selectbox("Back Leg", contract_opts, index=1)
        
        legs.append({"Side": "BUY", "Qty": 1, "Ticker": front, "Price": prices_map[front]})
        legs.append({"Side": "SELL", "Qty": -1, "Ticker": back, "Price": prices_map[back]})
        
    elif strat_type == "Butterfly (Fly)":
        cols = st.columns(3)
        wing1 = cols[0].selectbox("Wing 1", contract_opts, index=0)
        belly = cols[1].selectbox("Belly", contract_opts, index=3)
        wing2 = cols[2].selectbox("Wing 2", contract_opts, index=6)
        
        legs.append({"Side": "BUY", "Qty": 1, "Ticker": wing1, "Price": prices_map[wing1]})
        legs.append({"Side": "SELL", "Qty": -2, "Ticker": belly, "Price": prices_map[belly]})
        legs.append({"Side": "BUY", "Qty": 1, "Ticker": wing2, "Price": prices_map[wing2]})

    elif strat_type == "Condor":
        cols = st.columns(4)
        l1 = cols[0].selectbox("Leg 1", contract_opts, index=0)
        l2 = cols[1].selectbox("Leg 2", contract_opts, index=1)
        l3 = cols[2].selectbox("Leg 3", contract_opts, index=2)
        l4 = cols[3].selectbox("Leg 4", contract_opts, index=3)
        
        legs.append({"Side": "BUY", "Qty": 1, "Ticker": l1, "Price": prices_map[l1]})
        legs.append({"Side": "SELL", "Qty": -1, "Ticker": l2, "Price": prices_map[l2]})
        legs.append({"Side": "SELL", "Qty": -1, "Ticker": l3, "Price": prices_map[l3]})
        legs.append({"Side": "BUY", "Qty": 1, "Ticker": l4, "Price": prices_map[l4]})

    # 3. TICKET GENERATION
    st.divider()
    t_col, r_col = st.columns([1, 2])
    
    with t_col:
        st.markdown("##### 🎫 Deal Ticket")
        
        # Calculate Net Price
        net_price = 0
        ticket_data = []
        
        for leg in legs:
            leg_sign = 1 if leg['Qty'] > 0 else -1
            # For pricing display, we usually show spread price.
            # Spread Price = Sum(Price * Weight). 
            # E.g. Fly = P1 - 2P2 + P3
            weight = leg['Qty'] / abs(legs[0]['Qty']) # Normalize to first leg 
            net_price += (leg['Price'] * weight)
            
            ticket_data.append({
                "Side": "BUY" if leg['Qty'] > 0 else "SELL",
                "Qty": abs(leg['Qty']) * lot_size,
                "Ticker": leg['Ticker'],
                "Price": leg['Price']
            })
            
        st.dataframe(pd.DataFrame(ticket_data), hide_index=True, use_container_width=True)
        
        # Highlight Net Price
        st.metric("Package Price", f"{net_price:.3f}")
        
        # Net DV01 Calculation
        net_dv01 = 0
        current_dv01 = DV01_MAP.get(product[:2] if "SR" in product else "ZQ", 25)
        for leg in legs:
            # DV01 is risk. Long position loses money if rates go UP (Price Down).
            # So Long DV01 is negative PnL correlation to rate up?
            # Standard convention: DV01 is PnL for 1bp decline in rates (Price Up).
            # Long 1 Lot: +$25. Short 1 Lot: -$25.
            net_dv01 += (leg['Qty'] * lot_size * current_dv01)
            
        risk_color = "red" if abs(net_dv01) > 500 else "green"
        st.markdown(f"**Net DV01:** <span style='color:{risk_color}'>${net_dv01:,.2f}</span> / bp", unsafe_allow_html=True)
        st.caption("Delta Neutral if close to $0")

    with r_col:
        st.markdown("##### ⚠️ Risk Simulation")
        
        risk_mode = st.radio("Simulation Mode", ["Parallel Shift", "Curve Twist (Slope)"], horizontal=True)
        
        # Simulation Logic
        sim_range = np.linspace(-25, 25, 51) # -25bps to +25bps
        pnl_results = []
        
        for move in sim_range:
            sim_pnl = 0
            for i, leg in enumerate(legs):
                # Determine rate shift for this leg based on mode
                if risk_mode == "Parallel Shift":
                    # All rates move by 'move'
                    shift = move
                else:
                    # Curve Twist: Front leg anchored, back legs move more/less
                    # Simple steepener logic: Leg index * move
                    # e.g. Leg 0 moves 0, Leg 1 moves 1*move, Leg 2 moves 2*move
                    shift = move * i 
                
                # Price Change approx = -1 * Shift(bps) / 100
                # But careful: Price is 100-Rate. Rate up (+Shift) -> Price Down.
                # Price Delta = -Shift/100
                # PnL = Qty * LotSize * DV01_Per_BP * Shift
                # actually: PnL = Qty * PriceDelta * ValueOfPoint
                
                # Easier: PnL = Qty * LotSize * (-Shift * current_dv01) ?? No
                # DV01 is value of 1bp.
                # If Rate +1bp. Long Position PnL = -DV01.
                # PnL = Qty(signed) * LotSize * (Shift * -1) * (current_dv01 / 100? No DV01 is dollar value)
                
                # Correct: PnL = SignedQty * LotSize * (Shift * -1) * (current_dv01/LotSize?? No DV01 is per contract)
                # DV01 is usually defined as PnL per 1bp move.
                # PnL = SignedQty * LotSize * DV01 * (Shift * -1) ??
                
                # Let's simplify: 
                # Long 1 lot ZQ. Rate +1bp. Price -0.01. PnL = -$41.67.
                # Formula: Signed
