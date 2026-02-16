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

# Clean CSS to fix "Black Box" issue and standardise font sizes
st.markdown("""
<style>
    /* Clean Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
    /* Compact Tables */
    [data-testid="stDataFrame"] { 
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

# DV01 Constants ($ per bp per contract)
DV01 = {"ZQ": 41.67, "SR1": 41.67, "SR3": 25.00}

def get_days_in_month(dt):
    return calendar.monthrange(dt.year, dt.month)[1]

# -----------------------------------------------------------------------------
# 3. DATA ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def get_curve_data(base_rate, shock_bps):
    today = date.today()
    data = []
    slope = -0.03
    
    for i in range(24): # Extended to 2 years for Condors
        f_date = today + timedelta(days=30*i)
        if f_date < today: continue
        
        mc = MONTH_CODES[f_date.month]
        yc = str(f_date.year)[-1]
        suffix = f"{mc}{yc}"
        
        # Rate Logic
        raw_rate = base_rate + (i * slope) 
        final_effr = raw_rate + (shock_bps/100) 
        sofr_rate = final_effr - 0.05
        
        # Pricing
        zq_price = 100 - final_effr
        sr1_price = 100 - sofr_rate
        sr3_price = 100 - (sofr_rate + 0.02)

        data.append({
            "Label": f"{suffix} ({f_date.strftime('%b')})",
            "Month": f_date.strftime("%b %y"),
            "Suffix": suffix,
            "Date": f_date,
            "ZQ": zq_price,
            "SR1": sr1_price,
            "SR3": sr3_price,
            "ZQ_Rate": 100-zq_price,
            "SR1_Rate": 100-sr1_price,
            "SR3_Rate": 100-sr3_price
        })
        
    return pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 4. SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.header("STIR Desk Controls")
mode = st.sidebar.radio("Workstation Mode", ["Market Overview", "Strategy Builder", "Spread Matrix"])

st.sidebar.divider()
st.sidebar.subheader("Curve Assumptions")
base_effr = st.sidebar.number_input("Base EFFR (%)", 0.0, 20.0, 3.64, 0.01)
curve_shock = st.sidebar.slider("Parallel Shift (bps)", -50, 50, 0)
tape_source = st.sidebar.selectbox("Tape Instrument", ["ZQ", "SR1", "SR3"])

df = get_curve_data(base_effr, curve_shock)

# -----------------------------------------------------------------------------
# 5. DASHBOARD HEADER (TAPE)
# -----------------------------------------------------------------------------
st.title("STIR Trading Dashboard")

# Dynamic Tape (First 6 contracts)
cols = st.columns(6)
for i in range(min(6, len(df))):
    row = df.iloc[i]
    val = row[tape_source]
    chg = row[f"{tape_source}_Rate"] - base_effr
    with cols[i]:
        st.metric(
            label=f"{tape_source}{row['Suffix']}", 
            value=f"{val:.3f}", 
            delta=f"{chg:.2f} sprd", 
            delta_color="inverse"
        )
st.divider()

# -----------------------------------------------------------------------------
# MODULE: MARKET OVERVIEW
# -----------------------------------------------------------------------------
if mode == "Market Overview":
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Forward Term Structure")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Month'], y=df['ZQ_Rate'], name='ZQ', line=dict(color='#00F0FF', width=3)))
        fig.add_trace(go.Scatter(x=df['Month'], y=df['SR1_Rate'], name='SR1', line=dict(color='#FFE800', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Month'], y=df['SR3_Rate'], name='SR3', line=dict(color='#FF8C00', width=2, dash='dash')))
        fig.update_layout(template="plotly_dark", height=500, xaxis_title="Contract", yaxis_title="Implied Rate (%)", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader("Live Prices")
        st.dataframe(
            df[['Suffix', 'ZQ', 'SR1', 'SR3']], 
            column_config={
                "ZQ": st.column_config.NumberColumn(format="%.3f"),
                "SR1": st.column_config.NumberColumn(format="%.3f"),
                "SR3": st.column_config.NumberColumn(format="%.3f")
            },
            hide_index=True, use_container_width=True, height=500
        )

# -----------------------------------------------------------------------------
# MODULE: STRATEGY BUILDER (PRO)
# -----------------------------------------------------------------------------
elif mode == "Strategy Builder":
    st.subheader("🛠️ Strategy Lab")
    
    # 1. Strategy Selector
    strat_col, prod_col, qty_col = st.columns([2, 1, 1])
    with strat_col:
        strat_type = st.selectbox("Strategy Type", ["Calendar Spread", "Butterfly (Fly)", "Condor", "Custom"])
    with prod_col:
        prod_root = st.selectbox("Product", ["ZQ", "SR1", "SR3"])
    with qty_col:
        base_qty = st.number_input("Base Size (Lots)", 100, 10000, 100, 100)

    # 2. Leg Generation Logic
    legs = [] # Stores (Qty, Price, Name)
    
    contract_list = df['Label'].tolist()
    
    # --- AUTO-POPULATE LEGS BASED ON STRATEGY ---
    if strat_type == "Calendar Spread":
        c1, c2 = st.columns(2)
        with c1:
            front = st.selectbox("Front Leg", contract_list, index=0)
        with c2:
            back = st.selectbox("Back Leg", contract_list, index=1)
        
        # Add Legs: Buy Front / Sell Back
        f_row = df[df['Label'] == front].iloc[0]
        b_row = df[df['Label'] == back].iloc[0]
        legs.append({"Side": "BUY", "Qty": 1 * base_qty, "Contract": f"{prod_root}{f_row['Suffix']}", "Price": f_row[prod_root]})
        legs.append({"Side": "SELL", "Qty": -1 * base_qty, "Contract": f"{prod_root}{b_row['Suffix']}", "Price": b_row[prod_root]})

    elif strat_type == "Butterfly (Fly)":
        # Fly is Body - 2*Belly + Wing
        c_belly, c_width = st.columns([2, 1])
        with c_belly:
            belly_label = st.selectbox("Belly (Center)", contract_list, index=2)
        with c_width:
            width = st.number_input("Wing Width (Months)", 1, 6, 1)
            
        belly_idx = df[df['Label'] == belly_label].index[0]
        
        # Validate indices
        if belly_idx - width >= 0 and belly_idx + width < len(df):
            w1_row = df.iloc[belly_idx - width]
            b_row = df.iloc[belly_idx]
            w2_row = df.iloc[belly_idx + width]
            
            legs.append({"Side": "BUY", "Qty": 1 * base_qty, "Contract": f"{prod_root}{w1_row['Suffix']}", "Price": w1_row[prod_root]})
            legs.append({"Side": "SELL", "Qty": -2 * base_qty, "Contract": f"{prod_root}{b_row['Suffix']}", "Price": b_row[prod_root]})
            legs.append({"Side": "BUY", "Qty": 1 * base_qty, "Contract": f"{prod_root}{w2_row['Suffix']}", "Price": w2_row[prod_root]})
        else:
            st.error(f"Strategy out of bounds. Select a belly contract at least {width} months from start/end.")

    elif strat_type == "Condor":
        # Condor is +1, -1, -1, +1 usually (Iron Condor structure in price space)
        # Or Futures Condor: +1 A, -1 B, -1 C, +1 D
        st.info("Structure: Buy A, Sell B, Sell C, Buy D (Equidistant)")
        c_start, c_width = st.columns([2, 1])
        with c_start:
            start_label = st.selectbox("Front Wing (A)", contract_list, index=0)
        with c_width:
            width = st.number_input("Spacing (Months)", 1, 3, 1)
            
        start_idx = df[df['Label'] == start_label].index[0]
        
        if start_idx + (3*width) < len(df):
            l1 = df.iloc[start_idx]
            l2 = df.iloc[start_idx + width]
            l3 = df.iloc[start_idx + width*2]
            l4 = df.iloc[start_idx + width*3]
            
            legs.append({"Side": "BUY", "Qty": 1 * base_qty, "Contract": f"{prod_root}{l1['Suffix']}", "Price": l1[prod_root]})
            legs.append({"Side": "SELL", "Qty": -1 * base_qty, "Contract": f"{prod_root}{l2['Suffix']}", "Price": l2[prod_root]})
            legs.append({"Side": "SELL", "Qty": -1 * base_qty, "Contract": f"{prod_root}{l3['Suffix']}", "Price": l3[prod_root]})
            legs.append({"Side": "BUY", "Qty": 1 * base_qty, "Contract": f"{prod_root}{l4['Suffix']}", "Price": l4[prod_root]})
            
    # --- CALCULATION ENGINE ---
    if legs:
        st.divider()
        c_ticket, c_risk = st.columns([1, 1.5])
        
        # 1. Execution Ticket
        with c_ticket:
            st.markdown("#### 🎫 Ticket")
            ticket_df = pd.DataFrame(legs)
            ticket_df['Value'] = ticket_df['Qty'] * ticket_df['Price'] * DV01[prod_root] # Approx Notional
            
            # Display readable ticket
            st.dataframe(
                ticket_df[['Side', 'Contract', 'Qty', 'Price']], 
                hide_index=True, 
                use_container_width=True,
                column_config={"Price": st.column_config.NumberColumn(format="%.3f")}
            )
            
            net_price = sum([l['Price'] * (1 if l['Side']=="BUY" else -1) * (abs(l['Qty'])/base_qty) for l in legs])
            # For Butterfly, Price is typically nearly 0 (e.g., 0.05). 
            # If Qty is 1, -2, 1 -> Price = P1 - 2P2 + P3
            
            st.metric("Net Package Price", f"{net_price:.3f}")

        # 2. Risk & Payoff
        with c_risk:
            st.markdown("#### ⚠️ Risk Analysis")
            
            # Scenario Analysis: Parallel Shift AND Curvature
            # Parallel Shift (-20 to +20bps)
            shifts = np.linspace(-25, 25, 51)
            pnl_parallel = []
            
            # Curvature Shift (Belly moves, Wings stay) - mostly for Flys
            pnl_curve = []
            
            # Calculate Risk Vectors
            for s in shifts:
                # Parallel PnL
                run_pnl = 0
                for leg in legs:
                    # If rates UP (shift > 0), Price DOWN. 
                    # Price Delta ~= -Shift. 
                    # PnL = Qty * (PriceDelta) * DV01
                    # Note: Qty is signed (+ for Buy, - for Sell)
                    price_chg = -(s/100) # Simple duration approx
                    run_pnl += leg['Qty'] * price_chg * DV01[prod_root]
                pnl_parallel.append(run_pnl)

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=shifts, y=pnl_parallel, fill='tozeroy', name='Parallel Shift PnL', line=dict(color='#2ECC71')))
            
            fig.update_layout(
                title="PnL vs Market Parallel Shift",
                xaxis_title="Rate Move (bps)",
                yaxis_title="PnL ($)",
                template="plotly_dark",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# MODULE: SPREAD MATRIX (FIXED)
# -----------------------------------------------------------------------------
elif mode == "Spread Matrix":
    st.subheader("Calendar Spread Monitor")
    
    mat_curve = st.selectbox("Select Curve", ["ZQ", "SR1", "SR3"])
    
    # Generate Matrix Data
    matrix_data = []
    contracts = df['Suffix'].tolist()[:12] # First 12 months
    prices = df[mat_curve].tolist()[:12]
    
    # Create correlation-style matrix or list
    spread_list = []
    
    for i in range(len(contracts)-1):
        c1 = contracts[i]
        c2 = contracts[i+1]
        p1 = prices[i]
        p2 = prices[i+1]
        val = p1 - p2 # Buy Front, Sell Back
        spread_list.append({"Pair": f"{c1}/{c2}", "Value": val})

    s_df = pd.DataFrame(spread_list)
    
    # 1. Visual Chart
    st.bar_chart(data=s_df.set_index("Pair"), color="#00F0FF")
    
    # 2. Detailed Table (Formatted)
    st.dataframe(
        s_df.T, 
        use_container_width=True,
        column_config={c: st.column_config.NumberColumn(format="%.3f") for c in s_df.columns}
    )
    
    st.info("💡 Positive Spread = Inverted Curve (Front > Back). Negative Spread = Normal Curve.")
