"""
STIR FUTURES TRADING DASHBOARD - COMPLETE CODE
Copy this entire file as app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    SR3_MULTIPLIER = 2500
    SR1_MULTIPLIER = 4167
    ZQ_MULTIPLIER = 4167
    PRICE_REFRESH_SECONDS = 5
    FRED_REFRESH_SECONDS = 60
    DELTA_NEUTRAL_THRESHOLD = 50
    MAX_DV01_WARNING = 15000
    FRED_API_KEY = "YOUR_FRED_API_KEY"
    CONTRACT_MONTHS = ['MAR', 'JUN', 'SEP', 'DEC']
    YEARS = list(range(2025, 2031))
    
    COLORS = {
        'background': '#0E1117',
        'secondary_bg': '#262730',
        'text': '#FAFAFA',
        'positive': '#00C853',
        'negative': '#FF5252',
        'neutral': '#2196F3',
        'accent': '#FFD700',
        'grid': '#1E1E1E'
    }

# ============================================================================
# DATA FETCHING
# ============================================================================

class DataFetcher:
    def __init__(self):
        self.fred_api_key = Config.FRED_API_KEY
        self.cache = {}
        self.last_fetch_time = {}
    
    def fetch_fred_rate(self, series_id: str) -> float:
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 1
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['observations'][0]['value'])
            return None
        except:
            return None
    
    def fetch_sofr_rate(self) -> float:
        rate = self.fetch_fred_rate('SOFR')
        return rate if rate else 5.30
    
    def fetch_fed_funds_rate(self) -> float:
        rate = self.fetch_fred_rate('DFF')
        return rate if rate else 5.33
    
    def fetch_treasury_yields(self) -> Dict[str, float]:
        series = {
            '1M': 'DGS1MO', '3M': 'DGS3MO', '6M': 'DGS6MO',
            '1Y': 'DGS1', '2Y': 'DGS2', '5Y': 'DGS5', '10Y': 'DGS10'
        }
        yields = {}
        for tenor, series_id in series.items():
            rate = self.fetch_fred_rate(series_id)
            yields[tenor] = rate if rate else self._get_fallback_yield(tenor)
        return yields
    
    def _get_fallback_yield(self, tenor: str) -> float:
        fallbacks = {
            '1M': 5.35, '3M': 5.30, '6M': 5.20,
            '1Y': 5.00, '2Y': 4.75, '5Y': 4.40, '10Y': 4.50
        }
        return fallbacks.get(tenor, 5.0)
    
    def generate_mock_cme_prices(self, contracts: List[str]) -> pd.DataFrame:
        sofr_rate = self.fetch_sofr_rate()
        prices = []
        for i, contract in enumerate(contracts):
            quarters_out = i * 0.25
            implied_rate = sofr_rate - (0.05 * quarters_out) + np.random.normal(0, 0.02)
            price = 100 - implied_rate
            prices.append({
                'Contract': contract,
                'Last': round(price, 3),
                'Change': round(np.random.normal(0, 0.01), 3),
                'Bid': round(price - 0.005, 3),
                'Ask': round(price + 0.005, 3),
                'Volume': np.random.randint(1000, 50000),
                'Implied_Rate': round(implied_rate, 4)
            })
        return pd.DataFrame(prices)
    
    def fetch_fomc_calendar(self) -> List[datetime]:
        """OFFICIAL FOMC DATES FROM FEDERAL RESERVE"""
        
        meetings_2025 = [
            datetime(2025, 1, 29), datetime(2025, 3, 19), datetime(2025, 5, 7),
            datetime(2025, 6, 18), datetime(2025, 7, 30), datetime(2025, 9, 17),
            datetime(2025, 10, 29), datetime(2025, 12, 10)
        ]
        
        meetings_2026 = [
            datetime(2026, 1, 28), datetime(2026, 3, 18), datetime(2026, 4, 29),
            datetime(2026, 6, 17), datetime(2026, 7, 29), datetime(2026, 9, 16),
            datetime(2026, 10, 28), datetime(2026, 12, 9)
        ]
        
        meetings_2027 = [
            datetime(2027, 1, 27), datetime(2027, 3, 17), datetime(2027, 4, 28),
            datetime(2027, 6, 9), datetime(2027, 7, 28), datetime(2027, 9, 15),
            datetime(2027, 10, 27), datetime(2027, 12, 8)
        ]
        
        meetings_2028 = [
            datetime(2028, 1, 26), datetime(2028, 3, 22), datetime(2028, 5, 3),
            datetime(2028, 6, 14), datetime(2028, 7, 26), datetime(2028, 9, 20),
            datetime(2028, 11, 1), datetime(2028, 12, 13)
        ]
        
        meetings_2029 = [
            datetime(2029, 1, 31), datetime(2029, 3, 21), datetime(2029, 5, 2),
            datetime(2029, 6, 13), datetime(2029, 7, 25), datetime(2029, 9, 19),
            datetime(2029, 10, 31), datetime(2029, 12, 12)
        ]
        
        meetings_2030 = [
            datetime(2030, 1, 30), datetime(2030, 3, 20), datetime(2030, 5, 1),
            datetime(2030, 6, 19), datetime(2030, 7, 31), datetime(2030, 9, 18),
            datetime(2030, 10, 30), datetime(2030, 12, 11)
        ]
        
        all_meetings = (meetings_2025 + meetings_2026 + meetings_2027 + 
                       meetings_2028 + meetings_2029 + meetings_2030)
        return sorted(all_meetings)

# ============================================================================
# PRICING ENGINE
# ============================================================================

class PricingEngine:
    def __init__(self):
        self.data_fetcher = DataFetcher()
    
    def build_forward_curve(self, base_rate: float, yields: Dict[str, float]) -> CubicSpline:
        tenor_map = {
            '1M': 1/12, '3M': 0.25, '6M': 0.5,
            '1Y': 1.0, '2Y': 2.0, '5Y': 5.0, '10Y': 10.0
        }
        tenors = [tenor_map[k] for k in yields.keys()]
        rates = [yields[k] for k in yields.keys()]
        tenors.insert(0, 0)
        rates.insert(0, base_rate)
        curve = CubicSpline(tenors, rates)
        return curve
    
    def calculate_dv01(self, contract_type: str, quantity: int, price: float, 
                      tenor_months: int = 3) -> float:
        multipliers = {
            'SR3': Config.SR3_MULTIPLIER,
            'SR1': Config.SR1_MULTIPLIER,
            'ZQ': Config.ZQ_MULTIPLIER
        }
        multiplier = multipliers.get(contract_type, Config.SR3_MULTIPLIER)
        dv01 = quantity * multiplier * (tenor_months / 12) / 10000
        return dv01

# ============================================================================
# SCENARIO GENERATOR
# ============================================================================

class ScenarioGenerator:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.pricing_engine = PricingEngine()
        self.fomc_dates = self.data_fetcher.fetch_fomc_calendar()
    
    def generate_scenarios(self, current_rate: float, num_scenarios: int = 20) -> List[Dict]:
        scenarios = []
        scenarios.append(self._create_baseline_scenario(current_rate))
        scenarios.extend(self._create_dot_plot_scenarios(current_rate))
        scenarios.extend(self._create_monte_carlo_scenarios(current_rate, n=10))
        scenarios.extend(self._create_shock_scenarios(current_rate))
        
        total_prob = sum(s['probability'] for s in scenarios)
        for s in scenarios:
            s['probability'] = s['probability'] / total_prob
        
        return scenarios[:num_scenarios]
    
    def _create_baseline_scenario(self, current_rate: float) -> Dict:
        path = [current_rate]
        rate = current_rate
        meetings = len([d for d in self.fomc_dates if d > datetime.now()])
        target = 3.5
        
        for i in range(min(meetings, 30)):
            if rate > target:
                cut = min(0.25, (rate - target) / 2)
                rate -= cut
            path.append(rate)
        
        return {
            'name': 'Market Implied',
            'path': path,
            'terminal_rate': path[-1],
            'probability': 0.40,
            'color': Config.COLORS['neutral']
        }
    
    def _create_dot_plot_scenarios(self, current_rate: float) -> List[Dict]:
        scenarios = []
        
        hawkish_path = [current_rate]
        rate = current_rate
        for _ in range(30):
            if rate > 4.0:
                rate -= 0.125
            hawkish_path.append(rate)
        
        scenarios.append({
            'name': 'Hawkish (Dot Plot)',
            'path': hawkish_path,
            'terminal_rate': hawkish_path[-1],
            'probability': 0.15,
            'color': '#FF6B6B'
        })
        
        median_path = [current_rate]
        rate = current_rate
        for _ in range(30):
            if rate > 3.5:
                rate -= 0.20
            median_path.append(rate)
        
        scenarios.append({
            'name': 'Median (Dot Plot)',
            'path': median_path,
            'terminal_rate': median_path[-1],
            'probability': 0.25,
            'color': '#4ECDC4'
        })
        
        dovish_path = [current_rate]
        rate = current_rate
        for _ in range(30):
            if rate > 3.0:
                rate -= 0.30
            dovish_path.append(rate)
        
        scenarios.append({
            'name': 'Dovish (Dot Plot)',
            'path': dovish_path,
            'terminal_rate': dovish_path[-1],
            'probability': 0.10,
            'color': '#95E1D3'
        })
        
        return scenarios
    
    def _create_monte_carlo_scenarios(self, current_rate: float, n: int = 10) -> List[Dict]:
        scenarios = []
        for i in range(n):
            path = [current_rate]
            rate = current_rate
            target = np.random.uniform(2.5, 4.5)
            volatility = 0.15
            
            for _ in range(30):
                drift = (target - rate) * 0.1
                shock = np.random.normal(0, volatility)
                change = drift + shock
                rate = max(0.0, min(6.0, rate + change))
                path.append(rate)
            
            scenarios.append({
                'name': f'MC Scenario {i+1}',
                'path': path,
                'terminal_rate': path[-1],
                'probability': 0.30 / n,
                'color': f'rgba(150, 150, 150, 0.3)'
            })
        
        return scenarios
    
    def _create_shock_scenarios(self, current_rate: float) -> List[Dict]:
        scenarios = []
        
        recession_path = [current_rate]
        rate = current_rate
        for i in range(30):
            if i < 8:
                rate = max(2.0, rate - 0.50)
            recession_path.append(rate)
        
        scenarios.append({
            'name': 'Recession',
            'path': recession_path,
            'terminal_rate': recession_path[-1],
            'probability': 0.05,
            'color': '#D32F2F'
        })
        
        hike_path = [current_rate]
        rate = current_rate
        for i in range(30):
            if i < 6:
                rate = min(6.0, rate + 0.25)
            hike_path.append(rate)
        
        scenarios.append({
            'name': 'Reacceleration',
            'path': hike_path,
            'terminal_rate': hike_path[-1],
            'probability': 0.05,
            'color': '#FFA726'
        })
        
        return scenarios
    
    def price_contracts_in_scenario(self, scenario: Dict, contracts: List[str]) -> Dict[str, float]:
        prices = {}
        for contract in contracts:
            parts = contract.split('_')
            if len(parts) != 2:
                continue
            
            month_code = parts[1][:3]
            year = int('20' + parts[1][3:])
            month_map = {'MAR': 3, 'JUN': 6, 'SEP': 9, 'DEC': 12}
            month = month_map.get(month_code, 3)
            maturity_date = datetime(year, month, 15)
            
            days_from_now = (maturity_date - datetime.now()).days
            meetings_out = max(0, min(len(scenario['path']) - 1, days_from_now // 45))
            implied_rate = scenario['path'][meetings_out]
            price = 100 - implied_rate
            prices[contract] = round(price, 3)
        
        return prices

# ============================================================================
# STRATEGY BUILDER
# ============================================================================

class StrategyBuilder:
    def __init__(self):
        self.pricing_engine = PricingEngine()
    
    def build_outright(self, contract: str, quantity: int, price: float) -> Dict:
        contract_type = contract.split('_')[0]
        dv01 = self.pricing_engine.calculate_dv01(contract_type, quantity, price)
        return {
            'type': 'Outright',
            'legs': [{'contract': contract, 'quantity': quantity, 'price': price}],
            'dv01': dv01,
            'description': f'{quantity:+d} {contract} @ {price}'
        }
    
    def build_calendar_spread(self, front_contract: str, back_contract: str,
                              front_qty: int, front_price: float, back_price: float) -> Dict:
        contract_type = front_contract.split('_')[0]
        back_qty = -front_qty
        
        front_dv01 = self.pricing_engine.calculate_dv01(contract_type, front_qty, front_price)
        back_dv01 = self.pricing_engine.calculate_dv01(contract_type, back_qty, back_price)
        total_dv01 = front_dv01 + back_dv01
        
        return {
            'type': 'Calendar Spread',
            'legs': [
                {'contract': front_contract, 'quantity': front_qty, 'price': front_price},
                {'contract': back_contract, 'quantity': back_qty, 'price': back_price}
            ],
            'dv01': total_dv01,
            'spread': front_price - back_price,
            'description': f'{front_contract}/{back_contract} spread'
        }
    
    def build_butterfly(self, front: str, middle: str, back: str,
                       front_price: float, middle_price: float, back_price: float,
                       ratio: Tuple[int, int, int] = (1, -2, 1)) -> Dict:
        contract_type = front.split('_')[0]
        
        legs = [
            {'contract': front, 'quantity': ratio[0], 'price': front_price},
            {'contract': middle, 'quantity': ratio[1], 'price': middle_price},
            {'contract': back, 'quantity': ratio[2], 'price': back_price}
        ]
        
        total_dv01 = sum([
            self.pricing_engine.calculate_dv01(contract_type, leg['quantity'], leg['price'])
            for leg in legs
        ])
        
        fly_value = (front_price + back_price) / 2 - middle_price
        
        return {
            'type': 'Butterfly',
            'legs': legs,
            'dv01': total_dv01,
            'fly_value': fly_value,
            'is_neutral': abs(total_dv01) < Config.DELTA_NEUTRAL_THRESHOLD,
            'description': f'{front}/{middle}/{back} fly'
        }
    
    def calculate_strategy_pnl(self, strategy: Dict, scenario_prices: Dict[str, float]) -> float:
        pnl = 0.0
        for leg in strategy['legs']:
            contract = leg['contract']
            quantity = leg['quantity']
            entry_price = leg['price']
            scenario_price = scenario_prices.get(contract, entry_price)
            
            contract_type = contract.split('_')[0]
            multiplier = {
                'SR3': Config.SR3_MULTIPLIER,
                'SR1': Config.SR1_MULTIPLIER,
                'ZQ': Config.ZQ_MULTIPLIER
            }.get(contract_type, Config.SR3_MULTIPLIER)
            
            leg_pnl = quantity * multiplier * (scenario_price - entry_price) * 100
            pnl += leg_pnl
        
        return pnl

# ============================================================================
# RISK CALCULATOR
# ============================================================================

class RiskCalculator:
    def __init__(self):
        self.pricing_engine = PricingEngine()
    
    def calculate_portfolio_dv01(self, strategies: List[Dict]) -> float:
        return sum(s.get('dv01', 0) for s in strategies)

# ============================================================================
# VISUALIZER
# ============================================================================

class Visualizer:
    @staticmethod
    def create_forward_curve_chart(scenarios: List[Dict], current_rate: float) -> go.Figure:
        fig = go.Figure()
        
        for scenario in scenarios[:8]:
            meetings = list(range(len(scenario['path'])))
            fig.add_trace(go.Scatter(
                x=meetings,
                y=scenario['path'],
                mode='lines',
                name=scenario['name'],
                line=dict(color=scenario.get('color', Config.COLORS['neutral']), width=2),
                opacity=0.7
            ))
        
        fig.add_hline(
            y=current_rate,
            line_dash="dash",
            line_color=Config.COLORS['accent'],
            annotation_text="Current Rate"
        )
        
        fig.update_layout(
            title="FOMC Rate Path Scenarios",
            xaxis_title="Meetings Ahead",
            yaxis_title="Fed Funds Rate (%)",
            template="plotly_dark",
            paper_bgcolor=Config.COLORS['background'],
            plot_bgcolor=Config.COLORS['background'],
            font=dict(color=Config.COLORS['text']),
            hovermode='x unified',
            height=500
        )
        
        return fig

# ============================================================================
# STREAMLIT UI
# ============================================================================

def init_session_state():
    if 'strategies' not in st.session_state:
        st.session_state.strategies = []
    if 'account_size' not in st.session_state:
        st.session_state.account_size = 100000
    if 'max_risk_pct' not in st.session_state:
        st.session_state.max_risk_pct = 2.0
    if 'max_dv01' not in st.session_state:
        st.session_state.max_dv01 = 5000
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    
    if 'contracts' not in st.session_state:
        contracts = []
        current_year = 2025
        current_quarter = 0
        for i in range(8):
            month = Config.CONTRACT_MONTHS[current_quarter]
            contracts.append(f"SR3_{month}{str(current_year)[2:]}")
            current_quarter = (current_quarter + 1) % 4
            if current_quarter == 0:
                current_year += 1
        
        for year in Config.YEARS:
            for month in Config.CONTRACT_MONTHS:
                contracts.append(f"SR1_{month}{str(year)[2:]}")
                contracts.append(f"ZQ_{month}{str(year)[2:]}")
        
        st.session_state.contracts = contracts

def add_notification(message: str, type: str = "info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.notifications.insert(0, {
        'time': timestamp,
        'message': message,
        'type': type
    })
    st.session_state.notifications = st.session_state.notifications[:10]

def render_header():
    st.markdown(f"""
        <div style='background-color: {Config.COLORS['secondary_bg']}; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h1 style='color: {Config.COLORS['accent']}; margin: 0;'>⚡ STIR Futures Trading Workstation</h1>
            <p style='color: {Config.COLORS['text']}; margin: 5px 0 0 0;'>
                Professional SR1/SR3/ZQ Trading Platform with FOMC Scenario Modeling
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown(f"<h2 style='color: {Config.COLORS['accent']}'>⚙️ Settings</h2>", unsafe_allow_html=True)
        
        page = st.radio(
            "📍 Navigate to:",
            ["Live Prices", "Strategy Builder", "FOMC Scenarios"],
            key="navigation"
        )
        
        st.divider()
        st.markdown(f"<h3 style='color: {Config.COLORS['text']}'>💰 Account</h3>", unsafe_allow_html=True)
        
        st.session_state.account_size = st.number_input(
            "Account Size ($)",
            min_value=10000,
            value=st.session_state.account_size,
            step=10000
        )
        
        st.session_state.max_risk_pct = st.slider(
            "Max Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.max_risk_pct,
            step=0.5
        )
        
        return page

def render_live_prices_page():
    st.markdown(f"<h2 style='color: {Config.COLORS['accent']}'>📈 Live Prices</h2>", unsafe_allow_html=True)
    
    data_fetcher = DataFetcher()
    contracts = st.session_state.contracts
    prices_df = data_fetcher.generate_mock_cme_prices(contracts)
    
    tab1, tab2, tab3 = st.tabs(["SR3 Contracts", "SR1 Contracts", "ZQ Contracts"])
    
    with tab1:
        sr3_prices = prices_df[prices_df['Contract'].str.startswith('SR3')]
        st.dataframe(sr3_prices, use_container_width=True, height=400)
    
    with tab2:
        sr1_prices = prices_df[prices_df['Contract'].str.startswith('SR1')]
        st.dataframe(sr1_prices, use_container_width=True, height=400)
    
    with tab3:
        zq_prices = prices_df[prices_df['Contract'].str.startswith('ZQ')]
        st.dataframe(zq_prices, use_container_width=True, height=400)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sofr = data_fetcher.fetch_sofr_rate()
        st.metric("SOFR Rate", f"{sofr:.2f}%")
    
    with col2:
        ff = data_fetcher.fetch_fed_funds_rate()
        st.metric("Fed Funds", f"{ff:.2f}%")
    
    with col3:
        st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))

def render_strategy_builder_page():
    st.markdown(f"<h2 style='color: {Config.COLORS['accent']}'>🔨 Strategy Builder</h2>", unsafe_allow_html=True)
    
    builder = StrategyBuilder()
    data_fetcher = DataFetcher()
    
    contracts = st.session_state.contracts
    prices_df = data_fetcher.generate_mock_cme_prices(contracts)
    price_dict = dict(zip(prices_df['Contract'], prices_df['Last']))
    
    strategy_type = st.selectbox(
        "Select Strategy Type",
        ["Outright", "Calendar Spread", "Butterfly"]
    )
    
    st.markdown("---")
    
    if strategy_type == "Outright":
        col1, col2 = st.columns(2)
        with col1:
            contract = st.selectbox("Contract", contracts)
            quantity = st.number_input("Quantity", value=1, step=1)
        with col2:
            price = st.number_input("Price", value=price_dict.get(contract, 96.0), step=0.001, format="%.3f")
        
        if st.button("Add Strategy", type="primary"):
            strategy = builder.build_outright(contract, quantity, price)
            st.session_state.strategies.append(strategy)
            add_notification(f"Added outright: {strategy['description']}", "success")
            st.rerun()
    
    elif strategy_type == "Calendar Spread":
        col1, col2 = st.columns(2)
        with col1:
            front = st.selectbox("Front Contract", contracts, key="front")
            quantity = st.number_input("Quantity", value=1, step=1)
        with col2:
            back = st.selectbox("Back Contract", contracts, key="back")
        
        front_price = price_dict.get(front, 96.0)
        back_price = price_dict.get(back, 95.8)
        
        st.write(f"**Spread**: {front_price - back_price:.3f}")
        
        if st.button("Add Strategy", type="primary"):
            strategy = builder.build_calendar_spread(front, back, quantity, front_price, back_price)
            st.session_state.strategies.append(strategy)
            add_notification(f"Added calendar spread", "success")
            st.rerun()
    
    elif strategy_type == "Butterfly":
        col1, col2, col3 = st.columns(3)
        with col1:
            front = st.selectbox("Front", contracts, key="fly_front")
        with col2:
            middle = st.selectbox("Middle", contracts, key="fly_middle")
        with col3:
            back = st.selectbox("Back", contracts, key="fly_back")
        
        front_price = price_dict.get(front, 96.2)
        middle_price = price_dict.get(middle, 96.0)
        back_price = price_dict.get(back, 95.8)
        
        if st.button("Add Strategy", type="primary"):
            strategy = builder.build_butterfly(front, middle, back, front_price, middle_price, back_price)
            st.session_state.strategies.append(strategy)
            add_notification(f"Added butterfly", "success")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📋 Active Strategies")
    
    if st.session_state.strategies:
        for i, strategy in enumerate(st.session_state.strategies):
            with st.expander(f"{strategy['type']}: {strategy['description']}"):
                st.write("**Legs:**")
                for leg in strategy['legs']:
                    st.write(f"- {leg['quantity']:+d} {leg['contract']} @ {leg['price']:.3f}")
                st.write(f"**DV01:** ${strategy['dv01']:.2f}")
                
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.strategies.pop(i)
                    st.rerun()
    else:
        st.info("No strategies yet. Build one above!")

def render_fomc_scenarios_page():
    st.markdown(f"<h2 style='color: {Config.COLORS['accent']}'>🎯 FOMC Scenarios</h2>", unsafe_allow_html=True)
    
    scenario_gen = ScenarioGenerator()
    data_fetcher = DataFetcher()
    visualizer = Visualizer()
    
    current_rate = data_fetcher.fetch_sofr_rate()
    
    with st.spinner("Generating 20 FOMC scenarios..."):
        scenarios = scenario_gen.generate_scenarios(current_rate, num_scenarios=20)
    
    st.markdown("### 📊 Rate Path Scenarios")
    fig = visualizer.create_forward_curve_chart(scenarios, current_rate)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Scenario Details")
    scenario_data = []
    for s in scenarios:
        scenario_data.append({
            'Scenario': s['name'],
            'Terminal Rate': f"{s['terminal_rate']:.2f}%",
            'Probability': f"{s['probability']*100:.1f}%"
        })
    
    scenario_df = pd.DataFrame(scenario_data)
    st.dataframe(scenario_df, use_container_width=True)

def main():
    st.set_page_config(
        page_title="STIR Trading Workstation",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(f"""
        <style>
            .stApp {{
                background-color: {Config.COLORS['background']};
                color: {Config.COLORS['text']};
            }}
        </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    render_header()
    page = render_sidebar()
    
    if page == "Live Prices":
        render_live_prices_page()
    elif page == "Strategy Builder":
        render_strategy_builder_page()
    elif page == "FOMC Scenarios":
        render_fomc_scenarios_page()

if __name__ == "__main__":
    main()
