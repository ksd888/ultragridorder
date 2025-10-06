#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Dashboard for Ultra Grid Trading System
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from pathlib import Path


st.set_page_config(
    page_title="Ultra Grid Trader Dashboard",
    page_icon="📊",
    layout="wide"
)


@st.cache_data(ttl=5)
def load_decisions_log(csv_path: str = "logs/decisions.csv") -> pd.DataFrame:
    """Load decisions CSV"""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        df['bar_time_utc'] = pd.to_datetime(df['bar_time_utc'])
        return df
    except Exception as e:
        st.error(f"Error loading decisions log: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=5)
def load_positions_from_db(db_path: str = "state/bot_state.db") -> pd.DataFrame:
    """Load positions from SQLite"""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM positions ORDER BY entry_time DESC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading positions: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=5)
def load_dust_ledger(csv_path: str = "logs/dust_ledger.csv") -> pd.DataFrame:
    """Load dust ledger"""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading dust ledger: {e}")
        return pd.DataFrame()


def main():
    st.title("📊 Ultra Grid Trading System V2.0")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Settings")
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 10)
    lookback_hours = st.sidebar.slider("Lookback Hours", 1, 24, 6)
    
    # Auto-refresh
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Load data
    decisions_df = load_decisions_log()
    positions_df = load_positions_from_db()
    dust_df = load_dust_ledger()
    
    if decisions_df.empty:
        st.warning("No data available. Make sure the bot is running.")
        return
    
    # Filter by time
    cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
    recent_decisions = decisions_df[decisions_df['bar_time_utc'] >= cutoff_time]
    
    # === OVERVIEW METRICS ===
    st.header("📈 Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_bars = len(recent_decisions)
        st.metric("Total Bars", total_bars)
    
    with col2:
        signals_raw = recent_decisions['signal_raw'].sum()
        st.metric("Raw Signals", int(signals_raw))
    
    with col3:
        signals_confirmed = recent_decisions['signal_confirmed'].sum()
        st.metric("Confirmed Signals", int(signals_confirmed))
    
    with col4:
        confirmation_rate = signals_confirmed / signals_raw if signals_raw > 0 else 0
        st.metric("Confirmation Rate", f"{confirmation_rate:.1%}")
    
    with col5:
        open_positions = positions_df[positions_df['status'] == 'open'].shape[0] if not positions_df.empty else 0
        st.metric("Open Positions", open_positions)
    
    st.markdown("---")
    
    # === PRICE AND SIGNALS CHART ===
    st.header("💹 Price & Signals")
    
    fig = go.Figure()
    
    # Price
    fig.add_trace(go.Scatter(
        x=recent_decisions['bar_time_utc'],
        y=recent_decisions['mid_price'],
        mode='lines',
        name='Mid Price',
        line=dict(color='blue', width=2)
    ))
    
    # Buy signals
    buy_signals = recent_decisions[recent_decisions['action'] == 'PLACE_BUY']
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['bar_time_utc'],
            y=buy_signals['mid_price'],
            mode='markers',
            name='BUY Signal',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    
    # Grid candidates
    fig.add_trace(go.Scatter(
        x=recent_decisions['bar_time_utc'],
        y=recent_decisions['grid_candidate'],
        mode='markers',
        name='Grid Levels',
        marker=dict(color='gray', size=3, opacity=0.3)
    ))
    
    fig.update_layout(
        title="Price Action & Trading Signals",
        xaxis_title="Time",
        yaxis_title="Price",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # === ORDERFLOW INDICATORS ===
    st.header("🔍 Orderflow Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CVD Z-score
        fig_cvd = go.Figure()
        fig_cvd.add_trace(go.Scatter(
            x=recent_decisions['bar_time_utc'],
            y=recent_decisions['cvd_z'],
            mode='lines',
            name='CVD Z-score',
            line=dict(color='purple')
        ))
        fig_cvd.add_hline(y=1.5, line_dash="dash", line_color="green", annotation_text="Threshold")
        fig_cvd.add_hline(y=-1.5, line_dash="dash", line_color="red")
        fig_cvd.update_layout(title="CVD Z-score", height=300)
        st.plotly_chart(fig_cvd, use_container_width=True)
    
    with col2:
        # Trade Size Z-score
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=recent_decisions['bar_time_utc'],
            y=recent_decisions['ts_z'],
            mode='lines',
            name='TS Z-score',
            line=dict(color='orange')
        ))
        fig_ts.add_hline(y=1.5, line_dash="dash", line_color="green", annotation_text="Threshold")
        fig_ts.update_layout(title="Trade Size Z-score", height=300)
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # === TECHNICAL INDICATORS ===
    st.header("📊 Technical Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=recent_decisions['bar_time_utc'],
            y=recent_decisions['rsi'],
            mode='lines',
            line=dict(color='blue')
        ))
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.update_layout(title="RSI", height=250)
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        # VPIN
        fig_vpin = go.Figure()
        fig_vpin.add_trace(go.Scatter(
            x=recent_decisions['bar_time_utc'],
            y=recent_decisions['vpin'],
            mode='lines',
            line=dict(color='teal')
        ))
        fig_vpin.update_layout(title="VPIN", height=250)
        st.plotly_chart(fig_vpin, use_container_width=True)
    
    with col3:
        # Volatility
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=recent_decisions['bar_time_utc'],
            y=recent_decisions['volatility'],
            mode='lines',
            line=dict(color='red')
        ))
        fig_vol.update_layout(title="Volatility", height=250)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    # === POSITIONS TABLE ===
    st.header("💼 Positions")
    
    if not positions_df.empty:
        tab1, tab2 = st.tabs(["Open Positions", "Closed Positions"])
        
        with tab1:
            open_pos = positions_df[positions_df['status'] == 'open']
            if not open_pos.empty:
                st.dataframe(
                    open_pos[['position_id', 'entry_price', 'size_coin', 'size_usd', 
                             'tp_price', 'sl_price', 'entry_time']],
                    use_container_width=True
                )
            else:
                st.info("No open positions")
        
        with tab2:
            closed_pos = positions_df[positions_df['status'] == 'closed']
            if not closed_pos.empty:
                st.dataframe(
                    closed_pos[['position_id', 'entry_price', 'size_coin', 
                               'entry_time', 'exit_time', 'exit_reason']],
                    use_container_width=True
                )
            else:
                st.info("No closed positions")
    else:
        st.info("No positions data available")
    
    # === WYCKOFF PHASES ===
    st.header("🔄 Market Phases")
    
    if 'wyckoff_phase' in recent_decisions.columns:
        phase_counts = recent_decisions['wyckoff_phase'].value_counts()
        
        fig_phases = px.pie(
            values=phase_counts.values,
            names=phase_counts.index,
            title="Wyckoff Phase Distribution"
        )
        st.plotly_chart(fig_phases, use_container_width=True)
    
    # === DUST LEDGER ===
    if not dust_df.empty:
        st.header("🧹 Dust Ledger")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Dust Events", len(dust_df))
        
        with col2:
            total_dust_value = dust_df['est_cost_total_usdt'].sum()
            st.metric("Total Dust Value", f"${total_dust_value:.2f}")
        
        # Dust by reason
        dust_by_reason = dust_df['reason'].value_counts()
        fig_dust = px.bar(
            x=dust_by_reason.index,
            y=dust_by_reason.values,
            title="Dust Events by Reason"
        )
        st.plotly_chart(fig_dust, use_container_width=True)
        
        # Recent dust events
        st.subheader("Recent Dust Events")
        st.dataframe(
            dust_df.tail(20)[['timestamp', 'remainder_qty', 'unit_cost_usdt', 
                              'est_cost_total_usdt', 'reason']],
            use_container_width=True
        )
    
    # === SYSTEM STATUS ===
    st.header("⚙️ System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        data_quality = recent_decisions['data_quality'].value_counts()
        good_pct = data_quality.get('ok', 0) / len(recent_decisions) if len(recent_decisions) > 0 else 0
        st.metric("Data Quality (OK)", f"{good_pct:.1%}")
    
    with col2:
        circuit_breaker = recent_decisions['circuit_breaker_active'].iloc[-1] if not recent_decisions.empty else False
        st.metric("Circuit Breaker", "ACTIVE" if circuit_breaker else "OK")
    
    with col3:
        last_update = recent_decisions['bar_time_utc'].max()
        time_diff = (datetime.now() - last_update).total_seconds()
        st.metric("Last Update", f"{time_diff:.0f}s ago")
    
    # === STATISTICS ===
    st.header("📊 Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Signal Stats")
        stats_df = pd.DataFrame({
            'Metric': ['Total Bars', 'Raw Signals', 'Confirmed Signals', 
                      'Confirmation Rate', 'Active Levels', 'Open Orders'],
            'Value': [
                len(recent_decisions),
                int(recent_decisions['signal_raw'].sum()),
                int(recent_decisions['signal_confirmed'].sum()),
                f"{confirmation_rate:.1%}",
                int(recent_decisions['active_levels'].iloc[-1]) if not recent_decisions.empty else 0,
                int(recent_decisions['open_positions'].iloc[-1]) if not recent_decisions.empty else 0
            ]
        })
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.subheader("Market Stats")
        market_stats = pd.DataFrame({
            'Metric': ['Avg Price', 'Avg Volatility', 'Avg Spread', 'Avg RSI'],
            'Value': [
                f"${recent_decisions['mid_price'].mean():.6f}",
                f"{recent_decisions['volatility'].mean():.4f}",
                f"{recent_decisions['spread_pct'].mean():.4%}",
                f"{recent_decisions['rsi'].mean():.1f}"
            ]
        })
        st.dataframe(market_stats, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh every {refresh_rate}s")


if __name__ == "__main__":
    main()