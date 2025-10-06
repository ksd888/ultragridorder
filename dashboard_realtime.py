#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Trading Bot Dashboard (Dash)
Enhanced with features from template bot

Features:
- Live position monitoring
- Trade timeline visualization
- Order flow metrics (CVD, imbalance)
- Dust tracking
- Performance metrics
- Auto-refresh every 5 seconds
"""

import os
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Output, Input

# ================= Configuration =================
REFRESH_INTERVAL = 2000  # 2 seconds - faster refresh
STATE_DB = os.path.join("data", "bot_state_test.db")  # Force reload
LOG_DIR = "logs"
DUST_LEDGER = "logs/dust_ledger.csv"
DECISIONS_LOG = "logs/decisions_test.csv"  # Use test log
STRUCTURED_LOG = "logs/bot.jsonl"
WINDOW_MINUTES = 60  # Show last 60 minutes
TZ_NAME = "Asia/Bangkok"  # GMT+7

# ================= Helper Functions =================

def ensure_dirs():
    """Ensure necessary directories exist"""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STATE_DB), exist_ok=True)


def load_bot_state() -> Dict:
    """Load latest bot state from database"""
    try:
        if not os.path.exists(STATE_DB):
            print(f"[DEBUG] State DB not found: {STATE_DB}")
            return _default_state()

        conn = sqlite3.connect(STATE_DB)
        cursor = conn.execute(
            "SELECT * FROM state_snapshots ORDER BY timestamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            # Parse JSON state (new format)
            state_json = json.loads(row[2]) if row[2] else {}

            # Extract metrics from positions
            positions = state_json.get('positions', {})
            total_exposure = sum(p.get('size_usd', 0) for p in positions.values())

            state = {
                'timestamp': row[1],
                'active_positions': len(positions),
                'total_exposure': total_exposure,
                'active_levels': len(state_json.get('active_levels', [])),
                'circuit_breaker': state_json.get('circuit_breaker_active', False)
            }
            print(f"[DEBUG] State loaded: {state}")
            return state
        else:
            print("[DEBUG] No state snapshots found")
    except Exception as e:
        print(f"[ERROR] Failed to load state: {e}")
        import traceback
        traceback.print_exc()

    return _default_state()


def _default_state():
    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'active_positions': 0,
        'total_exposure': 0,
        'active_levels': 0,
        'circuit_breaker': False
    }


def load_positions() -> pd.DataFrame:
    """Load open positions from database"""
    try:
        if not os.path.exists(STATE_DB):
            print(f"[DEBUG] Database not found: {STATE_DB}")
            return pd.DataFrame()

        conn = sqlite3.connect(STATE_DB)
        df = pd.read_sql_query(
            "SELECT * FROM positions WHERE status='open' ORDER BY entry_time DESC",
            conn
        )
        conn.close()
        print(f"[DEBUG] Loaded {len(df)} positions from DB")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load positions: {e}")
        return pd.DataFrame()


def load_decisions_log() -> pd.DataFrame:
    """Load decisions log with order flow metrics"""
    try:
        if not os.path.exists(DECISIONS_LOG):
            return pd.DataFrame()

        df = pd.read_csv(DECISIONS_LOG)

        if 'bar_time_utc' in df.columns:
            df['time'] = pd.to_datetime(df['bar_time_utc'])
        elif 'bar_ts_ms' in df.columns:
            df['time'] = pd.to_datetime(df['bar_ts_ms'], unit='ms')

        # Filter to window
        if 'time' in df.columns and not df.empty:
            cutoff = df['time'].max() - pd.Timedelta(minutes=WINDOW_MINUTES)
            df = df[df['time'] >= cutoff]

        return df

    except Exception:
        return pd.DataFrame()


def load_trades_log() -> List[Dict]:
    """Load recent trades from structured log"""
    trades = []

    if not os.path.exists(STRUCTURED_LOG):
        return trades

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=WINDOW_MINUTES)

        with open(STRUCTURED_LOG, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())

                    if entry.get('event_type') == 'trade_executed':
                        ts = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))

                        if ts >= cutoff:
                            trades.append(entry)

                except json.JSONDecodeError:
                    continue

    except FileNotFoundError:
        pass

    return trades


def load_dust_ledger() -> pd.DataFrame:
    """Load dust ledger"""
    try:
        if os.path.exists(DUST_LEDGER):
            df = pd.read_csv(DUST_LEDGER)
            return df.tail(20)  # Last 20 entries
    except Exception:
        pass

    return pd.DataFrame()


def calculate_performance_metrics(trades: List[Dict]) -> Dict:
    """Calculate performance metrics from trades"""
    if not trades:
        return {
            'total_trades': 0,
            'total_volume': 0,
            'buy_count': 0,
            'sell_count': 0,
            'avg_buy_price': 0,
            'avg_sell_price': 0
        }

    buy_trades = [t for t in trades if t.get('side') == 'BUY']
    sell_trades = [t for t in trades if t.get('side') == 'SELL']

    buy_count = len(buy_trades)
    sell_count = len(sell_trades)
    total_volume = sum(t.get('notional', 0) for t in trades)

    avg_buy_price = np.mean([t.get('price', 0) for t in buy_trades]) if buy_trades else 0
    avg_sell_price = np.mean([t.get('price', 0) for t in sell_trades]) if sell_trades else 0

    return {
        'total_trades': len(trades),
        'total_volume': total_volume,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'avg_buy_price': avg_buy_price,
        'avg_sell_price': avg_sell_price
    }


# ================= Dash Application =================

ensure_dirs()

app = Dash(
    __name__,
    title="Grid Trading Monitor",
    suppress_callback_exceptions=True,
    update_title=None  # Disable title updates that trigger caching
)

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("[BOT] Grid Trading Monitor", style={
            'textAlign': 'center',
            'color': '#00d4aa',
            'marginBottom': '10px'
        }),
        html.P(f"Auto-refresh: {REFRESH_INTERVAL/1000}s", style={
            'textAlign': 'center',
            'color': '#888',
            'fontSize': '14px'
        })
    ]),

    # Auto-refresh
    dcc.Interval(id='refresh-interval', interval=REFRESH_INTERVAL, n_intervals=0),

    # Status Cards Row
    html.Div(id='status-cards', children=[], style={
        'display': 'flex',
        'justifyContent': 'space-around',
        'margin': '20px',
        'gap': '10px',
        'flexWrap': 'wrap'
    }),

    # Charts Row 1: Trade Timeline & Order Flow
    html.Div([
        html.Div([
            dcc.Graph(id='trade-timeline')
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='orderflow-chart')
        ], style={'width': '50%', 'display': 'inline-block'})
    ]),

    # Positions Table
    html.Div([
        html.H3("[POSITIONS] Open Positions", style={'color': '#00d4aa'}),
        html.Div(id='positions-table')
    ], style={'margin': '20px'}),

    # Dust Ledger
    html.Div([
        html.H3("[DUST] Dust Ledger (Remainders)", style={'color': '#ff6b6b'}),
        html.Div(id='dust-table')
    ], style={'margin': '20px'}),

    # Footer
    html.Div([
        html.P(f"Last Updated: ", id='last-update', style={
            'textAlign': 'center',
            'color': '#666',
            'fontSize': '12px',
            'marginTop': '30px'
        })
    ])

], style={
    'backgroundColor': '#0a0a0a',
    'color': '#fff',
    'fontFamily': 'Arial, sans-serif',
    'minHeight': '100vh',
    'padding': '20px'
})


# ================= Callbacks =================

@app.callback(
    [Output('status-cards', 'children'),
     Output('last-update', 'children')],
    Input('refresh-interval', 'n_intervals')
)
def update_status_cards(n):
    """Update status cards"""
    state = load_bot_state()
    trades = load_trades_log()
    metrics = calculate_performance_metrics(trades)

    card_style_base = {
        'backgroundColor': '#1a1a1a',
        'padding': '20px',
        'borderRadius': '10px',
        'textAlign': 'center',
        'border': '1px solid #333',
        'minWidth': '150px'
    }

    cards = [
        html.Div([
            html.H4("Active Positions", style={'color': '#888', 'fontSize': '14px', 'marginBottom': '10px'}),
            html.H2(str(state['active_positions']), style={'color': '#00d4aa', 'margin': '0'})
        ], style=card_style_base),

        html.Div([
            html.H4("Total Exposure", style={'color': '#888', 'fontSize': '14px', 'marginBottom': '10px'}),
            html.H2(f"${state['total_exposure']:.2f}", style={'color': '#00d4aa', 'margin': '0'})
        ], style=card_style_base),

        html.Div([
            html.H4("Active Levels", style={'color': '#888', 'fontSize': '14px', 'marginBottom': '10px'}),
            html.H2(str(state['active_levels']), style={'color': '#00d4aa', 'margin': '0'})
        ], style=card_style_base),

        html.Div([
            html.H4("Trades (1H)", style={'color': '#888', 'fontSize': '14px', 'marginBottom': '10px'}),
            html.H2(f"B:{metrics['buy_count']} / S:{metrics['sell_count']}", style={'color': '#00d4aa', 'margin': '0'})
        ], style=card_style_base),

        html.Div([
            html.H4("Volume (1H)", style={'color': '#888', 'fontSize': '14px', 'marginBottom': '10px'}),
            html.H2(f"${metrics['total_volume']:.0f}", style={'color': '#00d4aa', 'margin': '0'})
        ], style=card_style_base),

        html.Div([
            html.H4("Circuit Breaker", style={'color': '#888', 'fontSize': '14px', 'marginBottom': '10px'}),
            html.H2("[!] ACTIVE" if state['circuit_breaker'] else "[OK]",
                   style={'color': '#ff6b6b' if state['circuit_breaker'] else '#00ff00', 'margin': '0'})
        ], style={**card_style_base, 'backgroundColor': '#3a0000' if state['circuit_breaker'] else '#003a00'})
    ]

    last_update = f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return cards, last_update


@app.callback(
    Output('positions-table', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_positions_table(n):
    """Update positions table"""
    positions = load_positions()

    # Debug: print to console
    print(f"[DEBUG] Positions loaded: {len(positions)} rows")

    if positions.empty:
        return html.P("No open positions", style={'color': '#666', 'fontStyle': 'italic'})

    # Table styles
    table_style = {
        'width': '100%',
        'borderCollapse': 'collapse',
        'backgroundColor': '#1a1a1a',
        'border': '1px solid #333'
    }

    th_style = {
        'padding': '12px',
        'textAlign': 'left',
        'borderBottom': '2px solid #00d4aa',
        'color': '#00d4aa',
        'fontSize': '14px'
    }

    td_style = {
        'padding': '10px',
        'borderBottom': '1px solid #333',
        'fontSize': '13px'
    }

    # Create table header
    table_header = html.Thead(html.Tr([
        html.Th("Position ID", style=th_style),
        html.Th("Entry Price", style=th_style),
        html.Th("Size (Coin)", style=th_style),
        html.Th("TP Price", style=th_style),
        html.Th("TP %", style=th_style),
        html.Th("Entry Time", style=th_style)
    ]))

    # Create table rows
    rows = []
    for _, pos in positions.iterrows():
        entry_price = pos.get('entry_price', 0)
        tp_price = pos.get('tp_price', 0)
        tp_pct = ((tp_price / entry_price) - 1) * 100 if entry_price > 0 else 0

        rows.append(html.Tr([
            html.Td(str(pos.get('position_id', 'N/A'))[:16], style=td_style),
            html.Td(f"${entry_price:.6f}", style=td_style),
            html.Td(f"{pos.get('size_coin', 0):.4f}", style=td_style),
            html.Td(f"${tp_price:.6f}", style=td_style),
            html.Td(f"+{tp_pct:.2f}%", style={**td_style, 'color': '#00ff00'}),
            html.Td(str(pos.get('entry_time', 'N/A'))[:19], style=td_style)
        ]))

    table_body = html.Tbody(rows)

    return html.Table([table_header, table_body], style=table_style)


@app.callback(
    Output('trade-timeline', 'figure'),
    Input('refresh-interval', 'n_intervals')
)
def update_trade_timeline(n):
    """Update trade timeline chart"""
    trades = load_trades_log()

    fig = go.Figure()

    if trades:
        # Extract data
        timestamps = [datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00')) for t in trades]
        prices = [t.get('price', 0) for t in trades]
        sides = [t.get('side', '') for t in trades]
        quantities = [t.get('quantity', 0) for t in trades]

        # Separate buy/sell
        buy_times = [timestamps[i] for i, s in enumerate(sides) if s == 'BUY']
        buy_prices = [prices[i] for i, s in enumerate(sides) if s == 'BUY']
        buy_qtys = [quantities[i] for i, s in enumerate(sides) if s == 'BUY']

        sell_times = [timestamps[i] for i, s in enumerate(sides) if s == 'SELL']
        sell_prices = [prices[i] for i, s in enumerate(sides) if s == 'SELL']
        sell_qtys = [quantities[i] for i, s in enumerate(sides) if s == 'SELL']

        # Buy trades
        if buy_times:
            fig.add_trace(go.Scatter(
                x=buy_times,
                y=buy_prices,
                mode='markers',
                name='BUY',
                marker=dict(
                    color='#00d4aa',
                    size=[max(8, min(q * 100, 20)) for q in buy_qtys],
                    symbol='triangle-up',
                    line=dict(color='#fff', width=1)
                ),
                hovertemplate='<b>BUY</b><br>Price: $%{y:.6f}<br>%{x}<extra></extra>'
            ))

        # Sell trades
        if sell_times:
            fig.add_trace(go.Scatter(
                x=sell_times,
                y=sell_prices,
                mode='markers',
                name='SELL',
                marker=dict(
                    color='#ff6b6b',
                    size=[max(8, min(q * 100, 20)) for q in sell_qtys],
                    symbol='triangle-down',
                    line=dict(color='#fff', width=1)
                ),
                hovertemplate='<b>SELL</b><br>Price: $%{y:.6f}<br>%{x}<extra></extra>'
            ))

    fig.update_layout(
        title=f"Trade Timeline (Last {WINDOW_MINUTES} min)",
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        hovermode='closest',
        showlegend=True,
        height=400,
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#0a0a0a'
    )

    return fig


@app.callback(
    Output('orderflow-chart', 'figure'),
    Input('refresh-interval', 'n_intervals')
)
def update_orderflow_chart(n):
    """Update order flow chart (CVD, imbalance)"""
    decisions = load_decisions_log()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("CVD Z-Score", "Order Imbalance")
    )

    if not decisions.empty and 'time' in decisions.columns:
        # CVD Z-Score
        if 'cvd_z' in decisions.columns:
            fig.add_trace(
                go.Scatter(
                    x=decisions['time'],
                    y=decisions['cvd_z'],
                    mode='lines',
                    name='CVD Z',
                    line=dict(color='#00d4aa', width=2)
                ),
                row=1, col=1
            )

            # Add threshold lines
            fig.add_hline(y=2.0, line_dash="dash", line_color="#888", row=1, col=1)
            fig.add_hline(y=-2.0, line_dash="dash", line_color="#888", row=1, col=1)

        # Order Imbalance
        if 'order_imbalance' in decisions.columns:
            fig.add_trace(
                go.Scatter(
                    x=decisions['time'],
                    y=decisions['order_imbalance'],
                    mode='lines',
                    name='Imbalance',
                    line=dict(color='#ffa500', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 165, 0, 0.1)'
                ),
                row=2, col=1
            )

    fig.update_layout(
        title=f"Order Flow Metrics (Last {WINDOW_MINUTES} min)",
        template="plotly_dark",
        height=400,
        showlegend=False,
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#0a0a0a'
    )

    fig.update_yaxes(title_text="Z-Score", row=1, col=1)
    fig.update_yaxes(title_text="Imbalance", row=2, col=1)

    return fig


@app.callback(
    Output('dust-table', 'children'),
    Input('refresh-interval', 'n_intervals')
)
def update_dust_table(n):
    """Update dust ledger table"""
    dust = load_dust_ledger()

    if dust.empty:
        return html.P("No dust entries", style={'color': '#666', 'fontStyle': 'italic'})

    table_style = {
        'width': '100%',
        'borderCollapse': 'collapse',
        'backgroundColor': '#1a1a1a',
        'border': '1px solid #333'
    }

    th_style = {
        'padding': '10px',
        'textAlign': 'left',
        'borderBottom': '2px solid #ff6b6b',
        'color': '#ff6b6b',
        'fontSize': '13px'
    }

    td_style = {
        'padding': '8px',
        'borderBottom': '1px solid #333',
        'fontSize': '12px'
    }

    table_header = html.Thead(html.Tr([
        html.Th("Timestamp", style=th_style),
        html.Th("Symbol", style=th_style),
        html.Th("Remainder Qty", style=th_style),
        html.Th("Est. Cost ($)", style=th_style),
        html.Th("Reason", style=th_style)
    ]))

    rows = []
    for _, row in dust.iterrows():
        rows.append(html.Tr([
            html.Td(str(row.get('timestamp', 'N/A'))[:19], style=td_style),
            html.Td(row.get('symbol', 'N/A'), style=td_style),
            html.Td(f"{row.get('remainder_qty', 0):.8f}", style=td_style),
            html.Td(f"${row.get('est_cost_total_usdt', 0):.4f}", style=td_style),
            html.Td(row.get('reason', 'N/A'), style={**td_style, 'color': '#ff6b6b'})
        ]))

    table_body = html.Tbody(rows)

    return html.Table([table_header, table_body], style=table_style)


# ================= Main Entry Point =================

if __name__ == '__main__':
    print("=" * 60)
    print("Grid Trading Real-time Dashboard")
    print("=" * 60)
    print(f"[WEB] Opening dashboard at http://localhost:8050")
    print(f"[REFRESH] Auto-refresh: {REFRESH_INTERVAL / 1000}s")
    print(f"[WINDOW] Time window: {WINDOW_MINUTES} minutes")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")

    app.run(debug=True, host='127.0.0.1', port=8053, dev_tools_hot_reload=False)  # Debug mode ON
