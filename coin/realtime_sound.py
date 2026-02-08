# -*- coding: utf-8 -*-
"""
Bitcoin Real-time Chart with Price Alert (1-minute candles)
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import os

# Upbit API URL
UPBIT_MINUTE_URL = "https://api.upbit.com/v1/candles/minutes/1"  # 1-minute candle

# Alert settings
PRICE_DROP_THRESHOLD = -0.3  # -0.3% drop triggers alert
LOOKBACK_MINUTES = 5  # Compare with price 5 minutes ago

def play_alert_sound():
    """Play system alert sound 5 times"""
    for i in range(5):
        os.system('afplay /System/Library/Sounds/Sosumi.aiff')  # macOS alert sound

def fetch_minute_data(market="KRW-BTC", count=100):
    """Fetch 1-minute candle data from Upbit"""
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"market": market, "count": count}
    
    r = requests.get(UPBIT_MINUTE_URL, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    
    df = pd.DataFrame(data)
    df = df[["candle_date_time_kst", "trade_price"]].rename(columns={
        "candle_date_time_kst": "date",
        "trade_price": "close"
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def add_moving_averages(df):
    """Add moving averages (5min, 10min, 20min)"""
    df["MA5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["MA10"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["MA20"] = df["close"].rolling(window=20, min_periods=1).mean()
    return df

def check_price_drop(df, lookback=5, threshold=-0.3):
    """Check if price dropped more than threshold% from avg in last N minutes window"""
    if len(df) < lookback + 1:
        return False, 0, 0, 0, 0
    
    current_price = df['close'].iloc[-1]
    # Get average price in the last N minutes window
    window_avg = df['close'].iloc[-lookback-1:-1].mean()
    # Get max price in the last N minutes window
    window_max = df['close'].iloc[-lookback-1:-1].max()
    change_pct_avg = ((current_price - window_avg) / window_avg) * 100
    change_pct_max = ((current_price - window_max) / window_max) * 100
    
    return change_pct_avg <= threshold, change_pct_avg, window_avg, change_pct_max, window_max

# Initial data load
market = "KRW-BTC"
df = fetch_minute_data(market=market, count=100)
df = add_moving_averages(df)

# Chart settings
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6))

# Track last alert time to avoid spam
last_alert_time = None

def update(frame):
    """Real-time update function"""
    global df, last_alert_time
    
    try:
        # Fetch new 1-minute candle data
        df = fetch_minute_data(market=market, count=100)
        df = add_moving_averages(df)
        
        current_price = df['close'].iloc[-1]
        current_time = datetime.now()
        
        # Check for price drop alert (from avg in window)
        is_drop, change_pct_avg, window_avg, change_pct_max, window_max = check_price_drop(df, LOOKBACK_MINUTES, PRICE_DROP_THRESHOLD)
        
        alert_msg = ""
        if is_drop:
            # Only alert once per minute to avoid spam
            if last_alert_time is None or (current_time - last_alert_time).seconds >= 60:
                play_alert_sound()
                last_alert_time = current_time
                alert_msg = f" *** ALERT: {change_pct_avg:.2f}% from avg ***"
                print(f"[ALERT] Price dropped {change_pct_avg:.2f}% from {LOOKBACK_MINUTES}min window avg!")
        
        # Clear and redraw chart
        ax.clear()
        
        # Plot price and moving averages
        ax.plot(df["date"], df["close"], 'w.-', linewidth=1.5, markersize=3, label=f"Price: {current_price:,.0f} KRW")
        ax.plot(df["date"], df["MA5"], 'c-', linewidth=1.5, label=f"MA5: {df['MA5'].iloc[-1]:,.0f}")
        ax.plot(df["date"], df["MA10"], 'y-', linewidth=1.5, label=f"MA10: {df['MA10'].iloc[-1]:,.0f}")
        ax.plot(df["date"], df["MA20"], 'm-', linewidth=1.5, label=f"MA20: {df['MA20'].iloc[-1]:,.0f}")
        
        # Highlight current price
        ax.axhline(y=current_price, color='lime', linestyle='--', alpha=0.5, linewidth=1)
        ax.scatter([df["date"].iloc[-1]], [current_price], color='lime' if not is_drop else 'red', s=100, zorder=5)
        
        # 5-minute window avg and max price lines with percentage
        if len(df) >= LOOKBACK_MINUTES + 1:
            ax.axhline(y=window_avg, color='orange', linestyle=':', alpha=0.7, linewidth=1, label=f"5min Avg: {window_avg:,.0f}")
            ax.axhline(y=window_max, color='red', linestyle=':', alpha=0.7, linewidth=1, label=f"5min Max: {window_max:,.0f}")
            
            # Add percentage text annotations on the right side
            xlim = ax.get_xlim()
            x_pos = xlim[1]  # right edge
            
            # vs Avg percentage (orange)
            ax.annotate(f"vs Avg: {change_pct_avg:+.2f}%", 
                       xy=(x_pos, window_avg), 
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=10, color='orange', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            # vs Max percentage (red)
            ax.annotate(f"vs Max: {change_pct_max:+.2f}%", 
                       xy=(x_pos, window_max), 
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=10, color='red', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # Chart styling
        title_color = 'red' if is_drop else 'white'
        ax.set_title(f"Bitcoin (KRW-BTC) 1-min Chart | {current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                     f"vs Avg: {change_pct_avg:+.2f}% | vs Max: {change_pct_max:+.2f}%{alert_msg}", 
                     fontsize=14, color=title_color)
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Price (KRW)", fontsize=11)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Y-axis format
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # X-axis time format
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
    except Exception as e:
        print(f"Update error: {e}")

# Animation (update every 5 seconds)
ani = FuncAnimation(fig, update, interval=5000, cache_frame_data=False)

plt.show()
