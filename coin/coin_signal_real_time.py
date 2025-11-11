# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 21:44:59 2025

@author: kim
"""

# pip install websocket-client
import json
import websocket

WS_URL = "wss://api.upbit.com/websocket/v1"

def stream_upbit_ticker():
    ws = websocket.create_connection(WS_URL, header=["User-Agent: Mozilla/5.0"])
    # 업비트는 리스트(JSON Array) 메시지를 보냄
    sub_msg = [
        {"ticket": "test-ticket"},
        {"type": "ticker", "codes": ["KRW-BTC"]},
        # {"format": "SIMPLE"}  # 원하면 간단 포맷
    ]
    ws.send(json.dumps(sub_msg))
    try:
        cnt=0
        while cnt<5:
            # 업비트는 바이너리 프레임으로 보낼 수 있어 decode 필요
            raw = ws.recv()
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
            price = data.get("tp") or data.get("trade_price")  # SIMPLE 사용 시 tp
            chg_rt = data.get("cr") or data.get("change_rate")
            print(f"[{data.get('code','KRW-BTC')}] {price:,.0f} KRW (24h {chg_rt*100:+.2f}%)")
            cnt=cnt+1
        return data
    finally:
        ws.close()
# %%
if __name__ == "__main__":
    df = stream_upbit_ticker()
