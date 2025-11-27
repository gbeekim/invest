# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 22:05:16 2025

@author: kim
"""

# pip install requests pandas matplotlib

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

UPBIT_URL = "https://api.upbit.com/v1/candles/days"
# %%
def fetch_upbit_days(market="KRW-BTC", limit=1000):
    """
    Upbit 일봉을 여러 번 호출해서 과거 데이터까지 불러오기
    limit: 전체 가져올 최대 개수 (200단위로 쪼개서 반복)
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    all_data = []
    to = None  # 처음엔 최신부터

    while len(all_data) < limit:
        count = min(200, limit - len(all_data))
        params = {"market": market, "count": count}
        if to:
            params["to"] = to.isoformat().replace("+00:00", "Z")  # UTC ISO 형식

        r = requests.get(UPBIT_URL, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        all_data.extend(data)

        # 다음 요청용: 가장 오래된 캔들의 UTC 시간 - 1초
        oldest = datetime.fromisoformat(data[-1]["candle_date_time_utc"].replace("Z", "+00:00"))
        to = oldest - timedelta(seconds=1)

        if len(data) < count:
            break  # 더 이상 데이터 없음

    df = pd.DataFrame(all_data)
    df = df.rename(columns={
        "candle_date_time_kst": "date_kst",
        "trade_price": "close",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "candle_acc_trade_volume": "volume"
    })
    df["date_kst"] = pd.to_datetime(df["date_kst"])
    df = df.sort_values("date_kst").reset_index(drop=True)
    return df
# %%
def fetch_upbit_daily(market="KRW-BTC", count=200):
    """
    업비트 일봉 캔들 가져오기 (최신→과거 순으로 옴)
    반환: pandas.DataFrame (오래된→최신으로 정렬)
    """
    # ,time_end = (datetime.now(timezone.utc)).date()
    # market="KRW-BTC"
    # count=200
    headers = {"User-Agent": "Mozilla/5.0"}
    # params = {"market": market, "count": count,"to":time_end}
    params = {"market": market, "count": count}
    
    r = requests.get(UPBIT_URL, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()  # 최신이 0번
    df = pd.DataFrame(data)

    # 관심 열 정리
    df = df[[
        "candle_date_time_kst",'candle_date_time_utc', "opening_price", "high_price",
        "low_price", "trade_price", "candle_acc_trade_volume"
    ]].rename(columns={
        "candle_date_time_kst": "date_kst",
        'candle_date_time_utc': 'date_utc',
        "trade_price": "close",
        "candle_acc_trade_volume": "volume"
    })

    # 날짜형 변환 및 오래된→최신 정렬
    df["date_kst"] = pd.to_datetime(df["date_kst"])
    df["date_utc"] = pd.to_datetime(df["date_utc"])
    df = df.sort_values("date_kst").reset_index(drop=True)
    return df

def add_sma(df, windows=(5, 10, 15, 20), price_col="close"):
    """단순이동평균 컬럼 추가"""
    for w in windows:
        df[f"mov{w}"] = df[price_col].rolling(window=w, min_periods=1).mean()
    return df
# %%
# df['close'].iloc[:5].mean()
# df['mov5'].iloc[:5].mean()
# %%
# %%

'''
    mov5의 경우 1~5일까지 close의 평균값
    매수: mov5가 상승이면서 mov 10 혹은 mov 15와 역전
    매도: mov5가 하강이면서 mov 10 역전
    다만 최초 포인트에서만 매수하고 매도해야됨
    실시간으로 결과를 정하면 안됨 하루전 시그널을 이용해서 오늘 살지 말지 정해야됨
'''

def calc_trade_point(df):
    col_tot = df.columns.tolist()
    col_mov = [cnow for cnow in col_tot if 'mov' in cnow]
    df['cdiff'] = df['close'].diff()
    for i in range(len(col_mov)):#i=0
        cnow = col_mov[i]
        ynow = cnow.split('mov')[-1]
        mdiff = ynow+'diff'
        cdiff = 'c'+ynow+'_diff'
        cdiff2 = '2c'+ynow+'_diff'
        df[mdiff] = df[cnow].diff()
        df[cdiff] = df['mov5'] - df[cnow]
        df[cdiff2] = df[cdiff].shift(1)
        # df[cdiff] = df['mov15'] - df[cnow]b

        # df[cnow].iloc[1] - df[cnow].iloc[0]
    # df['c5_diff'] = df['close'] - df['mov5']
    return df
plt.close('all')
def get_flag(df):
    df['buy_flag'] = np.nan
    df.loc[(df['5diff']>=0)&(df['2c10_diff']<=0)&(df['c10_diff']>=0),'buy_flag'] = 1
    # df.loc[(df['cdiff']>=0)&(df['5diff']>=0)&(df['2c10_diff']<=0)&(df['c10_diff']>=0),'buy_flag'] = 1
    df.loc[(df['5diff']<=0)&(df['2c10_diff']>=0)&(df['c10_diff']<=0),'buy_flag'] = -1
    df['buy_flag'] = df['buy_flag'].fillna(method='ffill')
    df['bf_diff'] = df['buy_flag'].diff()
    
    df['cyc_cnt'] = 0
    df.loc[df['bf_diff']==2,'cyc_cnt']=1
    df['cyc_cnt'] = df['cyc_cnt'].cumsum()
    
    
    return df




def get_profit(df):
    df['oind'] = df.index.tolist()
    f_ind = df.loc[df['buy_flag']==1,:].groupby('cyc_cnt').first()['oind']
    l_ind = df.loc[df['buy_flag']==1,:].groupby('cyc_cnt').last()['oind']
    df['trade_flag'] = np.nan
    df.loc[f_ind,'trade_flag'] = 1
    df.loc[l_ind,'trade_flag'] = -1
    df_first = df.loc[df['buy_flag']==1,:].groupby('cyc_cnt').first().reset_index().copy()
    df_first = df_first.rename(columns={'buy_flag':'buy_flag_f','close':'close_f','mov5':'mov5_f','mov10':'mov10_f'})
    df_last = df.loc[df['buy_flag']==1,:].groupby('cyc_cnt').last().reset_index().copy()
    df_last = df_last.rename(columns={'buy_flag':'buy_flag_l','close':'close_l','mov5':'mov5_l','mov10':'mov10_l'})
    
    df_res = pd.merge(df_first[['cyc_cnt','buy_flag_f','close_f','mov5_f','mov10_f']],
                  df_last[['cyc_cnt','buy_flag_l','close_l','mov5_l','mov10_l']],on='cyc_cnt')
    df_res['profit'] = df_res['close_l'] - df_res['close_f']
    df_res['profit_rate'] = df_res['profit'] / df_res['close_f']
    
    return df,df_res
# %%

# if __name__ == "__main__":
market = "KRW-BTC"
# market = "KRW-ETH"
# df = fetch_upbit_daily(market=market, count=200)  # 필요 시 더 늘릴 수 있음(최대 200)
df = fetch_upbit_days(market=market, limit=2000)  # 필요 시 더 늘릴 수 있음(최대 200)
col_tot = df.columns.tolist()
df[['open','high','low','close']] = df[['open','high','low','close']]/1000000
df = add_sma(df, windows=(5, 10, 15,20))

df = calc_trade_point(df)
df = get_flag(df)
df, df_res = get_profit(df)

# 최신 값 출력
last = df.iloc[-1]
print(f"[{market}] {last['date_kst'].date()} 종가: {last['close']:,.0f} KRW")
# print(f"  SMA5 : {last['SMA5']:,.0f}  | SMA10: {last['SMA10']:,.0f}  | SMA20: {last['SMA20']:,.0f}")
# %%



# %%
# 간단 차트 (원하면 주석 해제)
%matplotlib qt5
# plt.figure(figsize=(11,5))
plt.figure()
a1 = plt.subplot(2,1,1)
a11 = a1.twinx()
a1.plot(df["date_kst"], df["close"],'.-', label="Close")
a1.plot(df["date_kst"], df["mov5"], label="mov 5")
a1.plot(df["date_kst"], df["mov10"], label="mov 10")
# plt.plot(df["date_kst"], df["mov15"], label="mov 15")
# plt.plot(df["date_kst"], df["mov20"], label="mov 20")
a11.plot(df["date_kst"], df["buy_flag"], label="buy flag")
# plt.title(f"{market} 일봉 & 5/10/20일 이동평균 (Upbit)")
lines_1, labels_1 = a1.get_legend_handles_labels()
lines_11, labels_11 = a11.get_legend_handles_labels()


plt.xlabel("Date (KST)")
plt.ylabel("Price (KRW)")
a1.legend(lines_1 + lines_11, labels_1 + labels_11, loc='upper left')
plt.grid(True, alpha=0.3)
a2=plt.subplot(2,1,2,sharex=a1)
a2.plot(df['date_kst'],df['c5_diff'],'.:')
a2.plot(df['date_kst'],df['c10_diff'],'.:')
a2.plot(df['date_kst'],df['c15_diff'],'.:')
a22 = a2.twinx()
a22.plot(df['date_kst'],df['5diff'],'.-')
a22.plot(df['date_kst'],df['10diff'],'.-')
a22.plot(df['date_kst'],df['15diff'],'.-')
plt.grid()
plt.suptitle(market)
plt.tight_layout()
# %%
df['cdiff'] = df['close']
df['bf_diff'] = df['buy_flag'].diff()
# plt.figure()
plt.figure()
a1 = plt.subplot(2,1,1)
plt.plot(df["date_kst"], df["close"],'.-', label="Close")
plt.plot(df["date_kst"], df["mov5"], '.-',label="mov 5")
plt.plot(df["date_kst"], df["mov10"], label="mov 10")
plt.plot(df["date_kst"], df["mov15"], label="mov 15")
plt.plot(df["date_kst"], df["mov20"], label="mov 20")
plt.legend()
plt.grid()
plt.subplot(2,1,2,sharex=a1)
plt.plot(df['date_kst'],df['buy_flag'],'.-')
plt.plot(df['date_kst'],df['bf_diff'],'.')
plt.grid()
# %% get cyc


# %%
plt.figure()
a1 = plt.subplot(1,1,1)
a1.plot(df['date_kst'],df['buy_flag'],'.-')
a11 = a1.twinx()
a11.plot(df['date_kst'],df['cyc_cnt'],'.')
# %%
df_first = df.loc[df['buy_flag']==1,:].groupby('cyc_cnt').first().reset_index().copy()
df_first = df_first.rename(columns={'buy_flag':'buy_flag_f','close':'close_f','mov5':'mov5_f','mov10':'mov10_f'})
df_last = df.loc[df['buy_flag']==-1,:].groupby('cyc_cnt').first().reset_index().copy()
df_last = df_last.rename(columns={'buy_flag':'buy_flag_l','close':'close_l','mov5':'mov5_l','mov10':'mov10_l'})

df_res = pd.merge(df_first[['cyc_cnt','buy_flag_f','close_f','mov5_f','mov10_f']],
              df_last[['date_kst','cyc_cnt','buy_flag_l','close_l','mov5_l','mov10_l']],on='cyc_cnt')
df_res['profit'] = df_res['close_l'] - df_res['close_f']
df_res['profit_rate'] = df_res['profit'] / df_res['close_f']
# %%
plt.figure()
plt.plot(df_res['cyc_cnt'],df_res['profit_rate'],'.-')
plt.grid()
plt.title(df_res['profit_rate'].sum())
# %%
f_ind = df.loc[df['buy_flag']==1,:].groupby('cyc_cnt').head(1).index
t_ind = df.loc[df['buy_flag']==-1,:].groupby('cyc_cnt').head(1).index
df['flag'] = 0
df.loc[f_ind,'flag'] =1
df.loc[t_ind,'flag'] =-1
df_buy = df.loc[f_ind,:].copy()
df_sell = df.loc[t_ind,:].copy()
# %%
# %matplotlib qt5
plt.figure()
a1 = plt.subplot(2,1,1)
plt.plot(df["date_kst"], df["close"],'.-', label="Close")
plt.plot(df["date_kst"], df["mov5"], label="mov 5")
plt.plot(df["date_kst"], df["mov10"], label="mov 10")
plt.plot(df["date_kst"], df["mov15"], label="mov 15")
plt.plot(df["date_kst"], df["mov20"], label="mov 20")
plt.plot(df_buy['date_kst'],df_buy['close'],'c*',label='buy')
plt.plot(df_sell['date_kst'],df_sell['close'],'r*',label='sell')
plt.legend()
plt.grid()
plt.subplot(2,1,2,sharex=a1)
plt.plot(df_res['date_kst'],df_res['profit_rate'],'.')
plt.grid()


# %%
plt.figure()
a1 = plt.subplot(2,1,1)
plt.plot(df["date_kst"], df["close"],'.-', label="Close")
plt.plot(df["date_kst"], df["mov5"], label="mov 5")
plt.plot(df["date_kst"], df["mov10"], label="mov 10")
plt.plot(df["date_kst"], df["mov15"], label="mov 15")
plt.plot(df["date_kst"], df["mov20"], label="mov 20")
# plt.plot(df.loc[df['trade_flag']==1,'date_kst'],df.loc[df['trade_flag']==1,'close'],'c*',label='buy')
# plt.plot(df.loc[df['trade_flag']==-1,'date_kst'],df.loc[df['trade_flag']==-1,'close'],'r*',label='sell')
# plt.grid()

plt.plot(df_buy['date_kst'],df_buy['close'],'c*',label='buy')
plt.plot(df_sell['date_kst'],df_sell['close'],'r*',label='sell')
plt.legend()
plt.grid()
plt.subplot(2,1,2,sharex=a1)
plt.plot(df_res['date_kst'],df_res['profit_rate'],'.')
plt.grid()

# %%
df_res.shape
df_res['profit_rate'].sum()
