import requests
import pandas as pd
def get_candles(symbol, interval, limit):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url,params=params, timeout=10)
    if response.status_code == 200:
        return response.json()[0:-1]
    else:
        print(f'Error:{response.status_code}')
        return None
    
def candles_to_df(candles):
    df = pd.DataFrame(candles)
    df = df[[0, 1, 2, 3, 4, 5]]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]

    price_cols = ["open", "high", "low", "close", "volume"]
    df[price_cols] = df[price_cols].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["is_green"] = df['close'] > df['open']

    return df

if __name__ == "__main__":
    candles = get_candles("BTCUSDT", "4h", 10)
    df = candles_to_df(candles)
    print(df)
    print(df[df["is_green"]])