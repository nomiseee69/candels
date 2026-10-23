import requests
from datetime import datetime

def get_candles(symbol, interval, limit):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url,params=params, timeout=10)
    if response.status_code == 200:
        return response.json()[0:-1]
    else:
        print(f'Error:{response.status_code}')
        return None
if __name__ == '__main__':
    pairs = ["BTCUSDT", "ETHUSDT"]
    for pair in pairs:
        print(f"Пара монет {pair}")
        candles = get_candles(pair, "4h", 10)
        if  not candles:
            continue
        for candle in candles:
            open_time = datetime.fromtimestamp(candle[0]/1000)
            open_price = float(candle[1])
            close_price = float(candle[4])
            if open_price > close_price:
                color= "🔴"
            else:
                color ="🟢"
            print(f'{open_time.strftime("%Y-%m-%d %H:%M")} | O: {open_price} C: {close_price} {color}')