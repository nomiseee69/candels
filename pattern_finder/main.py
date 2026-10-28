from market import get_candles

candles = get_candles("BTCUSDT", "4h", 5)
print(f"Полученно закрытых свечей: {len(candles)}")
