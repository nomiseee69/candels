import matplotlib.pyplot as plt


def plot_close(df,symbol,path,minima,maxima):
    plt.plot(df['open_time'], df['close'], label=symbol)
    plt.plot(df['open_time'], df['low'], linewidth=0.8, label="low")
    plt.plot(df['open_time'], df['high'], linewidth=0.8, label="high")
    plt.xlabel('Время (UTC)')
    plt.ylabel('price')
    plt.scatter(df.loc[maxima, "open_time"], df.loc[maxima, "high"], color='green', s=100)
    plt.scatter(df.loc[minima, 'open_time'], df.loc[minima,'low'], color='red', s=100)
    plt.legend()
    plt.title(f'{symbol} PRICE')
    plt.savefig(path)
    plt.close()
if __name__ == "__main__":
    from market import get_candles, candles_to_df
    from extremes import find_local_minima,find_local_maxima

    candles = get_candles("BTCUSDT", "4h", 100)
    df = candles_to_df(candles)
    minima = find_local_minima(df["low"].tolist(), 5)
    maxima = find_local_maxima(df["high"].tolist(), 5)
    print("Минимумы на свечах:", minima)
    plot_close(df, "BTCUSDT", "btc.png", minima,maxima)

