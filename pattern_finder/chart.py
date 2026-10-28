import matplotlib.pyplot as plt


def plot_close(df,symbol,path):
    plt.plot(df['open_time'], df['close'], label=symbol)
    plt.xlabel('Время (UTC)')
    plt.ylabel('price')
    low_index = df['close'].idxmin()
    plt.scatter(df.loc[low_index, 'open_time'], df.loc[low_index, 'low'], color='red', s=100)
    plt.legend()
    plt.title(f'{symbol} PRICE')
    plt.savefig(path)
    plt.close()
if __name__ == "__main__":
    from market import get_candles, candles_to_df

    candles = get_candles("BTCUSDT", "4h", 100)
    df = candles_to_df(candles)
    plot_close(df, "BTCUSDT", "btc.png")

