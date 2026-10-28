import pandas as pd
from main import candles
candles_story = candles
df = pd.DataFrame(candles_story, columns=["open_time", "open", "high", "low", "close", "volume" ] )

