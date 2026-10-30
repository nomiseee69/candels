from shape import photo_to_shape
from market import get_candles, candles_to_df
from matcher import find_best_match_multi
from chart import plot_match
from notifier import send_photo, tradingview_link
from scanner import scan_market, send_results
from notifier import CHAT_ID, send_message

good = scan_market("w_shape.png")
if not good:
    send_message("Свежих совпадений нет")
else:
    send_results(good, CHAT_ID)