"""مانیتور طلا: روند + پولبک + برگشت مومنتوم (تایم‌فریم ۴ ساعته)
اجرا:  python gold.py            -> بررسی سیگنال و ارسال تلگرام
       python gold.py backtest   -> بک‌تست روی حدود ۲ سال داده"""
import os, sys, json, urllib.request, urllib.parse
import pandas as pd, yfinance as yf

SYMBOL = os.getenv("SYMBOL", "GC=F")
STATE = "gold_state.json"
RR = 2.0  # نسبت سود به ریسک

def load(period="730d"):
    df = yf.download(SYMBOL, period=period, interval="1h", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    if df.index[-1] + pd.Timedelta(hours=4) > pd.Timestamp.now(tz=df.index.tz):
        df = df.iloc[:-1]  # کندل ناقص حذف شود
    return df

def signals(df):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    ema20, ema50, ema200 = (c.ewm(span=n, adjust=False).mean() for n in (20, 50, 200))
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    swl, swh = l.rolling(6).min(), h.rolling(6).max()
    sl_b, sl_s = swl - 0.25 * atr, swh + 0.25 * atr
    buy = (ema50 > ema200) & (c > ema200) & (swl <= ema20) & (rsi.shift() < 50) & (rsi >= 50) & (c > o) \
          & (c - sl_b > 0) & (c - sl_b < 3 * atr)
    sell = (ema50 < ema200) & (c < ema200) & (swh >= ema20) & (rsi.shift() > 50) & (rsi <= 50) & (c < o) \
           & (sl_s - c > 0) & (sl_s - c < 3 * atr)
    out = pd.DataFrame({"buy": buy, "sell": sell, "sl_b": sl_b, "sl_s": sl_s}, index=df.index)
    return out

def levels(df, sig, i):
    c = float(df["Close"].iloc[i])
    if sig["buy"].iloc[i]:
        sl = float(sig["sl_b"].iloc[i]); return "BUY", c, sl, c + RR * (c - sl)
    if sig["sell"].iloc[i]:
        sl = float(sig["sl_s"].iloc[i]); return "SELL", c, sl, c - RR * (sl - c)
    return None

def backtest(df):
    sig = signals(df); R = []; pos = None
    for i in range(len(df)):
        if pos:
            side, e, sl, tp = pos
            hi, lo = df["High"].iloc[i], df["Low"].iloc[i]
            hit_sl = lo <= sl if side == "BUY" else hi >= sl
            hit_tp = hi >= tp if side == "BUY" else lo <= tp
            if hit_sl: R.append(-1.0); pos = None   # اگر هر دو در یک کندل بخورد، ضرر حساب می‌شود (محافظه‌کارانه)
            elif hit_tp: R.append(RR); pos = None
            continue
        pos = levels(df, sig, i)
    if not R: print("معامله‌ای پیدا نشد"); return
    s = pd.Series(R); eq = s.cumsum()
    print(f"تعداد معامله: {len(s)} | نرخ برد: {(s>0).mean()*100:.1f}% | میانگین R: {s.mean():.2f} | مجموع R: {s.sum():.1f} | بدترین افت: {(eq-eq.cummax()).min():.1f}R")
    print("توجه: اسپرد، کمیسیون و اسلیپیج لحاظ نشده است.")

def notify(text):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    print(text)
    if tok and chat:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data, timeout=20)

def main():
    df = load()
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        return backtest(df)
    sig = signals(df); r = levels(df, sig, -1)
    ts = str(df.index[-1])
    last = json.load(open(STATE)).get("ts") if os.path.exists(STATE) else None
    if r and ts != last:
        side, e, sl, tp = r
        icon = "🟢 خرید" if side == "BUY" else "🔴 فروش"
        notify(f"{icon} طلا ({SYMBOL})\nورود: {e:.1f}\nحد ضرر: {sl:.1f}\nحد سود: {tp:.1f}\nریسک به سود: 1:{RR:.0f}\nکندل ۴ساعته: {ts}")
        json.dump({"ts": ts}, open(STATE, "w"))
    else:
        print("سیگنالی نیست. آخرین کندل:", ts)

if __name__ == "__main__":
    main()
