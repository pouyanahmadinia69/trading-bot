"""مانیتور طلا: روند + پولبک + برگشت مومنتوم (تایم‌فریم ۴ ساعته)
اجرا:  python gold.py            -> بررسی سیگنال و ارسال تلگرام
       python gold.py backtest   -> بک‌تست روی حدود ۲ سال داده"""
import os, sys, json, math, urllib.request, urllib.parse
import pandas as pd, yfinance as yf

SYMBOL = os.getenv("SYMBOL", "GC=F")
STATE = "gold_state.json"
RR = 2.0  # نسبت سود به ریسک
RISK_USD = float(os.getenv("RISK_USD", "1"))  # ریسک هر معامله (دلار)
PER_LOT = 1.0  # سود/ضرر دلاری به ازای هر ۱$ حرکت قیمت برای ۱.۰۰ لات (حساب سنتی XAUUSD_l)

def trim_incomplete(h1, df, now):
    """کندل ۴ ساعته ناقص را حذف می‌کند. اگر کندل تازه بسته شده (کمتر از ۱۵ دقیقه) ولی
    آخرین کندل ۱ ساعته‌اش هنوز در داده نیست، صبر می‌کنیم تا اجرای بعدی (داده کامل شود)."""
    h4 = pd.Timedelta(hours=4)
    if df.index[-1] + h4 > now:
        df = df.iloc[:-1]
    if len(df) and now < df.index[-1] + h4 + pd.Timedelta(minutes=15) and h1.index[-1] < df.index[-1] + pd.Timedelta(hours=3):
        df = df.iloc[:-1]
    return df

def load(period="730d"):
    df = yf.download(SYMBOL, period=period, interval="1h", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    h1 = df
    df = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return trim_incomplete(h1, df, pd.Timestamp.now(tz=df.index.tz))

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


def signals_donchian(df):
    """شکست سقف/کف ۲۰ کندل در جهت روند (آزمایش‌شده در gold_lab: بهترین نتیجه)"""
    h, l, c = df["High"], df["Low"], df["Close"]
    e50, e200 = (c.ewm(span=n, adjust=False).mean() for n in (50, 200))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    buy = (e50 > e200) & (c > e200) & (c > h.rolling(20).max().shift())
    sell = (e50 < e200) & (c < e200) & (c < l.rolling(20).min().shift())
    buy, sell = buy & ~buy.shift(fill_value=False), sell & ~sell.shift(fill_value=False)  # فقط اولین کندل شکست
    return pd.DataFrame({"buy": buy, "sell": sell, "atr": atr}, index=df.index)

def levels_donchian(df, sig, i):
    c, a = float(df["Close"].iloc[i]), float(sig["atr"].iloc[i])
    if sig["buy"].iloc[i]: return "BUY", c, c - 2 * a, c + 4 * a
    if sig["sell"].iloc[i]: return "SELL", c, c + 2 * a, c - 4 * a
    return None

def backtest(df):
    sig = signals_donchian(df); R = []; pos = None
    for i in range(len(df)):
        if pos:
            side, e, sl, tp = pos
            hi, lo = df["High"].iloc[i], df["Low"].iloc[i]
            hit_sl = lo <= sl if side == "BUY" else hi >= sl
            hit_tp = hi >= tp if side == "BUY" else lo <= tp
            if hit_sl: R.append(-1.0); pos = None   # اگر هر دو در یک کندل بخورد، ضرر حساب می‌شود (محافظه‌کارانه)
            elif hit_tp: R.append(RR); pos = None
            continue
        pos = levels_donchian(df, sig, i)
    if not R: print("معامله‌ای پیدا نشد"); return
    s = pd.Series(R); eq = s.cumsum()
    print(f"تعداد معامله: {len(s)} | نرخ برد: {(s>0).mean()*100:.1f}% | میانگین R: {s.mean():.2f} | مجموع R: {s.sum():.1f} | بدترین افت: {(eq-eq.cummax()).min():.1f}R")
    print("توجه: اسپرد، کمیسیون و اسلیپیج لحاظ نشده است.")

def lot_size(d):
    """حجم بر اساس ریسک ثابت؛ گرد به پایین روی گام ۰.۰۱"""
    lot = math.floor(RISK_USD / (d * PER_LOT) * 100 + 1e-9) / 100
    warn = ""
    if lot < 0.01:
        lot, warn = 0.01, "\n⚠️ حد ضرر دور است؛ حتی کمترین حجم (۰.۰۱) ریسک را از حد تعیین‌شده بالاتر می‌برد."
    return lot, lot * d * PER_LOT, warn

def send(text, photo=None):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    print(text)
    if not (tok and chat): return
    if photo:
        try:
            b = "----gold" + os.urandom(4).hex()
            parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                     for k, v in (("chat_id", chat), ("caption", text[:1000]))]
            parts.append(f'--{b}\r\nContent-Disposition: form-data; name="photo"; filename="chart.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
                         + open(photo, "rb").read() + b"\r\n")
            parts.append(f"--{b}--\r\n".encode())
            req = urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendPhoto", b"".join(parts),
                                         {"Content-Type": f"multipart/form-data; boundary={b}"})
            urllib.request.urlopen(req, timeout=30); return
        except Exception as ex:
            print("ارسال عکس ناموفق:", ex)
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data, timeout=20)

def main():
    df = load()
    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        return backtest(df)
    demo = os.getenv("DEMO", "").lower() == "true"
    sig = signals_donchian(df); r = levels_donchian(df, sig, -1)
    ts = str(df.index[-1])
    last = json.load(open(STATE)).get("ts") if os.path.exists(STATE) else None
    if demo and not r:  # سیگنال آزمایشی برای تست پیام و نمودار
        c, a = float(df["Close"].iloc[-1]), float(sig["atr"].iloc[-1])
        r = ("BUY", c, c - 2 * a, c + 4 * a)
    if r and (demo or ts != last):
        side, e, sl, tp = r
        d = abs(e - sl); lot, risk, warn = lot_size(d)
        icon = "🟢 خرید" if side == "BUY" else "🔴 فروش"
        text = (("🧪 تست (سیگنال واقعی نیست)\n" if demo else "") +
                f"{icon} طلا | کندل ۴ساعته\n"
                f"ورود (قیمت {SYMBOL}): حدود {e:.1f}\n"
                f"حد ضرر: {sl:.1f}  (فاصله {d:.1f}$)\n"
                f"حد سود: {tp:.1f}  (فاصله {2*d:.1f}$)\n"
                f"حجم: {lot:.2f} لات\n"
                f"ریسک: {risk:.2f}$ ({risk*100:.0f} سنت) | سود هدف: {2*risk:.2f}$\n"
                f"⚠️ قیمت بروکر چند دلار با {SYMBOL} فرق دارد؛ فاصله‌ها را از قیمت لحظه‌ای بروکر اعمال کنید.{warn}")
        photo = None
        try:
            import chart
            photo = chart.render_alert(df, sig, r, "alert.png")
        except Exception as ex:
            print("نمودار ساخته نشد:", ex)
        send(text, photo)
        if not demo:
            json.dump({"ts": ts}, open(STATE, "w"))
    else:
        print("سیگنالی نیست. آخرین کندل:", ts)

if __name__ == "__main__":
    main()
