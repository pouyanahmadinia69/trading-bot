"""آزمایشگاه استراتژی طلا: چند استراتژی را روی ۲ سال داده می‌سنجد.
داده به دو نیمه تقسیم می‌شود (قدیمی = آموزش، جدید = آزمون). فقط استراتژی‌ای که در هر دو نیمه مثبت باشد قابل اعتناست."""
import pandas as pd, yfinance as yf
import gold

COST = 0.5  # هزینه رفت‌وبرگشت (اسپرد+کمیسیون) به دلار برای هر اونس

def load_1h():
    df = yf.download(gold.SYMBOL, period="730d", interval="1h", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()

def to_4h(df):
    return df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()

def build(df):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    e20, e50, e200 = (c.ewm(span=n, adjust=False).mean() for n in (20, 50, 200))
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    m, s = c.rolling(20).mean(), c.rolling(20).std()
    bu, bl = m + 2 * s, m - 2 * s
    dh, dl = h.rolling(20).max().shift(), l.rolling(20).min().shift()
    pb = gold.signals(df)
    up_t, dn_t = (e50 > e200) & (c > e200), (e50 < e200) & (c < e200)
    S = {
        "pullback":   (pb["buy"], pb["sell"], 1.5, 3.0),
        "donchian":   (up_t & (c > dh), dn_t & (c < dl), 2.0, 4.0),
        "ema_cross":  (up_t & (e20 > e50) & (e20.shift() <= e50.shift()), dn_t & (e20 < e50) & (e20.shift() >= e50.shift()), 1.5, 3.0),
        "rsi_revert": (up_t & (rsi < 30), dn_t & (rsi > 70), 2.0, 2.0),
        "bollinger":  (up_t & (c > bu) & (c.shift() <= bu.shift()), dn_t & (c < bl) & (c.shift() >= bl.shift()), 1.5, 3.0),
    }
    return S, atr

def simulate(df, buy, sell, atr, slm, tpm):
    O, H, L = df["Open"].values, df["High"].values, df["Low"].values
    b, s, a = buy.shift().fillna(False).values, sell.shift().fillna(False).values, atr.shift().values
    out = []; i, n = 1, len(df)
    while i < n:
        side = 1 if b[i] else (-1 if s[i] else 0)
        if side == 0 or a[i] != a[i]: i += 1; continue
        e, risk = O[i], slm * a[i]
        sl, tp = e - side * risk, e + side * tpm * a[i]
        r = None; j = i
        while j < n:
            hit_sl = L[j] <= sl if side == 1 else H[j] >= sl
            hit_tp = H[j] >= tp if side == 1 else L[j] <= tp
            if hit_sl: r = -1.0; break
            if hit_tp: r = tpm / slm; break
            j += 1
        if r is None: break
        out.append((i, r - COST / risk)); i = j + 1
    return out

def stats(rs):
    if not rs: return "  0 معامله"
    s = pd.Series(rs)
    return f"{len(s):3d} معامله | برد {(s>0).mean()*100:4.1f}% | میانگین {s.mean():+.2f}R | مجموع {s.sum():+.1f}R"

def main():
    d1 = load_1h()
    rows = []
    for tf, df in (("1h", d1), ("4h", to_4h(d1))):
        S, atr = build(df); mid = len(df) // 2
        for name, (buy, sell, slm, tpm) in S.items():
            tr = simulate(df, buy, sell, atr, slm, tpm)
            a = [r for i, r in tr if i < mid]; b = [r for i, r in tr if i >= mid]
            ok = bool(a) and bool(b) and pd.Series(a).mean() > 0 and pd.Series(b).mean() > 0
            rows.append((f"{name}@{tf}", a, b, ok))
    print(f"هزینه فرض‌شده: {COST}$ برای هر معامله\n")
    for name, a, b, ok in sorted(rows, key=lambda x: (not x[3], -(pd.Series(x[2]).mean() if x[2] else -9))):
        print(("✅ " if ok else "❌ ") + name)
        print("   نیمه قدیم:", stats(a)); print("   نیمه جدید:", stats(b))
    print("\n✅ یعنی در هر دو نیمه سودده بوده. هنوز تضمین آینده نیست.")

if __name__ == "__main__":
    main()
