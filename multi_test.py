"""مقایسه استراتژی دانچیان روی چند ابزار و چند تایم‌فریم (۲ سال اخیر) -> multi_result.txt"""
import pandas as pd, yfinance as yf
import gold

# ابزار: (نماد یاهو، هزینه رفت‌وبرگشت به درصد قیمت — فرض تقریبی)
SYMBOLS = {
    "GC=F": 0.03, "SI=F": 0.05, "CL=F": 0.05,
    "EURUSD=X": 0.01, "GBPUSD=X": 0.01, "USDJPY=X": 0.01, "AUDUSD=X": 0.015, "USDCAD=X": 0.015,
    "BTC-USD": 0.10, "ETH-USD": 0.12,
    "^GSPC": 0.02, "AAPL": 0.03, "NVDA": 0.03, "TSLA": 0.04,
}
TFS = ["1h", "2h", "4h", "1d"]
MIN_TRADES = 30

def fetch(sym, tf):
    if tf == "1d":
        df = yf.download(sym, period="2y", interval="1d", auto_adjust=True, progress=False)
    else:
        df = yf.download(sym, period="730d", interval="1h", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if tf in ("2h", "4h"):
        df = df.resample(tf).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return df

def simulate(df, sig, cost_pct):
    O, H, L = df["Open"].values, df["High"].values, df["Low"].values
    b, s, a = sig["buy"].shift().fillna(False).values, sig["sell"].shift().fillna(False).values, sig["atr"].shift().values
    out, i, n = [], 1, len(df)
    while i < n:
        side = 1 if b[i] else (-1 if s[i] else 0)
        if side == 0 or a[i] != a[i] or a[i] <= 0: i += 1; continue
        e, risk = O[i], 2 * a[i]
        sl, tp = e - side * risk, e + side * 4 * a[i]
        r, j = None, i
        while j < n:
            hit_sl = L[j] <= sl if side == 1 else H[j] >= sl
            hit_tp = H[j] >= tp if side == 1 else L[j] <= tp
            if hit_sl: r = -1.0; break
            if hit_tp: r = 2.0; break
            j += 1
        if r is None: break
        out.append((i, r - e * cost_pct / 100 / risk)); i = j + 1
    return out

def main():
    rows, errs = [], []
    for sym, cost in SYMBOLS.items():
        for tf in TFS:
            try:
                df = fetch(sym, tf)
                if len(df) < 300: errs.append(f"{sym}@{tf}: داده کم"); continue
                tr = simulate(df, gold.signals_donchian(df), cost)
                mid = len(df) // 2
                a = [r for i, r in tr if i < mid]; b = [r for i, r in tr if i >= mid]
                allr = pd.Series([r for _, r in tr]) if tr else pd.Series(dtype=float)
                eq = allr.cumsum(); dd = float((eq - eq.cummax()).min()) if len(eq) else 0
                rows.append(dict(name=f"{sym}@{tf}", n=len(tr), win=(allr > 0).mean() * 100 if len(tr) else 0,
                                 avg=allr.mean() if len(tr) else 0, tot=allr.sum() if len(tr) else 0,
                                 a=pd.Series(a).mean() if a else float("nan"), b=pd.Series(b).mean() if b else float("nan"), dd=dd))
            except Exception as ex:
                errs.append(f"{sym}@{tf}: خطا {str(ex)[:60]}")
    for r in rows:
        r["ok"] = r["n"] >= MIN_TRADES and r["a"] > 0 and r["b"] > 0
    rows.sort(key=lambda r: (not r["ok"], -r["avg"]))
    print("استراتژی دانچیان ۲۰ + فیلتر روند | حد ضرر 2ATR | حد سود 4ATR | ۲ سال اخیر")
    print("هزینه معامله به درصد قیمت فرض شده است. نرخ برد سربه‌سر = 33٪")
    print(f"✅ = حداقل {MIN_TRADES} معامله و میانگین مثبت در هر دو نیمه\n")
    for r in rows:
        print(f"{'✅' if r['ok'] else '❌'} {r['name']:<14} {r['n']:4d} معامله | برد {r['win']:4.1f}% | میانگین {r['avg']:+.2f}R | "
              f"قدیم {r['a']:+.2f} / جدید {r['b']:+.2f} | مجموع {r['tot']:+.1f}R | بدترین افت {r['dd']:.1f}R")
    if errs: print("\nبدون نتیجه:\n" + "\n".join(errs))

if __name__ == "__main__":
    main()
