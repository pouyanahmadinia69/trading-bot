"""رسم نمودار ۴ ساعته طلا با EMA50/200، کانال دانچیان ۲۰ و سیگنال‌های استراتژی -> gold_chart.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gold

N = 220  # تعداد کندل نمایش‌داده‌شده

def main():
    df = gold.load()
    c, h, l, o = df["Close"], df["High"], df["Low"], df["Open"]
    e50, e200 = (c.ewm(span=n, adjust=False).mean() for n in (50, 200))
    dh, dl = h.rolling(20).max().shift(), l.rolling(20).min().shift()
    sig = gold.signals_donchian(df)
    d = df.iloc[-N:]; x = np.arange(len(d)); k = len(df) - N
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=130)
    up = d["Close"] >= d["Open"]
    ax.vlines(x, d["Low"], d["High"], color=np.where(up, "#1a9e5c", "#d6453d"), lw=1)
    ax.bar(x, (d["Close"] - d["Open"]).abs(), bottom=np.minimum(d["Open"], d["Close"]), width=0.65,
           color=np.where(up, "#1a9e5c", "#d6453d"))
    ax.plot(x, e50.iloc[k:], color="#f5a623", lw=1.6, label="EMA 50")
    ax.plot(x, e200.iloc[k:], color="#3b6fd4", lw=1.8, label="EMA 200")
    ax.plot(x, dh.iloc[k:], color="#7a7a7a", lw=1, ls="--", label="Donchian 20 (high/low)")
    ax.plot(x, dl.iloc[k:], color="#7a7a7a", lw=1, ls="--")
    ax.fill_between(x, dl.iloc[k:], dh.iloc[k:], color="#999999", alpha=0.08)
    pad = (d["High"].max() - d["Low"].min()) * 0.03
    for i in x:
        if sig["buy"].iloc[k + i]:
            ax.scatter(i, d["Low"].iloc[i] - pad, marker="^", s=140, color="#00b050", zorder=5)
        if sig["sell"].iloc[k + i]:
            ax.scatter(i, d["High"].iloc[i] + pad, marker="v", s=140, color="#e02020", zorder=5)
    idx = np.where((sig["buy"] | sig["sell"]).iloc[k:].values)[0]
    if len(idx):
        j = idx[-1]; r = gold.levels_donchian(df, sig, k + j); side, e, sl, tp = r
        ax.hlines([sl, tp], j, len(d) + 2, colors=["#e02020", "#00b050"], linestyles=":", lw=1.4)
        ax.text(len(d) + 2.5, sl, f"SL {sl:.0f}", va="center", fontsize=9, color="#e02020")
        ax.text(len(d) + 2.5, tp, f"TP {tp:.0f}", va="center", fontsize=9, color="#00b050")
    step = max(1, len(d) // 8)
    ax.set_xticks(x[::step]); ax.set_xticklabels([t.strftime("%m-%d %H:%M") for t in d.index[::step]], fontsize=8)
    ax.set_xlim(-1, len(d) + 9); ax.grid(alpha=0.2)
    ax.set_title(f"Gold {gold.SYMBOL} 4H  |  Donchian breakout strategy  |  last candle {d.index[-1].strftime('%Y-%m-%d %H:%M')} (New York time)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig("gold_chart.png"); print("saved")

if __name__ == "__main__":
    main()
