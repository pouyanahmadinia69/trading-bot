"""نمودار طلا (۴ ساعته) با EMA50/200، دانچیان ۲۰ و نتیجه هر معامله (TP یا SL)
خروجی: gold_chart.png (آخرین ۲۲۰ کندل)  و  gold_2y.png (۲ سال)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import gold

G, R_, B, Y, GR = "#00a650", "#e02020", "#3b6fd4", "#f5a623", "#7a7a7a"

def find_trades(df, sig):
    H, L = df["High"].values, df["Low"].values
    out, i, n = [], 0, len(df)
    while i < n:
        if not (sig["buy"].iloc[i] or sig["sell"].iloc[i]): i += 1; continue
        side, e, sl, tp = gold.levels_donchian(df, sig, i)
        res, j = "OPEN", i + 1
        while j < n:
            hit_sl = L[j] <= sl if side == "BUY" else H[j] >= sl
            hit_tp = H[j] >= tp if side == "BUY" else L[j] <= tp
            if hit_sl: res = "SL"; break   # اگر هر دو در یک کندل بخورد، SL حساب می‌شود
            if hit_tp: res = "TP"; break
            j += 1
        out.append(dict(i=i, j=min(j, n - 1), side=side, e=e, sl=sl, tp=tp, res=res))
        if res == "OPEN": break
        i = j + 1
    return out

def draw_trades(ax, trades, k, n, ms):
    for t in trades:
        if t["j"] < k or t["i"] >= k + n: continue
        xi, xj = t["i"] - k, t["j"] - k
        col = {"TP": G, "SL": R_, "OPEN": GR}[t["res"]]
        ex = {"TP": t["tp"], "SL": t["sl"], "OPEN": None}[t["res"]]
        if ex is not None:
            ax.plot([xi, xj], [t["e"], ex], color=col, lw=2.2, zorder=4)
            ax.scatter(xj, ex, marker="o" if t["res"] == "TP" else "X", s=ms * 0.9, color=col, edgecolor="white", zorder=6)
        else:
            ax.hlines([t["sl"], t["tp"]], xi, n + 2, colors=[R_, G], linestyles=":", lw=1.4)
        ax.scatter(xi, t["e"], marker="^" if t["side"] == "BUY" else "v", s=ms, color=GR, edgecolor="black", zorder=5)

def legend(ax):
    h = [Line2D([], [], color=Y, lw=1.6, label="EMA 50"), Line2D([], [], color=B, lw=1.8, label="EMA 200"),
         Line2D([], [], color=GR, lw=1, ls="--", label="Donchian 20"),
         Line2D([], [], marker="^", color=GR, lw=0, mec="black", label="Entry buy (▼ = sell)"),
         Line2D([], [], marker="o", color=G, lw=0, label="Hit TP (profit)"),
         Line2D([], [], marker="X", color=R_, lw=0, label="Hit SL (stop loss)")]
    ax.legend(handles=h, loc="upper left", fontsize=9)

def summary(trades):
    tp = sum(t["res"] == "TP" for t in trades); sl = sum(t["res"] == "SL" for t in trades)
    n = tp + sl
    return f"{n} closed trades | TP {tp} | SL {sl} | win rate {tp/n*100 if n else 0:.0f}% | total {2*tp - sl:+d}R (before costs)"

def render_alert(df, sig, r, path, bars=110):
    """نمودار ۴ ساعته برای پیام تلگرام: ورود، SL، TP + EMA و دانچیان"""
    side, e, sl, tp = r
    c, h, l = df["Close"], df["High"], df["Low"]
    e50, e200 = (c.ewm(span=n, adjust=False).mean() for n in (50, 200))
    dh, dl = h.rolling(20).max().shift(), l.rolling(20).min().shift()
    N = min(bars, len(df)); k = len(df) - N; d = df.iloc[k:]; x = np.arange(N)
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=110)
    up = (d["Close"] >= d["Open"]).values; cc = np.where(up, "#1a9e5c", "#d6453d")
    ax.vlines(x, d["Low"], d["High"], color=cc, lw=1)
    ax.bar(x, (d["Close"] - d["Open"]).abs(), bottom=np.minimum(d["Open"], d["Close"]), width=0.65, color=cc)
    ax.plot(x, e50.iloc[k:], color=Y, lw=1.6, label="EMA 50"); ax.plot(x, e200.iloc[k:], color=B, lw=1.8, label="EMA 200")
    ax.plot(x, dh.iloc[k:], color=GR, lw=1, ls="--", label="Donchian 20"); ax.plot(x, dl.iloc[k:], color=GR, lw=1, ls="--")
    ax.scatter(N - 1, e, marker="^" if side == "BUY" else "v", s=220, color="#222", zorder=6)
    for v, col, nm in ((e, "#222", "Entry"), (sl, R_, "SL"), (tp, G, "TP")):
        ax.hlines(v, N - 1, N + 8, colors=col, linestyles="-" if nm == "Entry" else ":", lw=1.6)
        ax.text(N + 8.5, v, f"{nm} {v:.1f}", va="center", fontsize=9, color=col)
    step = max(1, N // 7)
    ax.set_xticks(x[::step]); ax.set_xticklabels([t.strftime("%m-%d %H:%M") for t in d.index[::step]], fontsize=8)
    ax.set_xlim(-1, N + 18); ax.grid(alpha=0.2); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Gold {gold.SYMBOL} 4H - {side} (New York time)", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path

def write_report(df, trades, path="trades_2y.txt"):
    """لیست همه معاملات ۲ سال اخیر + آمار (نرخ برد کل، خرید/فروش، باخت‌های متوالی، افت)"""
    done = [t for t in trades if t["res"] in ("TP", "SL")]
    R = [2.0 if t["res"] == "TP" else -1.0 for t in done]
    def rate(sub):
        w = sum(t["res"] == "TP" for t in sub)
        return f"{len(sub)} معامله | TP {w} | SL {len(sub)-w} | نرخ برد {w/len(sub)*100:.1f}%" if sub else "0 معامله"
    eq, peak, dd, run, mx = 0.0, 0.0, 0.0, 0, 0
    for r in R:
        eq += r; peak = max(peak, eq); dd = min(dd, eq - peak)
        run = run + 1 if r < 0 else 0; mx = max(mx, run)
    half = len(df) // 2
    lines = ["بک‌تست استراتژی دانچیان ۲۰ + فیلتر روند | طلا ۴ ساعته | حد ضرر 2ATR | حد سود 4ATR (۱ به ۲)",
             f"بازه: {df.index[0].date()} تا {df.index[-1].date()}  (وقت نیویورک، ورود در قیمت بسته شدن کندل سیگنال، بدون هزینه اسپرد)", "",
             "نرخ برد کل : " + rate(done),
             "فقط خرید   : " + rate([t for t in done if t["side"] == "BUY"]),
             "فقط فروش   : " + rate([t for t in done if t["side"] == "SELL"]),
             "نیمه اول   : " + rate([t for t in done if t["i"] < half]),
             "نیمه دوم   : " + rate([t for t in done if t["i"] >= half]),
             f"میانگین هر معامله: {sum(R)/len(R):+.2f}R | مجموع: {sum(R):+.1f}R | بدترین افت: {dd:.1f}R | بیشترین باخت متوالی: {mx}" if R else "",
             f"سربه‌سر شدن در نسبت ۱ به ۲ نیاز به نرخ برد 33.3% دارد.", "",
             "شماره | تاریخ ورود (NY)   | نوع  | ورود    | SL      | TP      | نتیجه | R"]
    for k, t in enumerate(trades, 1):
        r = {"TP": "+2", "SL": "-1", "OPEN": "باز"}[t["res"]]
        lines.append(f"{k:>4}  | {df.index[t['i']].strftime('%Y-%m-%d %H:%M')} | {t['side']:<4} | {t['e']:7.1f} | {t['sl']:7.1f} | {t['tp']:7.1f} | {t['res']:<4}  | {r}")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))

def main():
    df = gold.load()
    c, h, l = df["Close"], df["High"], df["Low"]
    e50, e200 = (c.ewm(span=n, adjust=False).mean() for n in (50, 200))
    dh, dl = h.rolling(20).max().shift(), l.rolling(20).min().shift()
    sig = gold.signals_donchian(df); trades = find_trades(df, sig)
    sm = summary(trades)
    write_report(df, trades)

    # ---- نمودار ۱: آخرین ۲۲۰ کندل
    N = min(220, len(df)); k = len(df) - N; d = df.iloc[k:]; x = np.arange(N)
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=130)
    up = (d["Close"] >= d["Open"]).values; cc = np.where(up, "#1a9e5c", "#d6453d")
    ax.vlines(x, d["Low"], d["High"], color=cc, lw=1)
    ax.bar(x, (d["Close"] - d["Open"]).abs(), bottom=np.minimum(d["Open"], d["Close"]), width=0.65, color=cc)
    ax.plot(x, e50.iloc[k:], color=Y, lw=1.6); ax.plot(x, e200.iloc[k:], color=B, lw=1.8)
    ax.plot(x, dh.iloc[k:], color=GR, lw=1, ls="--"); ax.plot(x, dl.iloc[k:], color=GR, lw=1, ls="--")
    ax.fill_between(x, dl.iloc[k:], dh.iloc[k:], color="#999", alpha=0.08)
    draw_trades(ax, trades, k, N, 150)
    step = max(1, N // 8)
    ax.set_xticks(x[::step]); ax.set_xticklabels([t.strftime("%m-%d %H:%M") for t in d.index[::step]], fontsize=8)
    ax.set_xlim(-1, N + 9); ax.grid(alpha=0.2); legend(ax)
    ax.set_title(f"Gold {gold.SYMBOL} 4H (New York time)\n{sm}", fontsize=11)
    fig.tight_layout(); fig.savefig("gold_chart.png"); plt.close(fig)

    # ---- نمودار ۲: کل ۲ سال
    n = len(df); x = np.arange(n)
    fig, ax = plt.subplots(figsize=(26, 9), dpi=100)
    ax.plot(x, c.values, color="#444", lw=0.9)
    ax.plot(x, e50.values, color=Y, lw=1.4); ax.plot(x, e200.values, color=B, lw=1.6)
    ax.plot(x, dh.values, color=GR, lw=0.6, ls="--", alpha=0.7); ax.plot(x, dl.values, color=GR, lw=0.6, ls="--", alpha=0.7)
    draw_trades(ax, trades, 0, n, 110)
    step = max(1, n // 24)
    ax.set_xticks(x[::step]); ax.set_xticklabels([t.strftime("%Y-%m-%d") for t in df.index[::step]], fontsize=9, rotation=30)
    ax.set_xlim(0, n + 10); ax.grid(alpha=0.2); legend(ax)
    ax.set_title(f"Gold {gold.SYMBOL} 4H - last ~2 years - Donchian breakout strategy\n{sm}", fontsize=13)
    fig.tight_layout(); fig.savefig("gold_2y.png"); plt.close(fig)
    print(sm)

if __name__ == "__main__":
    main()
