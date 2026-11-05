
"""Gold (XAUUSD) Telegram signal bot. Signals only, never trades."""
import io
import json
import os
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

SYMBOLS = ["XAUUSD=X", "GC=F"]
TV_LINK = "https://www.tradingview.com/chart/?symbol=OANDA:XAUUSD"
CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
STATE_FILE = "state.json"
TEHRAN = dt.timezone(dt.timedelta(hours=3, minutes=30))

MIN_SL_USD = 2.5
MAX_SL_ATR = 3.5
PRE_ALERT_MIN = 30
QUIET_BEFORE_MIN = 15
QUIET_AFTER_MIN = 15
NEWS_MIN_MOVE = 4.0
NEWS_WINDOW_MIN = 25
BRIEFING_HOUR_TEHRAN = 8
SIGNAL_EXPIRE_HOURS = 24


def tg_text(text):
    if not TOKEN or not CHAT:
        print("[no telegram config]\n" + text + "\n")
        return
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT, "text": text,
                            "disable_web_page_preview": True}, timeout=30)
    print("telegram text:", r.status_code)


def tg_photo(buf, caption):
    if not TOKEN or not CHAT:
        print("[no telegram config] (photo)\n" + caption + "\n")
        return
    buf.seek(0)
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
                      data={"chat_id": CHAT, "caption": caption[:1000]},
                      files={"photo": ("chart.png", buf, "image/png")}, timeout=60)
    print("telegram photo:", r.status_code)
    if r.status_code != 200:
        tg_text(caption)


def load_state():
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
    except Exception:
        s = {}
    s.setdefault("sent", [])
    s.setdefault("open", [])
    s.setdefault("stats", {"win": 0, "loss": 0})
    return s


def save_state(state):
    state["sent"] = state["sent"][-400:]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def once(state, key):
    if key in state["sent"]:
        return False
    state["sent"].append(key)
    return True


def tehran_time(ts):
    return pd.Timestamp(ts).tz_convert(TEHRAN).strftime("%H:%M")


def get_df(interval, period):
    for sym in SYMBOLS:
        try:
            df = yf.download(sym, period=period, interval=interval,
                             progress=False, auto_adjust=False)
        except Exception as e:
            print("download error", sym, e)
            continue
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        if len(df) < 50:
            continue
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        return df, sym
    return None, None


def drop_unfinished(df, minutes):
    now = pd.Timestamp.now(tz="UTC")
    if len(df) and df.index[-1] + pd.Timedelta(minutes=minutes) > now:
        return df.iloc[:-1]
    return df


def apply_offset(df, off):
    if off == 0:
        return df
    df = df.copy()
    for c in ("Open", "High", "Low", "Close"):
        df[c] = df[c] + off
    return df


def tv_summary(interval):
    try:
        from tradingview_ta import TA_Handler
        h = TA_Handler(symbol="XAUUSD", exchange="OANDA", screener="forex", interval=interval)
        a = h.get_analysis()
        return a.summary.get("RECOMMENDATION"), a.indicators.get("close")
    except Exception as e:
        print("tradingview error:", interval, e)
        return None, None


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    h, l = df["High"], df["Low"]
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def get_news(state):
    now = pd.Timestamp.now(tz="UTC")
    cal = state.get("cal")
    if cal and (now - pd.Timestamp(cal["t"])).total_seconds() < 3 * 3600:
        return [(pd.Timestamp(a), b) for a, b in cal["events"]], True
    try:
        data = requests.get(CAL_URL, timeout=25, headers={"User-Agent": "Mozilla/5.0"}).json()
        if not isinstance(data, list):
            raise ValueError("unexpected calendar response")
        events = []
        for e in data:
            if e.get("country") == "USD" and e.get("impact") == "High":
                events.append((pd.to_datetime(e["date"], utc=True), e.get("title", "News")))
        state["cal"] = {"t": now.isoformat(),
                        "events": [[t.isoformat(), n] for t, n in events]}
        return events, True
    except Exception as e:
        print("calendar error:", e)
        if cal:
            return [(pd.Timestamp(a), b) for a, b in cal["events"]], True
        return [], False


def daily_briefing(state, events, cal_ok):
    now = pd.Timestamp.now(tz=TEHRAN)
    key = f"daily:{now.date()}"
    if now.hour < BRIEFING_HOUR_TEHRAN or key in state["sent"]:
        return
    state["sent"].append(key)
    today = sorted([(t, n) for t, n in events if t.tz_convert(TEHRAN).date() == now.date()])
    st = state["stats"]
    stats_line = f"\n\n📈 نتیجه سیگنال‌های قبلی: {st['win']} برد / {st['loss']} باخت" if (st["win"] + st["loss"]) else ""
    if not cal_ok:
        msg = "☀️ صبح بخیر\nتقویم خبری الان در دسترس نبود؛ خودت تقویم اقتصادی رو چک کن."
    elif today:
        lines = "\n".join(f"• {tehran_time(t)} — {n}" for t, n in today)
        msg = ("☀️ صبح بخیر\n📅 امروز خبر مهم دلار داریم (تایم خبری):\n" + lines +
               "\n\nنزدیک این ساعت‌ها اسپرد و نوسان طلا بالا می‌ره. ربات ۱۵ دقیقه قبل و بعد خبر "
               "سیگنال عادی نمی‌ده و بعد از خبر، در صورت حرکت واضح، سیگنال «تایم خبری» می‌ده.")
    else:
        msg = "☀️ صبح بخیر\n✅ امروز خبر مهم دلار نداریم؛ سیگنال‌ها فقط تکنیکالی هستن."
    tg_text(msg + stats_line)


def pre_alerts(state, events):
    now = pd.Timestamp.now(tz="UTC")
    quiet = False
    for t, title in events:
        mins = (t - now).total_seconds() / 60.0
        if -QUIET_AFTER_MIN <= mins <= QUIET_BEFORE_MIN:
            quiet = True
        if 0 < mins <= PRE_ALERT_MIN and once(state, f"pre:{title}:{t.isoformat()}"):
            tg_text(f"⚠️ تایم خبری نزدیکه\n{title}\nساعت {tehran_time(t)} (وقت ایران) — حدود {int(mins)} دقیقه دیگه\n"
                    "تا بعد از خبر معامله جدید باز نکن. اگه حرکت واضحی شد، سیگنال «تایم خبری» می‌فرستم.")
    return quiet


def build_levels(entry, side, swing, a, floor=MIN_SL_USD, cap_atr=MAX_SL_ATR):
    if side == "BUY":
        sl = min(swing - 0.2 * a, entry - 1.2 * a)
        risk = max(entry - sl, floor)
        sl = entry - risk
        tps = [entry + risk * k for k in (1, 2, 3)]
    else:
        sl = max(swing + 0.2 * a, entry + 1.2 * a)
        risk = max(sl - entry, floor)
        sl = entry + risk
        tps = [entry - risk * k for k in (1, 2, 3)]
    if risk > cap_atr * a:
        return None
    return sl, tps, risk


def chart_png(df, side, entry, sl, tps, title):
    d = df.iloc[-70:]
    e20 = ema(df["Close"], 20).iloc[-70:]
    e50 = ema(df["Close"], 50).iloc[-70:]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=110)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    for i, (o, h, l, c) in enumerate(zip(d["Open"], d["High"], d["Low"], d["Close"])):
        col = "#26a69a" if c >= o else "#ef5350"
        ax.plot([i, i], [l, h], color=col, lw=0.8)
        ax.add_patch(plt.Rectangle((i - 0.3, min(o, c)), 0.6, max(abs(c - o), 0.03), color=col))
    xs = range(len(d))
    ax.plot(xs, e20.values, color="#ffb74d", lw=1, label="EMA20")
    ax.plot(xs, e50.values, color="#64b5f6", lw=1, label="EMA50")
    ax.axhline(entry, color="#ffffff", lw=1, ls="--")
    ax.axhline(sl, color="#ef5350", lw=1.2)
    for i, tp in enumerate(tps, 1):
        ax.axhline(tp, color="#66bb6a", lw=1)
    x_txt = len(d) + 0.5
    ax.text(x_txt, entry, f"ENTRY {entry:.2f}", color="white", fontsize=7, va="center")
    ax.text(x_txt, sl, f"SL {sl:.2f}", color="#ef5350", fontsize=7, va="center")
    for i, tp in enumerate(tps, 1):
        ax.text(x_txt, tp, f"TP{i} {tp:.2f}", color="#66bb6a", fontsize=7, va="center")
    lo = min(d["Low"].min(), sl, *tps) - 1
    hi = max(d["High"].max(), sl, *tps) + 1
    ax.set_ylim(lo, hi)
    ax.set_xlim(-1, len(d) + 9)
    ax.tick_params(colors="#9aa0a6", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.set_title(title, color="white", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, facecolor="#0e1117", labelcolor="white", edgecolor="#30363d")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def register_signal(state, side, entry, sl, tps, name, news):
    state["open"].append({"side": side, "entry": entry, "sl": sl, "tps": tps,
                          "time": pd.Timestamp.now(tz="UTC").isoformat(),
                          "hit": 0, "name": name, "news": news})


def strategy_signals(df):
    c = df["Close"]
    out = []
    a, pdi, mdi = adx(df)
    adx_now = float(a.iloc[-1])
    hh = df["High"].iloc[-21:-1].max()
    ll = df["Low"].iloc[-21:-1].min()
    if adx_now >= 20:
        if c.iloc[-1] > hh:
            out.append(("Turtle", "BUY", "شکست سقف ۲۰ کندل"))
        elif c.iloc[-1] < ll:
            out.append(("Turtle", "SELL", "شکست کف ۲۰ کندل"))
    e20 = ema(c, 20).iloc[-1]
    if adx_now >= 30:
        if pdi.iloc[-1] > mdi.iloc[-1] and df["Low"].iloc[-1] <= e20 < c.iloc[-1]:
            out.append(("HolyGrail", "BUY", "پولبک به EMA20 در روند صعودی قوی"))
        if mdi.iloc[-1] > pdi.iloc[-1] and df["High"].iloc[-1] >= e20 > c.iloc[-1]:
            out.append(("HolyGrail", "SELL", "پولبک به EMA20 در روند نزولی قوی"))
    r2 = float(rsi(c, 2).iloc[-1])
    sma200 = c.rolling(200).mean().iloc[-1]
    if not np.isnan(sma200):
        if r2 < 10 and c.iloc[-1] > sma200:
            out.append(("Connors", "BUY", f"RSI(2)={r2:.0f} پولبک در روند صعودی"))
        if r2 > 90 and c.iloc[-1] < sma200:
            out.append(("Connors", "SELL", f"RSI(2)={r2:.0f} پولبک در روند نزولی"))
    return out, adx_now


def h1_trend(df1h):
    c = df1h["Close"]
    e50, e200, last = ema(c, 50).iloc[-1], ema(c, 200).iloc[-1], c.iloc[-1]
    if e50 > e200 and last > e50:
        return "UP"
    if e50 < e200 and last < e50:
        return "DOWN"
    return "FLAT"


def tv_agree(rec, side):
    if not rec:
        return None
    good = ("BUY", "STRONG_BUY") if side == "BUY" else ("SELL", "STRONG_SELL")
    bad = ("SELL", "STRONG_SELL") if side == "BUY" else ("BUY", "STRONG_BUY")
    return True if rec in good else (False if rec in bad else None)


def normal_signal(state, df15, df1h, rec15):
    now = pd.Timestamp.now(tz="UTC")
    if not (6 <= now.hour < 20) or (now.weekday() == 4 and now.hour >= 19):
        return
    if len(df15) < 210 or len(df1h) < 210 or state["open"]:
        return
    bar_key = df15.index[-1].isoformat()
    sigs, adx_now = strategy_signals(df15)
    if not sigs:
        return
    trend = h1_trend(df1h)
    a = float(atr(df15).iloc[-1])
    rsi14 = float(rsi(df15["Close"], 14).iloc[-1])
    rec1h = rec4h = None
    tv_loaded = False

    for side in ("BUY", "SELL"):
        mine = [s for s in sigs if s[1] == side]
        if not mine:
            continue
        if (side == "BUY" and trend != "UP") or (side == "SELL" and trend != "DOWN"):
            continue
        if (side == "BUY" and rsi14 > 75) or (side == "SELL" and rsi14 < 25):
            continue
        if not tv_loaded:
            rec1h, _ = tv_summary("1h")
            rec4h, _ = tv_summary("4h")
            tv_loaded = True
        ag1, ag4 = tv_agree(rec1h, side), tv_agree(rec4h, side)
        if ag1 is False:
            continue
        tv_ok = rec1h is not None
        score = min(len(mine), 2) + 1 + (1 if ag1 else 0) + (1 if ag4 else 0)
        if score < (4 if tv_ok else 3):
            continue
        entry = float(df15["Close"].iloc[-1])
        swing = float(df15["Low"].iloc[-9:].min() if side == "BUY" else df15["High"].iloc[-9:].max())
        lv = build_levels(entry, side, swing, a)
        if lv is None or not once(state, f"sig:{side}:{bar_key}"):
            continue
        sl, tps, risk = lv
        icon = "🟢 خرید" if side == "BUY" else "🔴 فروش"
        names = ", ".join(s[0] for s in mine)
        why = "\n".join(f"• {s[2]}" for s in mine)
        tvtxt = f"{rec1h or '—'} (۱ساعته) | {rec4h or '—'} (۴ساعته) | {rec15 or '—'} (۱۵دقیقه)"
        caption = (
            f"{icon} طلا (XAUUSD)\n"
            f"{'⭐' * min(score, 5)} امتیاز {min(score, 5)}/5\n\n"
            f"📍 ورود: {entry:.2f}\n"
            f"🛑 حد ضرر: {sl:.2f}  ({risk:.1f}$)\n"
            f"🎯 TP1: {tps[0]:.2f}\n🎯 TP2: {tps[1]:.2f}\n🎯 TP3: {tps[2]:.2f}\n\n"
            f"📊 تحلیل:\n{why}\n"
            f"• روند ۱ساعته: {'صعودی' if trend == 'UP' else 'نزولی'} (EMA50/200)\n"
            f"• TradingView: {tvtxt}\n"
            f"• ADX={adx_now:.0f} ، RSI={rsi14:.0f}\n\n"
            f"🧭 بعد از TP1 حد ضرر رو بیار روی نقطه ورود. ریسک هر معامله حداکثر ۱٪.\n"
            f"🔗 {TV_LINK}\n⚠️ سود تضمینی نیست."
        )
        buf = chart_png(df15, side, entry, sl, tps, f"XAUUSD 15m - {names} ({side})")
        tg_photo(buf, caption)
        register_signal(state, side, entry, sl, tps, names, False)
        return


def news_signal(state, events, df5_raw):
    if state["open"]:
        return
    now = pd.Timestamp.now(tz="UTC")
    closed = drop_unfinished(df5_raw, 5)
    if len(closed) < 30:
        return
    a5 = float(atr(closed).iloc[-1])
    for t, title in events:
        mins = (now - t).total_seconds() / 60.0
        if not (0 <= mins <= NEWS_WINDOW_MIN):
            continue
        key = f"post:{title}:{t.isoformat()}"
        if key in state["sent"]:
            continue
        before = closed[closed.index + pd.Timedelta(minutes=5) <= t]
        post = closed[closed.index >= t]
        if before.empty or len(post) < 2:
            continue
        ref = float(before["Close"].iloc[-1])
        last = float(closed["Close"].iloc[-1])
        move = last - ref
        if abs(move) < NEWS_MIN_MOVE:
            continue
        side = "BUY" if move > 0 else "SELL"
        last2 = closed.iloc[-2:]
        if side == "BUY" and not (last2["Close"] > last2["Open"]).all():
            continue
        if side == "SELL" and not (last2["Close"] < last2["Open"]).all():
            continue
        extreme = float(post["High"].max() if side == "BUY" else post["Low"].min())
        if abs(extreme - ref) == 0 or abs(extreme - last) / abs(extreme - ref) >= 0.5:
            continue
        entry = float(df5_raw["Close"].iloc[-1])
        recent = closed.iloc[-3:]
        swing = float(recent["Low"].min() if side == "BUY" else recent["High"].max())
        lv = build_levels(entry, side, swing, a5, floor=MIN_SL_USD, cap_atr=12 / max(a5, 0.01))
        if lv is None:
            continue
        sl, tps, risk = lv
        if risk > 12 or not once(state, key):
            continue
        icon = "🟢 خرید" if side == "BUY" else "🔴 فروش"
        caption = (
            f"📰 سیگنال تایم خبری — {icon} طلا\n"
            f"خبر: {title} ({tehran_time(t)} وقت ایران)\n"
            f"حرکت بعد از خبر: {move:+.1f}$ و ادامه‌دار (پولبک کمتر از ۵۰٪)\n\n"
            f"📍 ورود: {entry:.2f}\n"
            f"🛑 حد ضرر: {sl:.2f}  ({risk:.1f}$)\n"
            f"🎯 TP1: {tps[0]:.2f}\n🎯 TP2: {tps[1]:.2f}\n🎯 TP3: {tps[2]:.2f}\n\n"
            f"⚠️ ریسک خبری بالاست (اسپرد باز، لغزش قیمت). حجم رو نصف کن و بعد از TP1 استاپ رو بیار روی ورود.\n"
            f"🔗 {TV_LINK}"
        )
        buf = chart_png(closed, side, entry, sl, tps, f"XAUUSD 5m - NEWS ({side})")
        tg_photo(buf, caption)
        register_signal(state, side, entry, sl, tps, "News", True)
        return


def replay(s, bars):
    side, hit, closed = s["side"], 0, None
    for _, b in bars.iterrows():
        stop = s["sl"] if hit == 0 else (s["entry"] if hit == 1 else s["tps"][0])
        if (side == "BUY" and b["Low"] <= stop) or (side == "SELL" and b["High"] >= stop):
            closed = "SL" if hit == 0 else "BE"
            break
        while hit < 3 and ((side == "BUY" and b["High"] >= s["tps"][hit]) or
                           (side == "SELL" and b["Low"] <= s["tps"][hit])):
            hit += 1
        if hit == 3:
            closed = "TP3"
            break
    return hit, closed


def check_open(state, df5):
    keep = []
    now = pd.Timestamp.now(tz="UTC")
    for s in state["open"]:
        t0 = pd.Timestamp(s["time"])
        hit, closed = replay(s, df5[df5.index > t0])
        tag = "📰" if s.get("news") else "📌"
        for n in range(s["hit"] + 1, hit + 1):
            extra = "\nحد ضرر رو بیار روی نقطه ورود." if n == 1 else ""
            tg_text(f"✅ {tag} TP{n} خورد — {s['side']} طلا (ورود {s['entry']:.2f}){extra}")
        s["hit"] = hit
        if closed:
            if closed == "SL":
                state["stats"]["loss"] += 1
                tg_text(f"❌ {tag} حد ضرر خورد — {s['side']} طلا (ورود {s['entry']:.2f}، SL {s['sl']:.2f})")
            else:
                state["stats"]["win"] += 1
                msg = "همه TPها خورد 🏆" if closed == "TP3" else f"بعد از TP{hit} با استاپ جابه‌جا شده بسته شد"
                tg_text(f"🏁 {tag} {s['side']} طلا بسته شد: {msg}")
            continue
        if (now - t0).total_seconds() > SIGNAL_EXPIRE_HOURS * 3600:
            tg_text(f"⏱ سیگنال {s['side']} (ورود {s['entry']:.2f}) منقضی شد؛ اگه هنوز بازه دستی مدیریت کن.")
            if hit >= 1:
                state["stats"]["win"] += 1
            continue
        keep.append(s)
    state["open"] = keep


def main():
    state = load_state()
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        tg_text("✅ ربات روشن شد و به تلگرام وصله.\nسیگنال‌ها فقط وقتی میان که شرایط کامل باشه.")
    now = pd.Timestamp.now(tz="UTC")
    if now.weekday() >= 5:
        return
    events, cal_ok = get_news(state)
    daily_briefing(state, events, cal_ok)
    quiet = pre_alerts(state, events)

    raw15, src = get_df("15m", "30d")
    raw5, _ = get_df("5m", "5d")
    raw1h, _ = get_df("1h", "60d")
    if raw15 is None or raw5 is None or raw1h is None:
        print("no price data")
        save_state(state)
        return

    market_open = (now - raw15.index[-1]).total_seconds() / 60 <= 20

    rec15, tv_close = tv_summary("15m")
    off = 0.0
    if tv_close:
        o = float(tv_close) - float(raw15["Close"].iloc[-1])
        if abs(o) < 80 and (src == "GC=F" or abs(o) > 1.5):
            off = o
    df15, df5, df1h = (apply_offset(d, off) for d in (raw15, raw5, raw1h))

    if market_open:
        check_open(state, df5)
        news_signal(state, events, df5)
        if not quiet:
            normal_signal(state, drop_unfinished(df15, 15), drop_unfinished(df1h, 60), rec15)
    save_state(state)


if __name__ == "__main__":
    main()
