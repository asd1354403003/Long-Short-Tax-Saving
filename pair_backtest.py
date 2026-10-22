"""Trade a ranked, frozen pair pool with finite slots and repeat entries."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from market_data import YahooFinanceSource


@dataclass(frozen=True)
class PairBacktestConfig:
    max_pairs: int = 15
    long_amount: float = 1_000_000
    start_date: str = "2026-01-01"
    end_date: str | None = None
    allow_shared_tickers: bool = False
    borrow_fee_rate: float = 0.0
    transaction_cost_rate: float = 0.0

    def __post_init__(self):
        if isinstance(self.max_pairs, bool) or not isinstance(self.max_pairs, int) or self.max_pairs < 1:
            raise ValueError("max_pairs must be a positive integer")
        if not np.isfinite(self.long_amount) or self.long_amount <= 0:
            raise ValueError("long_amount must be positive and finite")
        if pd.Timestamp(self.start_date) < pd.Timestamp("2026-01-01"):
            raise ValueError("The out-of-sample backtest starts in 2026 or later.")
        if self.end_date and pd.Timestamp(self.end_date) < pd.Timestamp(self.start_date):
            raise ValueError("end_date must not precede start_date")
        if self.borrow_fee_rate != 0 or self.transaction_cost_rate != 0:
            raise NotImplementedError("This version supports zero fees and borrow costs only.")

    @property
    def leg_cap(self):
        return self.long_amount / self.max_pairs


class PairPoolBacktester:
    """Scan validation-Sharpe ranks; trade Close signals at the next Open.

    Closed pairs return to eligibility. Models are frozen through 2025. At entry a common
    strict_one_leg scale caps both legs, followed by whole-share rounding.
    Shares stay fixed until the pair closes. Unfilled slots remain in cash.
    """

    def __init__(self, pairs: pd.DataFrame, config: PairBacktestConfig | None = None, *, source=None):
        self.config = config or PairBacktestConfig()
        required = {"pair_id", "y_ticker", "x_ticker", "validation_sharpe", "validation_passed",
                    "validation_end", "refit_status", "refit_end", "refit_alpha", "refit_beta",
                    "refit_mu_e", "refit_sigma_eq", "entry_z", "exit_z", "stop_loss"}
        if not required.issubset(pairs.columns) or pairs.pair_id.duplicated().any():
            raise ValueError("Provide the trained pair_pool.csv with unique pair IDs and frozen model fields.")
        ranked = pairs.sort_values("validation_sharpe", ascending=False, na_position="last", kind="stable").copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked = ranked.loc[ranked.validation_passed.eq(True) & ranked.refit_status.eq("ok")].copy()
        if ranked.empty:
            raise ValueError("No pairs passed validation with a valid pre-2026 refit.")
        for field in ("validation_end", "refit_end"):
            if not pd.to_datetime(ranked[field]).lt(pd.Timestamp(self.config.start_date)).all():
                raise ValueError("Model fitting and validation must finish before the backtest starts.")
        numeric = ["validation_sharpe", "refit_alpha", "refit_beta", "refit_mu_e", "refit_sigma_eq",
                   "entry_z", "exit_z", "stop_loss"]
        if not np.isfinite(ranked[numeric].to_numpy(dtype=float)).all():
            raise ValueError("Eligible pairs need finite model estimates and thresholds.")
        if (not ranked.refit_beta.gt(0).all() or not ranked.refit_sigma_eq.gt(0).all()
                or not ranked.entry_z.gt(ranked.exit_z).all() or not ranked.exit_z.ge(0).all()
                or not ranked.stop_loss.between(0, 1, inclusive="right").all()
                or ranked.y_ticker.eq(ranked.x_ticker).any()):
            raise ValueError("Invalid hedge ratios, scales, thresholds or pair membership.")
        self.pairs = ranked.reset_index(drop=True)
        self.source = source if source is not None else YahooFinanceSource()

    @classmethod
    def from_csv(cls, path=Path("data/pair_pool.csv"), config=None, **kwargs):
        return cls(pd.read_csv(path), config, **kwargs)

    @staticmethod
    def size_pair(open_y, open_x, beta, leg_cap, direction):
        """One common continuous scale, then round both share magnitudes down."""
        qy = int(np.floor(min(leg_cap / open_y, leg_cap / (beta * open_x))))
        qx = int(np.floor(beta * qy))
        return direction * qy, -direction * qx

    @staticmethod
    def _clean(frame):
        if frame is None or not {"Open", "Close"}.issubset(frame.columns):
            return pd.DataFrame(columns=["Open", "Close"], index=pd.DatetimeIndex([]))
        frame = frame[["Open", "Close"]].copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
        frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
        frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return frame.where(frame > 0)

    @staticmethod
    def _event(state, date, signal_date, event, prices, reason, pnls=(0.0, 0.0)):
        positive = 0 if state["qy"] > 0 else 1
        negative = 1 - positive
        tickers, shares = (state["y"], state["x"]), (state["qy"], state["qx"])
        entry = (state["entry_y"], state["entry_x"])
        return {"execution_date": date, "signal_date": signal_date, "event": event,
                "trade_id": state["trade_id"], "pair_id": state["pair_id"],
                "rank": state["rank"], "entry_date": state["entry_date"],
                "long_ticker": tickers[positive], "short_ticker": tickers[negative],
                "long_shares": shares[positive], "short_shares": shares[negative],
                "long_price": prices[positive], "short_price": prices[negative],
                "long_amount": shares[positive] * prices[positive],
                "short_amount": shares[negative] * prices[negative],
                "long_entry_price": entry[positive], "short_entry_price": entry[negative],
                "long_realized_pnl": pnls[positive], "short_realized_pnl": pnls[negative],
                "realized_pnl": sum(pnls), "reason": reason,
                "hedge_beta": state["beta"] if positive == 0 else 1 / state["beta"]}

    def run(self, *, histories=None):
        config = self.config
        start = pd.Timestamp(config.start_date)
        # Include today's bar only after 17:00 New York time, allowing a data-publication buffer.
        end = (pd.Timestamp(config.end_date) if config.end_date else
               (pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(hours=17)).normalize().tz_localize(None))
        if end < start:
            raise ValueError("No completed backtest dates are available.")
        download_start = start - pd.Timedelta(days=10)
        tickers = list(dict.fromkeys(self.pairs[["y_ticker", "x_ticker"]].to_numpy().ravel()))
        if histories is None:
            histories = self.source.history(list(dict.fromkeys(tickers + ["SPY"])),
                download_start.strftime("%Y-%m-%d"), (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        clean = {ticker: self._clean(histories.get(ticker)).loc[download_start:end]
                 for ticker in set(tickers + ["SPY"])}
        spy = clean["SPY"].dropna()
        dates = spy.index[spy.index >= start]
        if dates.empty:
            raise ValueError("SPY needs valid Open/Close prices over the backtest dates.")
        all_dates = spy.index
        opening = pd.DataFrame({t: clean[t].Open.reindex(all_dates) for t in tickers})
        closing = pd.DataFrame({t: clean[t].Close.reindex(all_dates) for t in tickers})
        models = self.pairs.to_dict("records")
        by_id = {m["pair_id"]: m for m in models}
        z_scores = pd.DataFrame({m["pair_id"]: (
            closing[m["y_ticker"]] - m["refit_alpha"] - m["refit_beta"] * closing[m["x_ticker"]]
            - m["refit_mu_e"]) / m["refit_sigma_eq"] for m in models})

        def scan(date, active, exited_today):
            entries = []
            for model in models:
                pid = model["pair_id"]
                if pid in active:
                    continue
                if pid in exited_today and model.get("prevent_immediate_reentry", True):
                    continue
                value = z_scores.at[date, pid]
                if np.isfinite(value) and abs(value) > model["entry_z"]:
                    entries.append((pid, -1 if value > 0 else 1, date))
            return entries

        long_positions = pd.DataFrame(0.0, index=dates, columns=tickers)
        short_positions = long_positions.copy()
        active, traded_pairs, pending_exits = {}, set(), {}
        entry_count = 0
        previous = all_dates[all_dates < dates[0]]
        pending_entries = scan(previous[-1], active, set()) if len(previous) else []
        cash = {"long": config.long_amount, "short": config.long_amount}
        previous_nav = cash.copy()
        realized_gain = realized_loss = 0.0
        events, snapshots, summaries = [], [], []
        for date in dates:
            day_gain = day_loss = 0.0
            exited_today = set()
            for pid in list(pending_exits):
                state = active[pid]
                prices = opening.loc[date, [state["y"], state["x"]]].to_numpy(dtype=float)
                if not np.isfinite(prices).all():
                    continue
                reason, signal_date = pending_exits.pop(pid)
                quantities = (state["qy"], state["qx"])
                pnls = tuple(q * (p - cost) for q, p, cost in zip(quantities, prices,
                                                               (state["entry_y"], state["entry_x"])))
                for q, price, pnl in zip(quantities, prices, pnls):
                    cash["long" if q > 0 else "short"] += q * price
                    day_gain += max(pnl, 0)
                    day_loss += min(pnl, 0)
                events.append(self._event(state, date, signal_date, "close", prices, reason, pnls))
                del active[pid]
                exited_today.add(pid)
            for pid, direction, signal_date in pending_entries:
                if len(active) >= config.max_pairs:
                    break
                model = by_id[pid]
                y, x = model["y_ticker"], model["x_ticker"]
                if pid in active:
                    continue
                occupied = {t for state in active.values() for t in (state["y"], state["x"])}
                if not config.allow_shared_tickers and ({y, x} & occupied):
                    continue
                prices = opening.loc[date, [y, x]].to_numpy(dtype=float)
                if not np.isfinite(prices).all():
                    continue
                qy, qx = self.size_pair(*prices, model["refit_beta"], config.leg_cap, direction)
                if not qy or not qx:
                    continue
                entry_count += 1
                state = {"trade_id": entry_count, "pair_id": pid, "rank": model["rank"],
                         "y": y, "x": x, "qy": qy, "qx": qx,
                         "entry_y": prices[0], "entry_x": prices[1], "mark_y": prices[0], "mark_x": prices[1],
                         "entry_date": date, "beta": model["refit_beta"],
                         "entry_gross": abs(qy) * prices[0] + abs(qx) * prices[1]}
                for q, price in zip((qy, qx), prices):
                    cash["long" if q > 0 else "short"] -= q * price
                active[pid] = state
                traded_pairs.add(pid)
                events.append(self._event(state, date, signal_date, "open", prices, "z_entry"))
            realized_gain += day_gain
            realized_loss += day_loss
            day_long = day_short = unrealized = 0.0
            stale_legs = 0
            for pid, state in active.items():
                model = by_id[pid]
                quotes = closing.loc[date, [state["y"], state["x"]]].to_numpy(dtype=float)
                for side, quote in zip(("y", "x"), quotes):
                    if np.isfinite(quote):
                        state["mark_" + side] = quote
                    else:
                        stale_legs += 1
                marks = (state["mark_y"], state["mark_x"])
                pair_pnl = state["qy"] * (marks[0] - state["entry_y"]) + state["qx"] * (marks[1] - state["entry_x"])
                unrealized += pair_pnl
                snapshot = self._event(state, date, pd.NaT, "hold", marks, "")
                snapshot.update(Date=date, unrealized_pnl=pair_pnl, z_score=z_scores.at[date, pid],
                                holding_days=(date - state["entry_date"]).days)
                snapshots.append(snapshot)
                for ticker, q, mark in zip((state["y"], state["x"]), (state["qy"], state["qx"]), marks):
                    if q > 0:
                        long_positions.at[date, ticker] += q * mark
                        day_long += q * mark
                    else:
                        short_positions.at[date, ticker] += q * mark
                        day_short += q * mark
                value = z_scores.at[date, pid]
                if pid not in pending_exits and np.isfinite(quotes).all():
                    direction = 1 if state["qy"] > 0 else -1
                    if pair_pnl <= -model["stop_loss"] * state["entry_gross"]:
                        pending_exits[pid] = ("stop", date)
                    elif ((direction == 1 and value >= -model["exit_z"])
                          or (direction == -1 and value <= model["exit_z"])):
                        pending_exits[pid] = ("z_exit", date)
            nav_long, nav_short = cash["long"] + day_long, cash["short"] + day_short
            long_pnl, short_pnl = nav_long - previous_nav["long"], nav_short - previous_nav["short"]
            equity = nav_long + nav_short
            cumulative_pnl = equity - 2 * config.long_amount
            if not np.isclose(cumulative_pnl, realized_gain + realized_loss + unrealized, atol=1e-6):
                raise ArithmeticError("Cash/position accounting does not reconcile with realized and unrealized P&L.")
            summaries.append({"Date": date, "active_pairs": len(active), "entry_count": entry_count,
                "traded_pairs": len(traded_pairs),
                "available_pairs": len(models) - len(active) - sum(
                    bool(by_id[pid].get("prevent_immediate_reentry", True)) for pid in exited_today),
                "long_amount": day_long, "short_amount": day_short, "gross_amount": day_long - day_short,
                "net_amount": day_long + day_short, "cash": cash["long"] + cash["short"], "equity": equity,
                "long_pnl": long_pnl, "short_pnl": short_pnl, "daily_pnl": long_pnl + short_pnl,
                "daily_return": (long_pnl + short_pnl) / (2 * config.long_amount),
                "long_return": long_pnl / config.long_amount, "short_return": short_pnl / config.long_amount,
                "cumulative_pnl": cumulative_pnl, "cumulative_return": cumulative_pnl / (2 * config.long_amount),
                "realized_gain": day_gain, "realized_loss": day_loss,
                "cumulative_realized_gain": realized_gain, "cumulative_realized_loss": realized_loss,
                "cumulative_realized_pnl": realized_gain + realized_loss, "unrealized_pnl": unrealized,
                "stale_price_legs": stale_legs})
            previous_nav = {"long": nav_long, "short": nav_short}
            # Restart at the highest rank each Close; past trades do not blacklist a pair.
            pending_entries = scan(date, active, exited_today)
        daily = pd.DataFrame(summaries).set_index("Date")
        daily["spy_close"] = spy.Close.reindex(dates)
        daily["spy_return"] = daily.spy_close.pct_change(fill_method=None)
        first_open = spy.at[dates[0], "Open"]
        daily.loc[dates[0], "spy_return"] = daily.spy_close.iloc[0] / first_open - 1
        daily["spy_cumulative_return"] = daily.spy_close / first_open - 1
        event_columns = list(self._event({"trade_id": 0, "qy": 1, "qx": -1, "y": "", "x": "", "entry_y": 0,
            "entry_x": 0, "pair_id": "", "rank": 0, "entry_date": pd.NaT, "beta": 1},
            pd.NaT, pd.NaT, "", (0, 0), ""))
        event_frame = pd.DataFrame(events, columns=event_columns)
        position_frame = pd.DataFrame(snapshots, columns=event_columns + ["Date", "unrealized_pnl", "z_score", "holding_days"])
        open_positions = position_frame.loc[position_frame.Date.eq(dates[-1])].sort_values("rank")
        closed_trades = event_frame.loc[event_frame.event.eq("close")].copy()
        realized = closed_trades.groupby("pair_id").agg(
            closed_trades=("trade_id", "size"), realized_pnl=("realized_pnl", "sum"))
        unrealized = open_positions.set_index("pair_id")[["unrealized_pnl", "holding_days"]]
        performance = realized.join(unrealized, how="outer").fillna(0)
        performance["total_pnl"] = performance.realized_pnl + performance.unrealized_pnl
        performance = performance.join(self.pairs.set_index("pair_id")[["rank", "validation_sharpe"]])
        performance = performance.sort_values("total_pnl")
        if not np.isclose(performance.total_pnl.sum(), daily.cumulative_pnl.iloc[-1], atol=1e-6):
            raise ArithmeticError("Pair-level contributions do not reconcile with account P&L.")
        for frame in (long_positions, short_positions):
            frame.index.name, frame.columns.name = "Date", "ticker"
        return {"daily_summary": daily, "daily_positions_long": long_positions,
                "daily_positions_short": short_positions, "pair_positions": position_frame,
                "events": event_frame, "closed_trades": closed_trades,
                "pair_performance": performance,
                "open_positions": open_positions, "ranked_pairs": self.pairs,
                "z_scores": z_scores.reindex(dates), "pending_exits": pending_exits,
                "missing_tickers": [t for t in tickers if clean[t].dropna(how="all").empty],
                "source_events": list(self.source.events)}
