"""Construct, value and rotate whole-share long/short portfolios."""

import numpy as np
import pandas as pd


def _close_beta(close_history, y, x, start, cutoff):
    """Fit same-date Close OLS with an intercept, using no data after cutoff."""
    sample = close_history.loc[pd.Timestamp(start):pd.Timestamp(cutoff), [y, x]]
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 20 or (sample <= 0).any().any() or sample[x].nunique() < 2:
        raise ValueError(f"Insufficient valid Close history for {y}/{x} through {cutoff}.")
    design = np.column_stack([np.ones(len(sample)), sample[x].to_numpy()])
    beta = float(np.linalg.lstsq(design, sample[y].to_numpy(), rcond=None)[0][1])
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError(f"The Close hedge beta for {y}/{x} must be positive and finite.")
    return beta, {"beta_start": sample.index[0], "beta_date": sample.index[-1], "beta_nobs": len(sample)}


def _exposure_scale(long_value, short_value, long_amount, mode):
    """Return one common scale for both sides; short_value is an absolute amount."""
    if mode == "strict":
        exposure, limit = long_value + short_value, 2 * long_amount
    elif mode == "strict_one_leg":
        exposure, limit = max(long_value, short_value), long_amount
    else:
        return 1.0
    return min(1.0, limit / exposure) if exposure > 0 else 1.0


def construct_portfolios(entry_open, tickers_a, tickers_b, beta_ab,
                         long_amount=1_000_000, mode="beta", beta_column="quote_beta",
                         close_history=None, entry_date=None):
    """Construct equal-budget long A and beta-sized short B positions.

    long_amount is the target for A alone. In "beta" mode, short shares are
    -floor(beta * long shares). Short shares and amounts are negative.
    In "strict" mode, if gross exposure exceeds
    2 * long_amount, scale both legs by the same factor and round down again.
    In "strict_one_leg" mode, scale both legs by the same factor until
    long exposure and absolute short exposure are each <= long_amount.
    Limits apply at entry Open. All modes use whole shares only.
    Pass the pair_betas DataFrame to select AB coefficients with beta_column:
    "beta" is the frozen formation Close OLS slope. "quote_beta" refits Close
    OLS from formation_start through the last Close before entry_date.
    Supply full close_history (including formation data) for dynamic sizing.
    Numeric arrays are supported only as already-selected fixed "beta" inputs.
    This function only sizes positions; it does not calculate fees or returns.
    """
    if mode not in ("beta", "strict", "strict_one_leg"):
        raise ValueError('mode must be "beta", "strict" or "strict_one_leg".')
    if beta_column not in ("beta", "quote_beta"):
        raise ValueError('beta_column must be "beta" or "quote_beta".')
    if not len(tickers_a) or len(tickers_a) != len(tickers_b):
        raise ValueError("Provide the same nonzero number of A and B tickers.")
    beta_metadata = None
    if isinstance(beta_ab, pd.DataFrame):
        if "price_basis" in beta_ab and not beta_ab["price_basis"].eq("Close").all():
            raise ValueError("Rebuild pair_betas.csv with Close-only formation data.")
        ab = beta_ab.loc[beta_ab["pair"].eq("AB")].set_index("y_ticker")
        if not ab.index.is_unique:
            raise ValueError("Provide exactly one AB coefficient per A ticker.")
        ab = ab.reindex(list(tickers_a))
        if ab["x_ticker"].tolist() != list(tickers_b):
            raise ValueError("A and B tickers must match the AB coefficient rows.")
        beta_metadata = []
        if beta_column == "quote_beta":
            entry_date = entry_date if entry_date is not None else entry_open.name
            if close_history is None or entry_date is None:
                raise ValueError("quote_beta requires full Close history and an entry date.")
            close_history = close_history.sort_index()
            coefficients = []
            for y, row in ab.iterrows():
                coefficient, info = _close_beta(close_history, y, row["x_ticker"],
                                                row["formation_start"], pd.Timestamp(entry_date) - pd.Timedelta(days=1))
                coefficients.append(coefficient)
                beta_metadata.append(info)
            beta_ab = coefficients
        else:
            beta_metadata = [{"beta_start": pd.Timestamp(row.formation_start),
                              "beta_date": pd.Timestamp(row.formation_end), "beta_nobs": int(row.nobs)}
                             for row in ab.itertuples()]
            beta_ab = ab["beta"].to_numpy()
    elif beta_column == "quote_beta":
        raise ValueError("quote_beta requires the pair_betas DataFrame and full Close history.")
    beta_ab = np.asarray(beta_ab, dtype=float)
    if beta_ab.shape != (len(tickers_a),) or not (np.isfinite(beta_ab) & (beta_ab > 0)).all():
        raise ValueError("Provide one positive, finite sizing coefficient per pair.")
    if not np.isfinite(long_amount) or long_amount <= 0:
        raise ValueError("long_amount must be positive and finite.")

    portfolio_a = pd.DataFrame({"ticker": list(tickers_a)})
    portfolio_b = pd.DataFrame({"ticker": list(tickers_b), "beta": beta_ab, "beta_column": beta_column})
    if beta_metadata is not None:
        for column in ("beta_start", "beta_date", "beta_nobs"):
            portfolio_b[column] = [row[column] for row in beta_metadata]
    for portfolio in (portfolio_a, portfolio_b):
        portfolio["Open"] = portfolio["ticker"].map(entry_open)
        if not (np.isfinite(portfolio["Open"]) & (portfolio["Open"] > 0)).all():
            raise ValueError("Every ticker needs a positive, finite entry Open.")

    portfolio_a["shares"] = np.floor(
        (long_amount / len(portfolio_a)) / portfolio_a["Open"]
    ).astype(int)
    if not portfolio_a["shares"].any():
        raise ValueError("The long budget is too small to construct a hedged position.")
    portfolio_b["shares"] = -np.floor(
        portfolio_a["shares"].to_numpy() * portfolio_b["beta"]
    ).astype(int)

    long_value, short_value = (
        (portfolio["shares"].abs() * portfolio["Open"]).sum()
        for portfolio in (portfolio_a, portfolio_b)
    )
    scale = _exposure_scale(long_value, short_value, long_amount, mode)
    if scale < 1:
        for portfolio in (portfolio_a, portfolio_b):
            portfolio["shares"] = np.trunc(portfolio["shares"] * scale).astype(int)

    for portfolio in (portfolio_a, portfolio_b):
        portfolio["amount"] = portfolio["shares"] * portfolio["Open"]
    return portfolio_a, portfolio_b


def _daily_shares(dates, portfolio, side, history):
    """Expand initial shares or post-trade event shares to trading dates."""
    initial = portfolio.set_index("ticker")["shares"]
    direction = 1 if side == "long" else -1
    if not (np.isfinite(initial) & (initial * direction >= 0) & (initial % 1 == 0)).all():
        raise ValueError("Long shares must be nonnegative; short shares must be nonpositive whole shares.")
    if history is None:
        return pd.DataFrame(
            np.tile(initial.to_numpy(), (len(dates), 1)), index=dates, columns=initial.index,
        )
    rows = history.loc[history["side"].eq(side)].copy()
    rows["entry_date"] = pd.to_datetime(rows["entry_date"])
    if rows.empty or rows["entry_date"].min() != dates[0] or not rows["entry_date"].isin(dates).all():
        raise ValueError("History needs initial and rebalance events on the supplied trading dates.")
    if not (np.isfinite(rows["shares_after"]) & (rows["shares_after"] * direction >= 0) &
            (rows["shares_after"] % 1 == 0)).all():
        raise ValueError("History needs signed whole shares_after: long positive, short negative.")
    rows["held_ticker"] = rows["replace_with"].where(
        rows["replace_with"].fillna("").ne(""), rows["ticker"],
    )
    tickers = list(dict.fromkeys(initial.index.tolist() + rows["held_ticker"].tolist()))
    events = rows.groupby(["entry_date", "held_ticker"])["shares_after"].sum().unstack(fill_value=0)
    events = events.reindex(columns=tickers, fill_value=0).sort_index()
    if not np.array_equal(events.iloc[0].to_numpy(), initial.reindex(tickers, fill_value=0).to_numpy()):
        raise ValueError("History's initial shares must match the supplied portfolios.")
    # Zeros at each event remove retired tickers before forward-filling daily shares.
    shares = events.reindex(dates, method="ffill").astype(int)
    shares.columns.name = "ticker"
    return shares


def calculate_daily_positions(close_prices, portfolio_a, portfolio_b, entry_open=None,
                              *, rebalance_history=None, open_prices=None):
    """Return daily long/short market values and a price-only performance table.

    Shares stay fixed unless rebalance_history and open_prices are supplied.
    Rebalance events take effect at that date's Open. Deduct signed net buying
    on each side when calculating P&L, so position changes do not
    create artificial returns. This includes old shares' overnight P&L and
    the actual post-trade shares' Open-to-Close P&L, including strict trims.
    Long shares and values are positive; short shares and values are negative.
    A rise in a shorted asset's price produces a loss.
    Side returns divide signed daily P&L by the absolute previous Close amount
    (initial Open amount on day one). Combined returns use fixed initial
    gross exposure, without compounding. Fees are zero; dividends, financing
    and taxes are excluded. Long dividend receipts and short dividend
    compensation payments are both omitted, as are SPY distributions.
    Ex-dividend price moves remain in Close prices; returns are price-only.
    Realized loss/gain columns separate each closed position's signed P&L
    before grouping by execution date, including strict trims. Their cumulative
    net plus remaining unrealized P&L equals cumulative_pnl. With fixed shares,
    realized amounts stay zero and all price P&L remains unrealized.
    Pass entry_open to also include SPY prices, daily and cumulative returns.
    """
    a = portfolio_a.set_index("ticker")
    b = portfolio_b.set_index("ticker")
    closes = close_prices.sort_index()
    if closes.empty or not closes.index.is_unique:
        raise ValueError("Provide nonempty Close prices with unique trading dates.")
    shares_a = _daily_shares(closes.index, portfolio_a, "long", rebalance_history)
    shares_b = _daily_shares(closes.index, portfolio_b, "short", rebalance_history)
    tickers = list(dict.fromkeys(shares_a.columns.tolist() + shares_b.columns.tolist()))
    closes = closes.loc[:, tickers]
    if not np.isfinite(closes.to_numpy()).all() or not (closes > 0).all().all():
        raise ValueError("Every held ticker needs a positive, finite Close on each date.")

    daily_a = shares_a * closes.loc[:, shares_a.columns]
    daily_b = shares_b * closes.loc[:, shares_b.columns]
    initial_a = float((a["Open"] * a["shares"]).sum())
    initial_b = float((b["Open"] * b["shares"]).sum())
    if initial_a <= 0 or initial_b >= 0:
        raise ValueError("Entry long amount must be positive and short amount must be negative.")
    initial_gross = initial_a - initial_b

    daily = pd.DataFrame({
        "long_amount": daily_a.sum(axis=1),
        "short_amount": daily_b.sum(axis=1),
    })
    previous_a = daily["long_amount"].shift(1, fill_value=initial_a)
    previous_b = daily["short_amount"].shift(1, fill_value=initial_b)
    buys_a = pd.Series(0.0, index=daily.index)
    buys_b = pd.Series(0.0, index=daily.index)
    if rebalance_history is not None:
        if open_prices is None:
            raise ValueError("Provide Open prices to value rebalance trades.")
        opens = open_prices.sort_index().loc[:, tickers]
        if not opens.index.equals(closes.index):
            raise ValueError("Open and Close need matching trading dates for rebalances.")
        if not (np.isfinite(opens.to_numpy()) & (opens.to_numpy() > 0)).all():
            raise ValueError("Every held ticker needs positive, finite Open prices.")
        buys_a = (shares_a.diff().fillna(0) * opens.loc[:, shares_a.columns]).sum(axis=1)
        buys_b = (shares_b.diff().fillna(0) * opens.loc[:, shares_b.columns]).sum(axis=1)
    pnl_a = daily["long_amount"] - previous_a - buys_a
    pnl_b = daily["short_amount"] - previous_b - buys_b
    daily["gross_amount"] = daily["long_amount"] - daily["short_amount"]
    daily["long_pnl"] = pnl_a
    daily["short_pnl"] = pnl_b
    daily["long_return"] = pnl_a / previous_a
    daily["short_return"] = pnl_b / previous_b.abs()
    daily["daily_pnl"] = pnl_a + pnl_b
    daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()
    daily["daily_return"] = daily["daily_pnl"] / initial_gross
    daily["cumulative_return"] = daily["cumulative_pnl"] / initial_gross
    daily["realized_loss"] = 0.0
    daily["realized_gain"] = 0.0
    if rebalance_history is not None:
        realized = rebalance_history.loc[:, ["entry_date", "realized_pnl"]].copy()
        realized["entry_date"] = pd.to_datetime(realized["entry_date"])
        if not np.isfinite(realized["realized_pnl"]).all():
            raise ValueError("Every history row needs a finite realized_pnl.")
        # Split individual closing losses and gains before aggregating same-day trades.
        realized["realized_loss"] = realized["realized_pnl"].clip(upper=0)
        realized["realized_gain"] = realized["realized_pnl"].clip(lower=0)
        by_date = realized.groupby("entry_date")[["realized_loss", "realized_gain"]].sum()
        daily[["realized_loss", "realized_gain"]] = by_date.reindex(daily.index, fill_value=0)
    daily["realized_pnl"] = daily["realized_loss"] + daily["realized_gain"]
    daily["cumulative_realized_loss"] = daily["realized_loss"].cumsum()
    daily["cumulative_realized_gain"] = daily["realized_gain"].cumsum()
    daily["cumulative_realized_pnl"] = daily["realized_pnl"].cumsum()
    daily["unrealized_pnl"] = daily["cumulative_pnl"] - daily["cumulative_realized_pnl"]
    if entry_open is not None:
        spy_open = entry_open["SPY"]
        if not np.isfinite(spy_open) or spy_open <= 0:
            raise ValueError("SPY entry Open must be positive and finite.")
        daily["spy_close"] = close_prices.loc[daily.index, "SPY"]
        if not (np.isfinite(daily["spy_close"]) & (daily["spy_close"] > 0)).all():
            raise ValueError("SPY needs a positive, finite Close on each date.")
        daily["spy_return"] = daily["spy_close"] / daily["spy_close"].shift(1, fill_value=spy_open) - 1
        daily["spy_cumulative_return"] = daily["spy_close"] / spy_open - 1
    return daily_a, daily_b, daily


class PortfolioRebalancer:
    """Rotate either losing pair legs or the entire losing portfolio side.

    selection_mode="pair" selects at most one losing leg per triplet.
    selection_mode="portfolio" selects every asset on the side with the
    more negative aggregate unrealized P&L, including its individual winners.
    No replacement occurs if neither eligible leg/side has a loss. Ties go
    to the long side. Every replacement uses the unused triplet member.

    beta_column="beta" uses the frozen formation Close OLS slope.
    "quote_beta" refits same-date Close OLS from formation_start through
    check_date, before the next Open execution. Full close_history is required.
    "strict" caps combined gross at
    2 * long_amount after replacements. "strict_one_leg" caps the total
    long side and absolute short side separately at long_amount on each
    scheduled execution Open, even if no asset is replaced. Both constrained
    modes scale all positions together and may realize gains in retained
    positions. Price moves between events can exceed the limits. Costs are zero.
    Signals use Close. Cancel a selected leg's replacement if its loss has
    disappeared by the next Open; portfolio mode checks the selected side's
    aggregate Open P&L and cancels the whole basket together.
    If no regular check can execute, use the final two trading dates.
    """

    HISTORY_COLUMNS = [
        "entry_date", "check_date", "anchor_A", "basket", "basket_after", "side", "ticker", "paired_ticker", "shares",
        "cost_price", "price", "amount", "unrealized_pnl", "action",
        "replace_with", "shares_after", "realized_pnl",
    ]

    def __init__(self, open_prices, close_prices, portfolio_a, portfolio_b,
                 pair_betas, rebalance_frequency=3, long_amount=1_000_000,
                 mode="beta", selection_mode="pair", beta_column="quote_beta",
                 close_history=None):
        if mode not in ("beta", "strict", "strict_one_leg"):
            raise ValueError('mode must be "beta", "strict" or "strict_one_leg".')
        if selection_mode not in ("pair", "portfolio"):
            raise ValueError('selection_mode must be "pair" or "portfolio".')
        if beta_column not in ("beta", "quote_beta"):
            raise ValueError('beta_column must be "beta" or "quote_beta".')
        if "beta_column" in portfolio_b and not portfolio_b["beta_column"].eq(beta_column).all():
            raise ValueError("Use the same beta_column for initial construction and rebalancing.")
        if (isinstance(rebalance_frequency, bool) or
                not isinstance(rebalance_frequency, (int, np.integer)) or rebalance_frequency < 1):
            raise ValueError("rebalance_frequency must be a positive number of months.")
        if not np.isfinite(long_amount) or long_amount <= 0:
            raise ValueError("long_amount must be positive and finite.")
        self.opens = open_prices.sort_index().copy()
        self.closes = close_prices.sort_index().copy()
        if self.opens.empty or not self.opens.index.is_unique or not self.opens.index.equals(self.closes.index):
            raise ValueError("Open and Close need the same unique trading dates.")
        self.entry_date = self.opens.index[0]
        self.rebalance_frequency = rebalance_frequency
        self.long_amount = long_amount
        self.mode = mode
        self.selection_mode = selection_mode
        self.beta_column = beta_column
        self.close_history = close_history.sort_index().copy() if close_history is not None else None
        if beta_column == "quote_beta" and (self.close_history is None or not self.close_history.index.is_unique):
            raise ValueError("quote_beta requires full Close history with unique trading dates.")
        if "price_basis" in pair_betas and not pair_betas["price_basis"].eq("Close").all():
            raise ValueError("Rebuild pair_betas.csv with Close-only formation data.")
        self.beta_rows = pair_betas.copy()
        self.portfolio_a = portfolio_a.copy()
        self.portfolio_b = portfolio_b.copy()
        self.members, self.betas, self.beta_info = {}, {}, {}
        self.initial_holdings = self._initial_holdings(portfolio_a, portfolio_b, pair_betas)

    def _initial_holdings(self, portfolio_a, portfolio_b, pair_betas):
        ab = pair_betas.loc[pair_betas["pair"].eq("AB")].set_index("anchor_A")
        ac = pair_betas.loc[pair_betas["pair"].eq("AC")].set_index("anchor_A")
        bc = pair_betas.loc[pair_betas["pair"].eq("BC")].set_index("anchor_A")
        if (not all(frame.index.is_unique for frame in (ab, ac, bc)) or
                set(ab.index) != set(ac.index) or set(ac.index) != set(bc.index)):
            raise ValueError("Provide exactly one AB, AC and BC coefficient per triplet.")
        if len(portfolio_a) != len(portfolio_b) or not len(portfolio_a):
            raise ValueError("Provide matching, nonempty initial A and B portfolios.")
        if "beta" not in pair_betas:
            raise ValueError("The pair_betas table is missing beta.")
        rows = []
        for a, b in zip(portfolio_a.to_dict("records"), portfolio_b.to_dict("records")):
            anchor = a["ticker"]
            ab_row, ac_row, bc_row = ab.loc[anchor], ac.loc[anchor], bc.loc[anchor]
            c = ac_row["x_ticker"]
            if (ab_row["y_ticker"] != anchor or ab_row["x_ticker"] != b["ticker"] or
                    ac_row["y_ticker"] != anchor or bc_row["y_ticker"] != b["ticker"] or
                    bc_row["x_ticker"] != c):
                raise ValueError("Initial portfolios must follow their AB/AC/BC triplets.")
            self.members[anchor] = (anchor, b["ticker"], c)
            if len(set(self.members[anchor])) != 3:
                raise ValueError("Each triplet needs three distinct tickers.")
            self.betas[anchor] = {}
            self.beta_info[anchor] = {}
            for row in (ab_row, ac_row, bc_row):
                y, x, beta = row["y_ticker"], row["x_ticker"], row["beta"]
                if not np.isfinite(beta) or beta <= 0:
                    raise ValueError("All selected betas must be positive and finite.")
                self.betas[anchor][y, x] = float(beta)
                self.betas[anchor][x, y] = 1 / float(beta)
                info = {"beta_start": pd.Timestamp(row["formation_start"]),
                        "beta_date": pd.Timestamp(row["formation_end"]), "beta_nobs": int(row["nobs"])}
                self.beta_info[anchor][y, x] = self.beta_info[anchor][x, y] = info
            for side, position in (("long", a), ("short", b)):
                rows.append({"anchor_A": anchor, "book": "A" if side == "long" else "B",
                             "basket": self._basket(anchor, position["ticker"]),
                             "side": side, "ticker": position["ticker"],
                             "shares": position["shares"], "cost_date": self.entry_date,
                             "cost_price": position["Open"]})
        holdings = pd.DataFrame(rows)
        direction = holdings["side"].map({"long": 1, "short": -1})
        if not (np.isfinite(holdings["shares"]) & (holdings["shares"] * direction > 0) &
                (holdings["shares"] % 1 == 0)).all():
            raise ValueError("Every initial leg needs signed whole shares: long positive, short negative.")
        if not (np.isfinite(holdings["cost_price"]) & (holdings["cost_price"] > 0)).all():
            raise ValueError("Every initial leg needs a positive, finite cost price.")
        holdings["shares"] = holdings["shares"].astype(int)
        tickers = list(dict.fromkeys(t for triplet in self.members.values() for t in triplet))
        for prices in (self.opens.loc[:, tickers], self.closes.loc[:, tickers]):
            if not (np.isfinite(prices.to_numpy()) & (prices.to_numpy() > 0)).all():
                raise ValueError("Every triplet asset needs positive, finite Open/Close prices.")
        return holdings

    def _refresh_betas(self, cutoff):
        """Refit dynamic coefficients using only Close observations through cutoff."""
        if self.beta_column == "beta":
            return
        for row in self.beta_rows.itertuples():
            beta, info = _close_beta(self.close_history, row.y_ticker, row.x_ticker,
                                     row.formation_start, cutoff)
            self.betas[row.anchor_A][row.y_ticker, row.x_ticker] = beta
            self.betas[row.anchor_A][row.x_ticker, row.y_ticker] = 1 / beta
            self.beta_info[row.anchor_A][row.y_ticker, row.x_ticker] = info
            self.beta_info[row.anchor_A][row.x_ticker, row.y_ticker] = info

    def _basket(self, anchor, ticker):
        """Return the original A/B/C membership of the actual held ticker."""
        return "ABC"[self.members[anchor].index(ticker)]

    def _events(self):
        first_end = (self.entry_date.normalize().replace(day=1) +
                     pd.DateOffset(months=self.rebalance_frequency) - pd.Timedelta(days=1))
        ends = pd.date_range(first_end, self.closes.index[-1],
                             freq=pd.offsets.MonthEnd(self.rebalance_frequency))
        events = []
        dates = self.opens.index
        for period_end in ends:
            check_index = dates.searchsorted(period_end, side="right") - 1
            if check_index + 1 < len(dates):
                events.append({"entry_date": dates[check_index + 1], "check_date": dates[check_index],
                               "period_end": period_end, "fallback": False})
        # For a short sample, inspect the penultimate Close and fill at the final Open.
        if not events and len(dates) >= 2:
            events.append({"entry_date": dates[-1], "check_date": dates[-2],
                           "period_end": first_end, "fallback": True})
        return events

    def _snapshot(self, holdings, entry_date, check_date, marks):
        snapshot = holdings.drop(columns=["cost_date", "book"]).copy()
        snapshot.insert(0, "entry_date", entry_date)
        snapshot["check_date"] = check_date
        partners = holdings.groupby("anchor_A")["ticker"].agg(list)
        snapshot["paired_ticker"] = [
            next(t for t in partners[anchor] if t != ticker)
            for anchor, ticker in zip(snapshot["anchor_A"], snapshot["ticker"])
        ]
        snapshot["price"] = snapshot["ticker"].map(marks)
        snapshot["amount"] = snapshot["shares"] * snapshot["price"]
        snapshot["unrealized_pnl"] = snapshot["shares"] * (snapshot["price"] - snapshot["cost_price"])
        snapshot["action"] = "hold"
        snapshot["replace_with"] = ""
        snapshot["basket_after"] = snapshot["basket"]
        snapshot["shares_after"] = snapshot["shares"]
        snapshot["realized_pnl"] = 0.0
        return snapshot[self.HISTORY_COLUMNS]

    def _current_pairs(self, holdings, entry_date, marks):
        """Verify each current pair and expose its actual tickers and hedge ratio.

        Internal A/B books retain the long/short slots. The basket label tracks
        the actual ticker's A/B/C membership and changes on replacement.
        """
        rows, long_groups, short_groups = [], [], []
        for anchor, group in holdings.groupby("anchor_A", sort=False):
            if len(group) != 2 or set(group["book"]) != {"A", "B"}:
                raise ValueError("Each pair must retain one position in each A/B book.")
            books = group.set_index("book")
            long, short = books.loc["A"], books.loc["B"]
            if (long["side"] != "long" or short["side"] != "short" or
                    long["shares"] <= 0 or short["shares"] >= 0 or
                    long["ticker"] == short["ticker"]):
                raise ValueError("Each current pair must contain one long and one distinct short asset.")
            members = self.members[anchor]
            if (long["basket"] != self._basket(anchor, long["ticker"]) or
                    short["basket"] != self._basket(anchor, short["ticker"])):
                raise ValueError("Basket labels must match the actual A/B/C tickers.")
            long_groups.append(members.index(long["ticker"]))
            short_groups.append(members.index(short["ticker"]))
            rows.append({
                "entry_date": entry_date, "anchor_A": anchor,
                "long_basket": long["basket"],
                "long_ticker": long["ticker"], "long_shares": int(long["shares"]),
                "short_basket": short["basket"],
                "short_ticker": short["ticker"], "short_shares": int(short["shares"]),
                "hedge_beta": self.betas[anchor][long["ticker"], short["ticker"]],
                "beta_column": self.beta_column,
                **self.beta_info[anchor][long["ticker"], short["ticker"]],
                "held_ratio": -short["shares"] / long["shares"],
                "long_amount": float(long["shares"] * marks[long["ticker"]]),
                "short_amount": float(short["shares"] * marks[short["ticker"]]),
            })
        if self.selection_mode == "portfolio" and (len(set(long_groups)) != 1 or len(set(short_groups)) != 1):
            raise ValueError("Whole-portfolio rotation must retain two complete, distinct baskets.")
        return pd.DataFrame(rows)

    def _replacements(self, snapshot, open_pnl):
        """Confirm Close-selected losses at Open without selecting another leg."""
        selected_side = None
        if self.selection_mode == "portfolio":
            totals = snapshot.groupby("side")["unrealized_pnl"].sum().reindex(["long", "short"])
            if totals.min() >= 0:
                return {}
            selected_side = totals.idxmin()
            if open_pnl.loc[snapshot["side"].eq(selected_side)].sum() >= 0:
                return {}
        replacements = {}
        for anchor, group in snapshot.groupby("anchor_A", sort=False):
            if selected_side is not None:
                selected = group.index[group["side"].eq(selected_side)][0]
            else:
                selected = group["unrealized_pnl"].idxmin()
                if snapshot.at[selected, "unrealized_pnl"] >= 0 or open_pnl.at[selected] >= 0:
                    continue
            kept = group.index[group.index != selected][0]
            ticker = next(t for t in self.members[anchor] if t not in group["ticker"].values)
            replacements[selected] = (kept, ticker)
        return replacements

    @staticmethod
    def _trade(position, shares, price, closing, reason, check_date, entry_date):
        side = position["side"]
        action = ("sell" if side == "long" else "cover") if closing else ("buy" if side == "long" else "short")
        trade_shares = -shares if closing else shares
        return {"entry_date": entry_date, "check_date": check_date,
                "anchor_A": position["anchor_A"], "side": side, "ticker": position["ticker"],
                "action": action, "shares": int(trade_shares), "Open": float(price),
                "amount": float(trade_shares * price),
                "realized_pnl": float(shares * (price - position["cost_price"])) if closing else 0.0,
                "reason": reason}

    def run(self):
        """Return event-only history, trades, schedule and final holdings in memory.

        History starts with initial Open holdings. Other rows show pre-trade
        holdings marked at check_date Close and their next-Open outcomes.
        entry_date is the event's execution date for every row, including
        retained assets. Actual position cost dates stay in final holdings.
        Each call starts again from the original supplied positions.
        """
        holdings = self.initial_holdings.copy()
        self._refresh_betas(self.entry_date - pd.Timedelta(days=1))
        pairs = [self._current_pairs(holdings, self.entry_date, self.opens.iloc[0])]
        initial = self._snapshot(holdings, self.entry_date, self.entry_date, self.opens.iloc[0])
        initial["action"] = "open"
        snapshots, trades, schedule = {self.entry_date: initial}, [], []
        for event in self._events():
            entry_date, check_date = event["entry_date"], event["check_date"]
            snapshot = self._snapshot(holdings, entry_date, check_date, self.closes.loc[check_date])
            next_open = self.opens.loc[entry_date]
            open_pnl = holdings["shares"] * (holdings["ticker"].map(next_open) - holdings["cost_price"])
            replacements = self._replacements(snapshot, open_pnl)
            row = dict(event, replacements=len(replacements), scale=1.0)
            row["long_pnl"] = snapshot.loc[snapshot["side"].eq("long"), "unrealized_pnl"].sum()
            row["short_pnl"] = snapshot.loc[snapshot["side"].eq("short"), "unrealized_pnl"].sum()
            self._refresh_betas(check_date)
            planned = holdings.copy()
            for selected, (kept, ticker) in replacements.items():
                anchor = holdings.at[selected, "anchor_A"]
                beta = self.betas[anchor][holdings.at[kept, "ticker"], ticker]
                side_sign = 1 if holdings.at[selected, "side"] == "long" else -1
                new_basket = self._basket(anchor, ticker)
                planned.loc[selected, ["basket", "ticker", "shares", "cost_date", "cost_price"]] = [
                    new_basket,
                    ticker, side_sign * int(np.floor(abs(holdings.at[kept, "shares"]) * beta)),
                    entry_date, float(next_open[ticker]),
                ]
                snapshot.at[selected, "action"] = "replace"
                snapshot.at[selected, "replace_with"] = ticker
                snapshot.at[selected, "basket_after"] = new_basket
            if replacements or self.mode == "strict_one_leg":
                amounts = planned["shares"] * planned["ticker"].map(next_open)
                long_value = float(amounts.loc[planned["side"].eq("long")].sum())
                short_value = float(-amounts.loc[planned["side"].eq("short")].sum())
                row["scale"] = _exposure_scale(long_value, short_value, self.long_amount, self.mode)
            if row["scale"] < 1:
                # Scale retained positions too, including positions with unrealized gains.
                # Their partial closes can realize gains, recorded below as strict_cap.
                planned["shares"] = np.trunc(planned["shares"] * row["scale"]).astype(int)
            direction = planned["side"].map({"long": 1, "short": -1})
            if (planned["shares"] * direction <= 0).any():
                raise ValueError("The budget is too small for nonzero whole shares after replacement.")
            pairs.append(self._current_pairs(planned, entry_date, next_open))
            snapshot["shares_after"] = planned["shares"]
            for index, old in holdings.iterrows():
                new = planned.loc[index]
                if index in replacements:
                    reason = "loss_harvest" if self.selection_mode == "pair" else "portfolio_replace"
                    closing = self._trade(old, old["shares"], next_open[old["ticker"]], True, reason, check_date, entry_date)
                    trades.append(closing)
                    trades.append(self._trade(new, new["shares"], next_open[new["ticker"]], False, "replacement", check_date, entry_date))
                    snapshot.at[index, "realized_pnl"] = closing["realized_pnl"]
                elif abs(old["shares"]) > abs(new["shares"]):
                    # Keep the actual gain/loss sign; scaling is separate from loss selection.
                    closing = self._trade(old, old["shares"] - new["shares"], next_open[old["ticker"]], True, "strict_cap", check_date, entry_date)
                    trades.append(closing)
                    snapshot.at[index, "action"] = "trim"
                    snapshot.at[index, "realized_pnl"] = closing["realized_pnl"]
            snapshots[entry_date] = snapshot
            holdings = planned
            row["gross_after"] = float((holdings["shares"].abs() * holdings["ticker"].map(next_open)).sum())
            schedule.append(row)
        return {
            "history": pd.concat(snapshots.values(), ignore_index=True),
            "pairs": pd.concat(pairs, ignore_index=True),
            "snapshots": snapshots,
            "trades": pd.DataFrame(trades, columns=[
                "entry_date", "check_date", "anchor_A", "side", "ticker", "action", "shares", "Open", "amount", "realized_pnl", "reason",
            ]),
            "schedule": pd.DataFrame(schedule, columns=[
                "entry_date", "check_date", "period_end", "fallback", "replacements", "long_pnl", "short_pnl", "scale", "gross_after",
            ]),
            "positions": holdings,
        }

    def calculate_daily_positions(self, rebalance_result=None):
        """Value daily holdings, retaining all A/B/C tickers in triplet order."""
        if rebalance_result is None:
            rebalance_result = self.run()
        entry_open = self.opens.iloc[0] if "SPY" in self.opens.columns else None
        daily_long, daily_short, daily = calculate_daily_positions(
            self.closes, self.portfolio_a, self.portfolio_b, entry_open=entry_open,
            rebalance_history=rebalance_result["history"], open_prices=self.opens,
        )
        column_order = [ticker for triplet in self.members.values() for ticker in triplet]
        return (
            daily_long.reindex(columns=column_order, fill_value=0.0),
            daily_short.reindex(columns=column_order, fill_value=0.0),
            daily,
        )


def rebalance_portfolios(open_prices, close_prices, portfolio_a, portfolio_b,
                         pair_betas, rebalance_frequency=3, long_amount=1_000_000,
                         mode="beta", selection_mode="pair", beta_column="quote_beta",
                         close_history=None):
    """Call PortfolioRebalancer while retaining the original function interface."""
    return PortfolioRebalancer(
        open_prices, close_prices, portfolio_a, portfolio_b, pair_betas,
        rebalance_frequency, long_amount, mode, selection_mode, beta_column=beta_column,
        close_history=close_history,
    ).run()
