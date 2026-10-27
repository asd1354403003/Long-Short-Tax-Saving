"""A curated spot-FX universe normalized to USD per unit of base currency."""

import pandas as pd


class SpotFX:
    """Use one USD quote direction; never pair a rate with its own inverse.

    Yahoo's USD/JPY, for example, is inverted to JPY/USD before OLS and P&L.
    Quantities are base-currency units, not shares. Screening/execution use the
    shared SPY session calendar; exchange-volume/market-cap filters do not apply.
    """

    # Canonical symbol: (Yahoo source symbol, reciprocal required, base currency).
    rates = {
        "EURUSD=X": ("EURUSD=X", False, "EUR"),
        "GBPUSD=X": ("GBPUSD=X", False, "GBP"),
        "AUDUSD=X": ("AUDUSD=X", False, "AUD"),
        "NZDUSD=X": ("NZDUSD=X", False, "NZD"),
        "CADUSD=X": ("CAD=X", True, "CAD"),
        "CHFUSD=X": ("CHF=X", True, "CHF"),
        "JPYUSD=X": ("JPY=X", True, "JPY"),
        "NOKUSD=X": ("NOK=X", True, "NOK"),
        "SEKUSD=X": ("SEK=X", True, "SEK"),
        "DKKUSD=X": ("DKK=X", True, "DKK"),
        "SGDUSD=X": ("SGD=X", True, "SGD"),
        "HKDUSD=X": ("HKD=X", True, "HKD"),
        "MXNUSD=X": ("MXN=X", True, "MXN"),
        "ZARUSD=X": ("ZAR=X", True, "ZAR"),
        "PLNUSD=X": ("PLN=X", True, "PLN"),
    }
    source_url = "https://finance.yahoo.com/markets/currencies/"

    @classmethod
    def universe(cls, unit_step=1000):
        return pd.DataFrame([
            {"ticker": ticker, "name": base + "/USD spot FX", "industry": "Spot FX: USD quotes",
             "instrument_type": "FX", "pair_family": "spot_fx", "asset_class": "currency",
             "exposure_group": "spot_fx_usd", "implementation": "spot_currency",
             "base_currency": base, "quote_currency": "USD", "unit_step": unit_step,
             "source_ticker": source, "price_transform": "reciprocal" if inverse else "identity",
             "source_urls": cls.source_url}
            for ticker, (source, inverse, base) in cls.rates.items()
        ])

    @classmethod
    def source_ticker(cls, ticker):
        return cls.rates.get(ticker, (ticker, False, ""))[0]

    @classmethod
    def normalize_prices(cls, ticker, frame):
        """Convert all OHLC fields consistently, including reciprocal High/Low."""
        if ticker not in cls.rates or not cls.rates[ticker][1]:
            return frame
        frame = frame.copy()
        for column in ("Open", "Close"):
            if column in frame:
                frame[column] = 1 / frame[column].where(frame[column] > 0)
        if {"High", "Low"}.issubset(frame.columns):
            high, low = frame.High.copy(), frame.Low.copy()
            frame["High"], frame["Low"] = 1 / low.where(low > 0), 1 / high.where(high > 0)
        return frame
