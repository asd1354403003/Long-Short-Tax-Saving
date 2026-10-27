"""Independent ranked long/short lists; does not claim to estimate fair value."""
import pandas as pd
from selection import PoolSelection, independent_pool
from spot_fx import SpotFX


class Selector:
    def __init__(self, parameters, config, result_dir):
        self.parameters = parameters

    def select(self, *, rebuild=False):
        params = self.parameters
        rows = []
        for side in ("long", "short"):
            names = params[side + "_tickers"]
            if isinstance(names, str):
                raise ValueError(side + "_tickers must be a list.")
            for rank, ticker in enumerate(names):
                ticker = ticker.upper().strip()
                rows.append(dict(as_of=params["as_of"], ticker=ticker, side=side,
                    score=len(names)-rank, instrument_type="FX" if ticker in SpotFX.rates else "EQUITY",
                    unit_step=1000 if ticker in SpotFX.rates else 1))
        pool = independent_pool(pd.DataFrame(rows, columns=["as_of", "ticker", "side", "score", "instrument_type", "unit_step"]))
        return PoolSelection("independent", pool, dict(method="manual", estimated_fair_value=False))
