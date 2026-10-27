"""Public weighting choices, arithmetic and unchanged selected-model dispatch."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from methods.aqr_factors.construction import ConstructionConfig, PortfolioConstructor
from methods.aqr_factors.run import (load_parameters, parameter_fingerprint,
                                    portfolio_targets, run, weighting_method)


class WeightingMethodTests(unittest.TestCase):
    def example(self, values=(-5., -15., -66., -85., -25.)):
        dates = pd.DatetimeIndex(["2022-01-31"])
        tickers = list("ABCDE")
        raw = pd.DataFrame(dict(as_of=dates[0], ticker=tickers, alpha=values,
            score_group="Tech", in_universe=True, input_status="ok", close=10., market_cap=100.))
        parameters = dict(weighting_method=1, alpha_column="alpha",
            portfolio=dict(method="demean_l1", group_by=None, min_group_size=2),
            signals=dict(group_by="sector"))
        return parameters, raw, tickers, dates, np.zeros((1, 5), int), np.full((1, 5), np.nan), np.full((1, 5), np.nan)

    def test_original_five_stock_alpha_without_ranking_or_risk_overlays(self):
        args = self.example()
        with patch("methods.aqr_factors.run._model_targets", side_effect=AssertionError("Wrong branch")):
            weights = portfolio_targets(*args)[0]
        np.testing.assert_allclose(weights, np.array([34.2, 24.2, -26.8, -45.8, 14.2]) / 145.2,
                                   atol=1e-14, rtol=0)
        self.assertAlmostEqual(weights[weights > 0].sum(), .5)
        self.assertAlmostEqual(weights[weights < 0].sum(), -.5)
        # No 5% cap is silently applied to the supplied alpha formula.
        self.assertGreater(abs(weights).max(), .3)

    def test_translation_scale_and_constant_alpha(self):
        original = self.example()
        reference = portfolio_targets(*original)
        shifted = self.example(tuple(10 * np.array([-5., -15., -66., -85., -25.]) + 1000))
        np.testing.assert_allclose(portfolio_targets(*shifted), reference, atol=1e-14)
        np.testing.assert_array_equal(portfolio_targets(*self.example((2.,) * 5)), np.zeros((1, 5)))

    def test_missing_alpha_excluded_only_on_its_date(self):
        parameters, raw, tickers, dates, groups, vol, beta = self.example()
        later = raw.assign(as_of=pd.Timestamp("2022-02-28"))
        raw.loc[0, "alpha"] = np.nan
        combined = pd.concat([raw, later], ignore_index=True)
        weights = portfolio_targets(parameters, combined, tickers,
            dates.append(pd.DatetimeIndex(["2022-02-28"])),
            np.repeat(groups, 2, axis=0), np.repeat(vol, 2, axis=0), np.repeat(beta, 2, axis=0))
        self.assertEqual(weights[0, 0], 0.)
        self.assertNotEqual(weights[1, 0], 0.)
        np.testing.assert_allclose(abs(weights).sum(axis=1), 1.)

    def test_four_stock_rank_example_explains_zero_mean_and_denominator(self):
        cf_rank = np.array([4., 3., 2., 1.])
        quality_rank = np.array([2., 4., 1., 3.])
        zcf = (cf_rank - 2.5) / np.sqrt(1.25)
        zquality = (quality_rank - 2.5) / np.sqrt(1.25)
        score = .6 * zcf + .4 * zquality
        self.assertAlmostEqual(score.mean(), 0.)
        self.assertAlmostEqual(abs(score).sum(), 2.862167011199731)
        raw = pd.DataFrame(dict(as_of=pd.Timestamp("2022-01-31"), ticker=list("ABCD"), score=score))
        actual = PortfolioConstructor(ConstructionConfig()).build(raw).set_index("ticker")
        np.testing.assert_allclose(actual.loc[list("ABCD"), "target_weight"],
                                   [.21875, .28125, -.28125, -.21875], atol=1e-14)

    def test_choice_two_uses_exact_existing_model_path(self):
        args = list(self.example())
        args[0]["weighting_method"] = 2
        args[0]["model"] = {"test": "existing model"}
        expected = np.array([[.1, .2, -.2, -.3, .2]])
        with patch("methods.aqr_factors.run._model_targets", return_value=expected) as build:
            self.assertIs(portfolio_targets(*args), expected)
            self.assertIs(build.call_args.args[0], args[0]["model"])

    def test_reserved_choice_rejected_before_loading_any_data(self):
        with self.assertRaisesRegex(NotImplementedError, "reserved"):
            run({"weighting_method": 3})
        for invalid in [0, 4, True, "1", 1.5]:
            with self.assertRaises(ValueError):
                weighting_method({"weighting_method": invalid})

    def test_existing_result_remains_valid_but_simple_choice_is_distinct(self):
        parameters, path = load_parameters()
        saved = json.loads(path.with_name("result.json").read_text(encoding="utf-8"))
        self.assertEqual(parameters["weighting_method"], 2)
        self.assertEqual(saved["parameters_sha256"], parameter_fingerprint(parameters))
        legacy = copy.deepcopy(parameters)
        del legacy["weighting_method"], legacy["alpha_column"]
        self.assertEqual(parameter_fingerprint(legacy), parameter_fingerprint(parameters))
        changed = dict(parameters, weighting_method=1)
        self.assertNotEqual(parameter_fingerprint(changed), parameter_fingerprint(parameters))
        self.assertNotEqual(parameter_fingerprint(changed),
                            parameter_fingerprint(dict(changed, alpha_column="cashflow_yield")))


if __name__ == "__main__":
    unittest.main()
