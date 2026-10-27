"""Preset precedence and stale-result detection, without data downloads or replays."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from methods.aqr_factors.run import load_parameters, parameter_fingerprint
from methods.aqr_factors.settings import AQRSettings


class ConfigLoadingTests(unittest.TestCase):
    def test_none_uses_selected_monthly_preset(self):
        parameters, path = load_parameters()
        self.assertEqual(path.name, "parameters.py")
        self.assertEqual(parameters["backtest"]["rebalance_frequency"], "monthly")
        self.assertEqual(parameters["performance"]["risk_free"]["source"], "fred_dgs1mo")

    def test_explicit_fields_override_and_missing_fields_use_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "parameters.py"
            path.write_text("PARAMETERS = {'backtest': {'borrow_fee_rate': 0.025}, "
                            "'performance': {'risk_free': {'max_quote_age_days': 3}}}", encoding="utf-8")
            with patch.object(AQRSettings, "selected_parameter_file", str(path)):
                parameters, resolved = load_parameters()
            self.assertEqual(resolved, path.resolve())
            self.assertEqual(parameters["backtest"]["borrow_fee_rate"], 0.025)
            self.assertEqual(parameters["backtest"]["initial_nav"], AQRSettings.initial_nav)
            self.assertEqual(parameters["performance"]["risk_free"]["max_quote_age_days"], 3)
            self.assertEqual(parameters["performance"]["risk_free"]["day_count"], 365)

    def test_missing_explicit_file_does_not_switch_models(self):
        with self.assertRaises(FileNotFoundError):
            load_parameters("result/aqr/nonexistent/parameters.py")

    def test_fingerprint_detects_rf_and_execution_edits(self):
        parameters, _ = load_parameters()
        original = parameter_fingerprint(parameters)
        changed = copy.deepcopy(parameters)
        changed["performance"]["risk_free"]["max_quote_age_days"] = 3
        self.assertNotEqual(original, parameter_fingerprint(changed))
        changed = copy.deepcopy(parameters)
        changed["backtest"]["borrow_fee_rate"] = .02
        self.assertNotEqual(original, parameter_fingerprint(changed))
        self.assertEqual(original, parameter_fingerprint(dict(reversed(list(parameters.items())))))


if __name__ == "__main__":
    unittest.main()
