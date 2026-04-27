# tests/test_main.py
import sys
import os
import unittest.mock as mock
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_collect_subcommand_calls_collect_main():
    import __main__ as m
    with mock.patch("collect.main") as mock_main:
        with mock.patch("sys.argv", ["grid_prophet", "collect"]):
            m.main()
    mock_main.assert_called_once()


def test_features_subcommand_calls_features_main():
    import __main__ as m
    with mock.patch("features.main") as mock_main:
        with mock.patch("sys.argv", ["grid_prophet", "features"]):
            m.main()
    mock_main.assert_called_once()


def test_train_subcommand_calls_train_main():
    import __main__ as m
    with mock.patch("train.main") as mock_main:
        with mock.patch("sys.argv", ["grid_prophet", "train"]):
            m.main()
    mock_main.assert_called_once()


def test_predict_subcommand_calls_predict_main():
    import __main__ as m
    with mock.patch("predict.main") as mock_main:
        with mock.patch("sys.argv", ["grid_prophet", "predict"]):
            m.main()
    mock_main.assert_called_once()


def test_run_subcommand_calls_all_four():
    import __main__ as m
    with mock.patch("collect.main") as mc, \
         mock.patch("features.main") as mf, \
         mock.patch("train.main") as mt, \
         mock.patch("predict.main") as mp:
        with mock.patch("sys.argv", ["grid_prophet", "run"]):
            m.main()
    mc.assert_called_once()
    mf.assert_called_once()
    mt.assert_called_once()
    mp.assert_called_once()


def test_unknown_subcommand_exits():
    import __main__ as m
    with mock.patch("sys.argv", ["grid_prophet", "bogus"]):
        with pytest.raises(SystemExit):
            m.main()
