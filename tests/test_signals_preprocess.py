"""
test_signals_preprocess.py — 信号预处理、中性化与风控扩展单元测试。
"""

import unittest
import numpy as np
import pandas as pd

from signals.preprocess import (
    winsorize_mad,
    standardize_zscore,
    neutralize,
    preprocess_factor_pipeline,
)
from risk.portfolio import detect_limit_moves, apply_volatility_target


class TestSignalsPreprocess(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dates = pd.date_range("2023-01-01", periods=20, freq="B")
        self.symbols = [f"stock_{i:02d}" for i in range(30)]

        # 构造带极值的因子矩阵
        vals = np.random.randn(len(self.dates), len(self.symbols))
        vals[0, 0] = 100.0  # 极大异常值
        vals[0, 1] = -100.0 # 极小异常值
        self.factor_df = pd.DataFrame(vals, index=self.dates, columns=self.symbols)

        # 构造行业与市值矩阵
        ind_list = ["Bank", "Tech", "Medical", "Consumer", "Energy"] * 6
        self.industry_matrix = pd.DataFrame(
            [ind_list] * len(self.dates), index=self.dates, columns=self.symbols
        )
        self.market_cap_df = pd.DataFrame(
            np.random.uniform(1e9, 1e11, (len(self.dates), len(self.symbols))),
            index=self.dates,
            columns=self.symbols,
        )

    def test_winsorize_mad(self):
        df_win = winsorize_mad(self.factor_df, n=3.0)
        self.assertLess(df_win.iloc[0, 0], 50.0)
        self.assertGreater(df_win.iloc[0, 1], -50.0)

    def test_standardize_zscore(self):
        df_z = standardize_zscore(self.factor_df)
        np.testing.assert_allclose(df_z.mean(axis=1).values, 0.0, atol=1e-7)
        np.testing.assert_allclose(df_z.std(axis=1).values, 1.0, atol=1e-5)

    def test_neutralize_orthogonal_to_size(self):
        # 构造强市值相关的因子
        log_mcap = np.log(self.market_cap_df)
        biased_factor = log_mcap * 2.0 + np.random.randn(*log_mcap.shape) * 0.1
        neutral_factor = neutralize(biased_factor, market_cap_df=self.market_cap_df)

        # 检验中性化后因子与对数市值的截面相关性显著接近 0
        corrs = [neutral_factor.loc[d].corr(log_mcap.loc[d]) for d in self.dates]
        mean_corr = np.nanmean(np.abs(corrs))
        self.assertLess(mean_corr, 1e-5)

    def test_preprocess_factor_pipeline(self):
        df_pipe = preprocess_factor_pipeline(
            self.factor_df,
            industry_matrix=self.industry_matrix,
            market_cap_df=self.market_cap_df,
        )
        self.assertEqual(df_pipe.shape, self.factor_df.shape)
        np.testing.assert_allclose(df_pipe.mean(axis=1).values, 0.0, atol=1e-7)

    def test_detect_limit_moves(self):
        open_m = pd.DataFrame([[11.0, 9.0]], index=self.dates[:1], columns=["000001", "300001"])
        high_m = pd.DataFrame([[11.0, 9.0]], index=self.dates[:1], columns=["000001", "300001"])
        low_m = pd.DataFrame([[11.0, 9.0]], index=self.dates[:1], columns=["000001", "300001"])
        pre_close = pd.DataFrame([[10.0, 10.0]], index=self.dates[:1], columns=["000001", "300001"])

        up_lock, down_lock = detect_limit_moves(open_m, high_m, low_m, pre_close)
        self.assertTrue(up_lock.loc[self.dates[0], "000001"])  # 10% 一字涨停
        self.assertFalse(up_lock.loc[self.dates[0], "300001"]) # 创业板 20% 限额，9 元不是涨停

    def test_apply_volatility_target(self):
        rets = pd.Series(np.random.randn(100) * 0.02, index=pd.date_range("2023-01-01", periods=100))
        vol_df = apply_volatility_target(rets, target_vol=0.08, max_leverage=2.0)
        self.assertIn("leverage", vol_df.columns)
        self.assertIn("targeted_ret", vol_df.columns)
        self.assertLessEqual(vol_df["leverage"].max(), 2.0)


if __name__ == "__main__":
    unittest.main()
