"""Transaction-cost model for backtests (spreads, slippage, commissions, regulatory fees)."""
from __future__ import annotations

from dataclasses import dataclass

from ..config import BacktestCfg


@dataclass
class CostModel:
    stock_slippage_bps: float
    stock_spread_bps: float
    stock_commission_per_share: float
    option_half_spread_pct: float
    option_slippage_pct: float
    option_fee_per_contract: float   # commission + regulatory, per contract per side

    @classmethod
    def from_cfg(cls, c: BacktestCfg) -> "CostModel":
        return cls(c.stock_slippage_bps, c.stock_spread_bps, c.stock_commission_per_share,
                   c.option_half_spread_pct, c.option_slippage_pct,
                   c.option_commission_per_contract + c.option_regulatory_fee_per_contract)

    # stock: price per share actually paid / received
    def stock_buy(self, px: float) -> float:
        return px * (1 + (self.stock_slippage_bps + self.stock_spread_bps / 2) / 10_000) + self.stock_commission_per_share

    def stock_sell(self, px: float) -> float:
        return px * (1 - (self.stock_slippage_bps + self.stock_spread_bps / 2) / 10_000) - self.stock_commission_per_share

    # options: premium per share-equivalent actually paid / received (fees spread over the 100 multiplier)
    def option_buy(self, mid: float) -> float:
        return mid * (1 + (self.option_half_spread_pct + self.option_slippage_pct) / 100) + self.option_fee_per_contract / 100

    def option_sell(self, mid: float) -> float:
        return max(0.0, mid * (1 - (self.option_half_spread_pct + self.option_slippage_pct) / 100) - self.option_fee_per_contract / 100)
