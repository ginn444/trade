"""Acceptance criteria. A strategy may trade autonomously only if its research report passes.

Criteria are applied to OUT-OF-SAMPLE walk-forward results (not in-sample), plus a robustness
check that out-of-sample Sharpe has not collapsed relative to in-sample. Reports produced on
SYNTHETIC data are recorded but never authorize paper/live trading (enforced in the bot).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ..config import AcceptanceCfg


def evaluate(report: dict, acc: AcceptanceCfg) -> tuple[bool, list[str]]:
    if "error" in report:
        return False, [report["error"]]
    wf = report["walk_forward"]
    ins = report["in_sample"]
    bench = report["benchmark_buy_hold"]
    fails = []
    if wf["trades"] < acc.min_trades:
        fails.append(f"OOS trades {wf['trades']} < {acc.min_trades}")
    if wf["profit_factor"] < acc.min_profit_factor:
        fails.append(f"OOS profit factor {wf['profit_factor']} < {acc.min_profit_factor}")
    if wf["max_drawdown_pct"] > acc.max_drawdown_pct:
        fails.append(f"OOS max drawdown {wf['max_drawdown_pct']}% > {acc.max_drawdown_pct}%")
    if wf["expectancy_r"] < acc.min_expectancy_r:
        fails.append(f"OOS expectancy {wf['expectancy_r']}R < {acc.min_expectancy_r}R")
    if wf["sharpe"] < acc.min_sharpe:
        fails.append(f"OOS Sharpe {wf['sharpe']} < {acc.min_sharpe}")
    if wf["positive_windows_frac"] < acc.min_positive_wf_windows_frac:
        fails.append(f"positive walk-forward windows {wf['positive_windows_frac']} < {acc.min_positive_wf_windows_frac}")
    if ins["sharpe"] > 0 and wf["sharpe"] < (1 - acc.max_oos_degradation) * ins["sharpe"]:
        fails.append(f"OOS Sharpe {wf['sharpe']} degraded >{acc.max_oos_degradation:.0%} from in-sample {ins['sharpe']}")
    if acc.require_beat_benchmark_sharpe and wf["sharpe"] <= bench["sharpe"]:
        fails.append(f"OOS Sharpe {wf['sharpe']} does not beat buy&hold {bench['sharpe']}")
    return not fails, fails


def load_approvals(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def record_result(path: Path, report: dict, approved: bool, reasons: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = load_approvals(path)
    wf = report.get("walk_forward", {})
    data[report["strategy"]] = {
        "approved": approved,
        "reasons": reasons,
        "data_source": report.get("data_source"),
        "option_pricing": report.get("option_pricing"),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "oos": {k: wf.get(k) for k in ("trades", "win_rate", "profit_factor", "expectancy_r", "max_drawdown_pct",
                                       "sharpe", "total_return_pct", "positive_windows_frac")},
        "period": report.get("period"),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)
