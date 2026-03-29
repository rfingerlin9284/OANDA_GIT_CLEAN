#!/usr/bin/env bash
set -euo pipefail

cd /home/rfing/RBOTZILLA_OANDA_CLEAN

echo 'Patch CLEAN for real compounding behavior: add watermark-based size growth, wake router earlier, relax hard stop, compile, restart, and print proof. Phoenix remains untouched.'

ts="$(date +%Y%m%d_%H%M%S)"
cp -v engine/trade_engine.py "logs/repair_snapshots/trade_engine.py.compound_patch.${ts}.bak"
cp -v engine/capital_router.py "logs/repair_snapshots/capital_router.py.compound_patch.${ts}.bak"
cp -v engine/trade_manager.py "logs/repair_snapshots/trade_manager.py.compound_patch.${ts}.bak"

chmod u+w engine/trade_engine.py engine/capital_router.py engine/trade_manager.py

python3 - <<'PY'
from pathlib import Path
import re

pr = Path("engine/capital_router.py")
sr = pr.read_text()

if "def compute_watermark_compounded_units(" not in sr:
    anchor = "def compute_compounded_units("
    idx = sr.find(anchor)
    if idx < 0:
        raise SystemExit("FAIL: compute_compounded_units not found")
    helper = '''
def compute_watermark_compounded_units(
    base_units: int,
    current_nav: float,
    initial_nav: float,
    watermark_nav: float,
    growth_exponent: float = 1.0,
    drawdown_floor_ratio: float = 1.0,
    max_growth_multiple: float = 3.0,
) -> int:
    """
    Growth sizing using the higher of initial NAV or watermark NAV.
    Default behavior never de-sizes below base units during drawdown.
    """
    try:
        base_units = int(base_units)
        current_nav = float(current_nav)
        initial_nav = float(initial_nav)
        watermark_nav = float(watermark_nav)
        growth_exponent = float(growth_exponent)
        drawdown_floor_ratio = float(drawdown_floor_ratio)
        max_growth_multiple = float(max_growth_multiple)
    except Exception:
        return int(base_units)

    if base_units <= 0:
        return 0

    anchor_nav = max(initial_nav, watermark_nav, 1.0)
    if current_nav <= 0:
        return int(base_units)

    ratio = (current_nav / anchor_nav) ** growth_exponent
    ratio = max(drawdown_floor_ratio, ratio)
    ratio = min(ratio, max_growth_multiple)

    return max(1, int(round(base_units * ratio)))


'''
    sr = sr[:idx] + helper + sr[idx:]

old_gate = '''        free_slots = max_positions - len(open_positions)
        if free_slots > 0:
            return None
'''
new_gate = '''        free_slots = max_positions - len(open_positions)
        router_min_open_positions = int(os.getenv("RBOT_ROUTER_MIN_OPEN_POSITIONS", "3"))
        if len(open_positions) < router_min_open_positions:
            return None
'''
if old_gate in sr:
    sr = sr.replace(old_gate, new_gate, 1)

pr.write_text(sr)

pe = Path("engine/trade_engine.py")
se = pe.read_text()

old_import = "from engine.capital_router import CapitalRouter, compute_compounded_units\n"
new_import = "from engine.capital_router import CapitalRouter, compute_compounded_units, compute_watermark_compounded_units\n"
if old_import in se and new_import not in se:
    se = se.replace(old_import, new_import, 1)

old_state = '''        self._initial_nav     = 0.0   # set in run() after first account query
        self._router: Optional[CapitalRouter] = None
'''
new_state = '''        self._initial_nav     = 0.0   # set in run() after first account query
        self._watermark_nav   = 0.0   # highest observed NAV since engine start
        self._router: Optional[CapitalRouter] = None
'''
if old_state in se:
    se = se.replace(old_state, new_state, 1)

old_nav_init = '''            _acct = self.connector.get_account_info()
            self._initial_nav = _acct.balance + _acct.unrealized_pl
            self._router = CapitalRouter(self.connector, initial_nav=self._initial_nav)
            print(f"  ✅ CapitalRouter ACTIVE  initial_nav=${self._initial_nav:,.2f}")
'''
new_nav_init = '''            _acct = self.connector.get_account_info()
            self._initial_nav = _acct.balance + _acct.unrealized_pl
            self._watermark_nav = self._initial_nav
            self._router = CapitalRouter(self.connector, initial_nav=self._initial_nav)
            print(f"  ✅ CapitalRouter ACTIVE  initial_nav=${self._initial_nav:,.2f}  watermark=${self._watermark_nav:,.2f}")
'''
if old_nav_init in se:
    se = se.replace(old_nav_init, new_nav_init, 1)

loop_old = "                placed = await self._run_scan_cycle()\n"
loop_new = '''                try:
                    _acct_live = self.connector.get_account_info()
                    _nav_live = _acct_live.balance + _acct_live.unrealized_pl
                    if _nav_live > self._watermark_nav:
                        self._watermark_nav = _nav_live
                except Exception:
                    pass

                placed = await self._run_scan_cycle()
'''
if loop_old in se and "_nav_live = _acct_live.balance + _acct_live.unrealized_pl" not in se:
    se = se.replace(loop_old, loop_new, 1)

pat = re.compile(
    r'def _compute_units\\(self,\\s*symbol:\\s*str,\\s*sig:\\s*AggregatedSignal,\\s*nav:\\s*float(?:\\s*=\\s*0\\.0)?\\)\\s*->\\s*int:\\n(?:        .*\\n)+?(?=\\n    # ── Main loop)',
    re.M
)
m = pat.search(se)
if not m:
    raise SystemExit("FAIL: _compute_units block not found")

replacement = '''def _compute_units(self, symbol: str, sig: AggregatedSignal, nav: float = 0.0) -> int:
        """
        Sizing stack:
        1) start from RBOT_BASE_UNITS
        2) apply watermark-based compounding without shrinking below base by default
        3) enforce Charter notional floor with broker USD-notional math
        """
        base_units = int(os.getenv("RBOT_BASE_UNITS", "14000"))
        growth_exponent = float(os.getenv("RBOT_COMPOUND_GROWTH_EXPONENT", "1.15"))
        drawdown_floor_ratio = float(os.getenv("RBOT_COMPOUND_DRAWDOWN_FLOOR_RATIO", "1.00"))
        max_growth_multiple = float(os.getenv("RBOT_COMPOUND_MAX_GROWTH_MULTIPLE", "3.00"))

        scaled = base_units
        if nav > 0 and self._initial_nav > 0:
            scaled = compute_watermark_compounded_units(
                base_units=base_units,
                current_nav=nav,
                initial_nav=self._initial_nav,
                watermark_nav=(self._watermark_nav or self._initial_nav),
                growth_exponent=growth_exponent,
                drawdown_floor_ratio=drawdown_floor_ratio,
                max_growth_multiple=max_growth_multiple,
            )

        units = scaled if sig.direction == "BUY" else -scaled

        live_prices = self.connector.get_live_prices([symbol]) or {}
        live_mid = (live_prices.get(symbol) or {}).get("mid", 0.0)
        try:
            live_mid = float(live_mid or 0.0)
        except Exception:
            live_mid = 0.0

        units = int(self._apply_min_notional_floor(symbol, units, live_mid))
        return units
'''
se = se[:m.start()] + replacement + se[m.end():]
pe.write_text(se)

pm = Path("engine/trade_manager.py")
sm = pm.read_text()
if 'self._hard_stop_usd = abs(float(os.getenv("RBOT_MAX_LOSS_USD_PER_TRADE", "30")))' in sm:
    sm = sm.replace(
        'self._hard_stop_usd = abs(float(os.getenv("RBOT_MAX_LOSS_USD_PER_TRADE", "30")))',
        'self._hard_stop_usd = abs(float(os.getenv("RBOT_MAX_LOSS_USD_PER_TRADE", "45")))',
        1
    )
pm.write_text(sm)

print("OK: true compounding patches applied")
PY

chmod 444 engine/trade_engine.py engine/capital_router.py engine/trade_manager.py

echo
echo '=== COMPILE CHECK ==='
.venv/bin/python -m py_compile \
    engine/trade_engine.py \
    engine/capital_router.py \
    engine/trade_manager.py \
    brokers/oanda_connector.py \
    engine/trail_logic.py

echo
echo '=== RESTART ENGINE ==='
bash scripts/restart.sh
sleep 12

echo
echo '=== PROOF: WATERMARK COMPOUNDING ==='
grep -nE 'compute_watermark_compounded_units|_watermark_nav|RBOT_COMPOUND_GROWTH_EXPONENT|RBOT_COMPOUND_DRAWDOWN_FLOOR_RATIO|RBOT_COMPOUND_MAX_GROWTH_MULTIPLE' engine/trade_engine.py engine/capital_router.py || true

echo
echo '=== PROOF: ROUTER WAKEUP ==='
grep -nE 'RBOT_ROUTER_MIN_OPEN_POSITIONS|len\(open_positions\) < router_min_open_positions' engine/capital_router.py || true

echo
echo '=== PROOF: HARD STOP DEFAULT ==='
grep -n 'RBOT_MAX_LOSS_USD_PER_TRADE' engine/trade_manager.py || true

echo
echo '=== LAST 120 IMPORTANT LOG LINES ==='
tail -n 120 logs/engine_continuous.out | grep -E 'CapitalRouter ACTIVE|watermark=|CANDIDATE|→ Placing|OPENED|CLOSED|GREEN_LOCK|BREAK_EVEN|STAGNANT|HARD_DOLLAR_STOP|PROFIT_TARGET_CLOSE|ORDER REJECTED|Engine cycle error|ROUTER' || true
