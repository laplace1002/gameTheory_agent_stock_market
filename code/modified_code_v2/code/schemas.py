from __future__ import annotations


ALLOWED_SIDES = {"BUY", "SELL", "HOLD"}
ALLOWED_DIRECTIONS = {"bullish", "bearish", "neutral"}


def validate_llm_plan(plan: dict, agent_id: str, strategy_identity: str, valid_tickers: set[str]) -> list[str]:
    violations = []
    if not isinstance(plan, dict):
        return ["plan must be a JSON object"]
    if str(plan.get("agent_id", agent_id)) != agent_id:
        violations.append("agent_id does not match caller")
    if str(plan.get("strategy_identity", strategy_identity)) != strategy_identity:
        violations.append("strategy_identity drift")

    trade = plan.get("trade", {})
    if isinstance(trade, dict):
        side = str(trade.get("side", "HOLD")).upper()
        ticker = str(trade.get("ticker", "")).upper()
        if side not in ALLOWED_SIDES:
            violations.append("trade.side must be BUY, SELL, or HOLD")
        if ticker and ticker not in valid_tickers:
            violations.append(f"unknown trade ticker {ticker}")
    else:
        violations.append("trade must be an object")

    forecast = plan.get("forecast", [])
    if forecast in ("", None):
        forecast = []
    if not isinstance(forecast, list):
        violations.append("forecast must be a list")
    else:
        for index, item in enumerate(forecast):
            if not isinstance(item, dict):
                violations.append(f"forecast[{index}] must be an object")
                continue
            ticker = str(item.get("ticker", "")).upper()
            direction = str(item.get("direction", "neutral"))
            if ticker not in valid_tickers:
                violations.append(f"forecast[{index}] unknown ticker {ticker}")
            if direction not in ALLOWED_DIRECTIONS:
                violations.append(f"forecast[{index}] invalid direction")
            for key in ["probability_up", "confidence"]:
                try:
                    value = float(item.get(key, 0.5))
                except (TypeError, ValueError):
                    violations.append(f"forecast[{index}].{key} must be numeric")
                    continue
                if value < 0.0 or value > 1.0:
                    violations.append(f"forecast[{index}].{key} must be in [0, 1]")

    target_weights = plan.get("target_weights", {})
    if target_weights in ("", None):
        target_weights = {}
    if not isinstance(target_weights, dict):
        violations.append("target_weights must be an object")
    else:
        total = 0.0
        for ticker, value in target_weights.items():
            ticker = str(ticker).upper()
            if ticker not in valid_tickers:
                violations.append(f"target_weights unknown ticker {ticker}")
            try:
                weight = float(value)
            except (TypeError, ValueError):
                violations.append(f"target_weights[{ticker}] must be numeric")
                continue
            total += max(0.0, weight)
            if weight < 0.0 or weight > 0.35:
                violations.append(f"target_weights[{ticker}] outside [0, 0.35]")
        if total > 0.950001:
            violations.append("target_weights sum exceeds 0.95")
    return violations
