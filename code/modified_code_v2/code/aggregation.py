import math

import numpy as np
import pandas as pd

from message import DIRECTION_SCORE, Message, evidence_quality


def clamp_signal(value: float) -> float:
    return float(max(-1.0, min(1.0, value)))


def message_signal(msg: Message, current_date: str | None = None) -> float:
    age_decay = 1.0
    if current_date is not None:
        if msg.expires_at and pd.Timestamp(current_date) > pd.Timestamp(msg.expires_at):
            return 0.0
        age_days = max(0, (pd.Timestamp(current_date) - pd.Timestamp(msg.timestamp)).days)
        age_decay = math.exp(-age_days / 15.0)
    return clamp_signal(DIRECTION_SCORE.get(msg.direction, 0.0) * msg.confidence * evidence_quality(msg.evidence) * age_decay)


def reputation_weighted(own_signal, messages, reputation_tracker, current_date: str | None = None) -> float:
    own_signal = clamp_signal(float(own_signal))
    if not messages:
        return own_signal

    scores = []
    signals = []
    for msg in messages:
        signal = message_signal(msg, current_date=current_date)
        reputation = reputation_tracker.get_reputation(msg.sender_id)
        conflict = max(0.0, -own_signal * signal)
        age = 0.0
        if current_date is not None:
            age = max(0, (pd.Timestamp(current_date) - pd.Timestamp(msg.timestamp)).days) / 30.0
        score = 1.4 * reputation + 0.9 * msg.confidence + 0.7 * evidence_quality(msg.evidence) - 0.6 * conflict - 0.25 * age
        scores.append(score)
        signals.append(signal)

    weights = _softmax(scores)
    social_signal = float(np.dot(weights, signals))
    blended = 0.4 * own_signal + 0.6 * social_signal
    return clamp_signal(blended)


def degroot_consensus(belief_vector, adjacency_matrix, steps=10) -> np.ndarray:
    beliefs = np.asarray(belief_vector, dtype=float)
    adjacency = np.asarray(adjacency_matrix, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency_matrix must be square")
    if adjacency.shape[0] != beliefs.shape[0]:
        raise ValueError("belief_vector length must match adjacency_matrix size")

    row_sums = adjacency.sum(axis=1, keepdims=True)
    stochastic = np.divide(adjacency, row_sums, out=np.zeros_like(adjacency), where=row_sums > 1e-12)
    empty_rows = np.where(row_sums.flatten() <= 1e-12)[0]
    stochastic[empty_rows, empty_rows] = 1.0

    for _ in range(max(0, int(steps))):
        beliefs = stochastic @ beliefs
    return beliefs


def black_litterman(pi, sigma, P, q, omega, tau=0.05) -> np.ndarray:
    pi = np.asarray(pi, dtype=float).reshape(-1, 1)
    sigma = np.asarray(sigma, dtype=float)
    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1, 1)
    omega = np.asarray(omega, dtype=float)

    tau_sigma_inv = np.linalg.pinv(tau * sigma)
    omega_inv = np.linalg.pinv(omega)
    precision = tau_sigma_inv + P.T @ omega_inv @ P
    view_adjusted = tau_sigma_inv @ pi + P.T @ omega_inv @ q
    return (np.linalg.pinv(precision) @ view_adjusted).flatten()


def hedge_weights(agent_scores, max_drawdowns, calibration_errors, eta=1.0, xi=1.0, zeta=1.0) -> dict:
    agents = sorted(set(agent_scores) | set(max_drawdowns) | set(calibration_errors))
    if not agents:
        return {}

    logits = []
    for agent in agents:
        score = float(agent_scores.get(agent, 0.0))
        drawdown_penalty = abs(float(max_drawdowns.get(agent, 0.0)))
        calibration_penalty = float(calibration_errors.get(agent, 0.25))
        logits.append(eta * score - xi * drawdown_penalty - zeta * calibration_penalty)

    weights = _softmax(logits)
    return {agent: float(weight) for agent, weight in zip(agents, weights)}


def _softmax(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    shifted = values - np.nanmax(values)
    exp_values = np.exp(shifted)
    total = exp_values.sum()
    if total <= 1e-12:
        return np.ones_like(exp_values) / len(exp_values)
    return exp_values / total
