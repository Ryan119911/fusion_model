"""Limit-aware angular representatives for UR joint-space planning."""
import numpy as np


def equivalents(value, lower, upper):
    period = 2 * np.pi
    first = int(np.ceil((lower - value - 1e-9) / period))
    last = int(np.floor((upper - value + 1e-9) / period))
    return value + period * np.arange(first, last + 1)


def nearest_equivalent(values, reference, lower, upper):
    result = np.asarray(values, float).copy()
    for j, value in enumerate(result):
        choices = equivalents(value, lower[j], upper[j])
        if not len(choices):
            raise ValueError(f"joint {j + 1} has no limit-valid equivalent")
        result[j] = choices[np.argmin(abs(choices - reference[j]))]
    return result


def continuous_joint_path(values, initial, lower, upper, max_step=np.pi):
    """Global squared-motion search; initial approach is explicitly exempt.

    Internal transitions exceeding max_step are infeasible, not merely slowed.
    Returned angles preserve every endpoint FK. Recheck swept collisions after
    this choice because different revolutions have different swept paths.
    """
    values = np.asarray(values, float)
    output = values.copy()
    for j in range(values.shape[1]):
        options = [equivalents(v, lower[j], upper[j]) for v in values[:, j]]
        if any(len(v) == 0 for v in options):
            raise ValueError(f"joint {j + 1} has no limit-valid equivalent")
        costs = (options[0] - initial[j]) ** 2
        if abs(values[0, j] - initial[j]) < 1e-9:
            costs[abs(options[0] - initial[j]) > 1e-9] = np.inf
        parents = []
        for i, (left, right) in enumerate(zip(options[:-1], options[1:]), 1):
            delta = right[None, :] - left[:, None]
            scores = costs[:, None] + delta ** 2
            # Target 0 is the inserted initial posture; target 1 is approach.
            if i > 1:
                scores[abs(delta) > max_step + 1e-9] = np.inf
            parents.append(scores.argmin(axis=0))
            costs = scores.min(axis=0)
            if not np.isfinite(costs).any():
                raise ValueError(
                    f"joint continuity rejected at target={i}, joint={j + 1}: "
                    "no continuous limit-valid angular path; replan IK branch"
                )
        k = int(costs.argmin())
        output[-1, j] = options[-1][k]
        for i in range(len(parents) - 1, -1, -1):
            k = int(parents[i][k])
            output[i, j] = options[i][k]
    return output
