import numpy as np


def compute_resource_score(cpu, ram, battery, internet_speed, bandwidth):
    """Combine simulated resource metrics into a single raw score (Ri)."""
    return (cpu * 0.25) + (ram * 0.2) + (battery * 0.15) + (internet_speed * 0.2) + (bandwidth * 0.2)


def normalize_scores(raw_scores):
    """Min-max normalize a list of raw resource scores across all clients."""
    raw_scores = np.array(raw_scores, dtype=np.float32)
    r_min = raw_scores.min()
    r_max = raw_scores.max()
    eps = 1e-8
    norm_scores = (raw_scores - r_min) / (r_max - r_min + eps)
    return norm_scores


def compute_historical_trust(behavior_history):
    """
    H = (sum of honest-round indicators) / (total rounds)
    behavior_history: list of 1s (honest) and 0s (malicious) for a client across rounds.
    """
    if len(behavior_history) == 0:
        return 1.0  # no history yet, assume trustworthy
    return sum(behavior_history) / len(behavior_history)


def compute_current_trust(r_norm, historical_trust):
    """
    Trust_curr = (Rgood / (Rgood + Rbad)) * Rnorm
    Rgood = H, Rbad = (1 - H)
    """
    r_good = historical_trust
    r_bad = 1.0 - historical_trust
    denom = r_good + r_bad  # always 1.0, but kept explicit per the paper's formula
    trust_curr = (r_good / denom) * r_norm
    return trust_curr


def update_trust(prev_trust, curr_trust):
    """Smooth trust over time: T(t+1) = (T(t-1) + T(t)) / 2"""
    return (prev_trust + curr_trust) / 2.0


def compute_threshold(trust_scores):
    """Threshold = average trust score across all clients."""
    return float(np.mean(trust_scores))


def select_clients(trust_scores, threshold):
    """Return indices of clients whose trust score meets or exceeds the threshold."""
    return [i for i, t in enumerate(trust_scores) if t >= threshold]


def trust_weighted_average(params_list, trust_scores, selected_indices):
    """
    Aggregate model parameters from selected clients, weighted by their trust scores.
    params_list: list of parameter arrays (one per client, each a list of numpy arrays for model layers)
    trust_scores: list of trust scores (all clients)
    selected_indices: indices of clients that passed the threshold
    """
    selected_trusts = [trust_scores[i] for i in selected_indices]
    total_trust = sum(selected_trusts) + 1e-8

    # Initialize aggregated params as zeros, matching shape of first selected client's params
    aggregated = [np.zeros_like(layer) for layer in params_list[selected_indices[0]]]

    for idx in selected_indices:
        weight = trust_scores[idx] / total_trust
        client_params = params_list[idx]
        for layer_idx, layer in enumerate(client_params):
            aggregated[layer_idx] += weight * layer

    return aggregated
    
def generate_behavior_histories(num_clients, num_rounds, seed=42):
    rng = np.random.default_rng(seed)
    histories = []
    for i in range(num_clients):
        if i == 1:
            history = [int(rng.random() < 0.2) for _ in range(num_rounds)]
        else:
            history = [int(rng.random() < 0.95) for _ in range(num_rounds)]
        histories.append(history)
    return histories