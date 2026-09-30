from trust import (
    compute_resource_score, normalize_scores, compute_historical_trust,
    compute_current_trust, update_trust, compute_threshold, select_clients
)

# Simulate 5 clients with random resource stats
raw_scores = [
    compute_resource_score(cpu=0.8, ram=0.7, battery=0.9, internet_speed=0.6, bandwidth=0.7),
    compute_resource_score(cpu=0.3, ram=0.4, battery=0.2, internet_speed=0.3, bandwidth=0.2),  # weak/malicious-like client
    compute_resource_score(cpu=0.9, ram=0.8, battery=0.95, internet_speed=0.85, bandwidth=0.9),
    compute_resource_score(cpu=0.6, ram=0.5, battery=0.6, internet_speed=0.5, bandwidth=0.6),
    compute_resource_score(cpu=0.7, ram=0.7, battery=0.7, internet_speed=0.7, bandwidth=0.7),
]

norm_scores = normalize_scores(raw_scores)
print("Normalized resource scores:", norm_scores)

# Simulate behavior history: client 1 behaves maliciously often
histories = [
    [1, 1, 1, 1],
    [0, 1, 0, 0],  # mostly dishonest
    [1, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 1, 0],
]

trust_scores = []
for r_norm, history in zip(norm_scores, histories):
    H = compute_historical_trust(history)
    curr_trust = compute_current_trust(r_norm, H)
    trust_scores.append(curr_trust)

print("Trust scores:", trust_scores)

threshold = compute_threshold(trust_scores)
print("Threshold:", threshold)

selected = select_clients(trust_scores, threshold)
print("Selected clients (indices):", selected)