import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np

from src.trust import (
    compute_resource_score, normalize_scores, compute_historical_trust,
    compute_current_trust, update_trust, compute_threshold, select_clients
)


class TrustFedAvg(fl.server.strategy.FedAvg):
    """Custom Flower strategy implementing trust-based client selection and aggregation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.behavior_history = {}   # client_id -> list of 0/1
        self.prev_trust = {}         # client_id -> last trust score

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # --- Extract per-client info returned from fit() ---
        client_ids = []
        param_list = []
        num_examples_list = []
        raw_scores = []

        for _, fit_res in results:
            metrics = fit_res.metrics
            cid = metrics["client_id"]
            client_ids.append(cid)
            param_list.append(parameters_to_ndarrays(fit_res.parameters))
            num_examples_list.append(fit_res.num_examples)

            raw_score = compute_resource_score(
                cpu=metrics["cpu"],
                ram=metrics["ram"],
                battery=metrics["battery"],
                internet_speed=metrics["internet_speed"],
                bandwidth=metrics["bandwidth"],
            )
            raw_scores.append(raw_score)

            # Update this client's behavior history
            honest = metrics["honest_round"]
            self.behavior_history.setdefault(cid, []).append(honest)

        # --- Normalize resource scores across participating clients ---
        norm_scores = normalize_scores(raw_scores)

        # --- Compute trust scores (current + smoothed with previous) ---
        trust_scores = []
        for cid, r_norm in zip(client_ids, norm_scores):
            H = compute_historical_trust(self.behavior_history[cid])
            curr_trust = compute_current_trust(r_norm, H)
            prev = self.prev_trust.get(cid, 0.0)
            smoothed = update_trust(prev, curr_trust)
            self.prev_trust[cid] = smoothed
            trust_scores.append(smoothed)

        print(f"\n[Round {server_round}] Trust scores: "
              f"{dict(zip(client_ids, [round(t, 3) for t in trust_scores]))}")

        # --- Threshold-based client selection ---
        threshold = compute_threshold(trust_scores)
        selected_local_indices = select_clients(trust_scores, threshold)

        print(f"[Round {server_round}] Threshold: {round(threshold, 3)} | "
              f"Selected clients: {[client_ids[i] for i in selected_local_indices]}")

        if not selected_local_indices:
            # Safety fallback: if none pass threshold, use all clients (avoid crash)
            selected_local_indices = list(range(len(client_ids)))

        # --- Trust-weighted aggregation ---
        selected_trusts = [trust_scores[i] for i in selected_local_indices]
        total_trust = sum(selected_trusts) + 1e-8

        num_layers = len(param_list[0])
        aggregated = [np.zeros_like(layer) for layer in param_list[0]]

        for idx in selected_local_indices:
            weight = trust_scores[idx] / total_trust
            for layer_idx in range(num_layers):
                aggregated[layer_idx] += weight * param_list[idx][layer_idx]

        aggregated_parameters = ndarrays_to_parameters(aggregated)
        
        agg_sum = sum(layer.sum() for layer in aggregated)
        print(f"[SERVER Round {server_round}] Aggregated params - sum: {agg_sum}")

        metrics_aggregated = {
            "threshold": threshold,
            "num_selected": len(selected_local_indices),
        }

        return aggregated_parameters, metrics_aggregated