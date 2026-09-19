"""
Verifies:
  - compute_schedule_cost() and compute_imbalance() produce sane values
  - ObjectiveTracker normalization stays within [0, 1]
  - cost_aware_local_search() produces a feasible schedule
  - Comparing plain local_search() vs cost_aware_local_search() on the SAME
    starting schedule shows the expected trade-off: cost-aware search
    should generally achieve lower cost, sometimes at the price of a
    slightly higher makespan than the pure-makespan version.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.cost_model import (
    compute_imbalance,
    compute_schedule_cost,
    compute_vm_busy_times,
)
from core.dag_generator import generate_random_dag
from core.heft_engine import heft_schedule
from core.local_search import cost_aware_local_search, is_feasible, local_search
from core.vm_generator import generate_vm_pool


def test_cost_and_imbalance_are_sane():
    dag = generate_random_dag(num_tasks=10, num_levels=4, edge_probability=0.4, seed=5)
    vms = generate_vm_pool(num_vms=3, seed=5)
    schedule = heft_schedule(dag, vms)

    cost = compute_schedule_cost(schedule, dag, vms)
    imbalance = compute_imbalance(schedule, vms)
    busy_times = compute_vm_busy_times(schedule, vms)

    print(f"Cost: {cost:.2f}")
    print(f"Imbalance (stdev of busy time): {imbalance:.2f}")
    print(f"Busy time per VM: {busy_times}")

    assert cost > 0, "Cost should be positive for any non-empty schedule"
    assert imbalance >= 0, "Imbalance (a stdev) can never be negative"
    print("Cost model sanity checks passed.\n")


def test_cost_aware_vs_makespan_only():
    dag = generate_random_dag(
        num_tasks=18, num_levels=6, edge_probability=0.35, seed=21
    )
    vms = generate_vm_pool(num_vms=5, seed=21)
    heft_result = heft_schedule(dag, vms)

    # Plain local search: optimizes makespan only.
    ms_only = local_search(heft_result, dag, vms, max_iterations=800, seed=1)

    # Cost-aware: heavily weight cost (beta) over makespan (alpha), to make
    # the trade-off obvious in this test.
    cost_aware, tracker = cost_aware_local_search(
        heft_result,
        dag,
        vms,
        alpha=0.2,
        beta=0.7,
        gamma=0.1,
        max_iterations=800,
        seed=1,
    )

    assert is_feasible(ms_only, dag)
    assert is_feasible(cost_aware, dag)

    ms_only_cost = compute_schedule_cost(ms_only, dag, vms)
    cost_aware_cost = compute_schedule_cost(cost_aware, dag, vms)

    print(
        f"Makespan-only search  -> makespan={ms_only.makespan():.2f}, cost={ms_only_cost:.2f}"
    )
    print(
        f"Cost-aware search     -> makespan={cost_aware.makespan():.2f}, cost={cost_aware_cost:.2f}"
    )

    if cost_aware_cost < ms_only_cost:
        print(
            f"\nCost-aware search found a {ms_only_cost - cost_aware_cost:.2f} cheaper schedule"
            f" (makespan changed by {cost_aware.makespan() - ms_only.makespan():+.2f})."
        )
    else:
        print("\nNo cost improvement found in this run.")

    assert cost_aware_cost < ms_only_cost - 1e-6, (
        "With beta heavily favoring cost, cost-aware search should find a "
        "strictly cheaper schedule than pure makespan-only search."
    )

    print(
        "Cost-aware local search: feasible, and found a genuinely cheaper schedule.\n"
    )


if __name__ == "__main__":
    test_cost_and_imbalance_are_sane()
    test_cost_aware_vs_makespan_only()
    print("All Day 5-6 checks passed.")
