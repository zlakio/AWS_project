"""
Verifies local_search() and iterated_local_search():
  - the result is always a feasible schedule
  - makespan never gets worse than the starting HEFT schedule
  - (typically) makespan improves on a reasonably sized random DAG
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.dag_generator import generate_random_dag
from core.heft_engine import heft_schedule
from core.local_search import is_feasible, iterated_local_search, local_search
from core.vm_generator import generate_vm_pool


def test_local_search_improves_or_matches_heft():
    dag = generate_random_dag(num_tasks=15, num_levels=5, edge_probability=0.4, seed=7)
    vms = generate_vm_pool(num_vms=4, seed=7)

    heft_result = heft_schedule(dag, vms)
    print(f"HEFT makespan: {heft_result.makespan():.2f}")
    assert is_feasible(heft_result, dag), "HEFT's own output should always be feasible"

    refined = local_search(heft_result, dag, vms, max_iterations=500, seed=42)
    print(f"After local search: {refined.makespan():.2f}")
    assert is_feasible(refined, dag), (
        "Local search must never return an infeasible schedule"
    )
    assert refined.makespan() <= heft_result.makespan() + 1e-6, (
        "Local search must never make the schedule worse (greedy accept-only-if-better)"
    )

    print("Local search: feasible and no worse than HEFT.\n")


def test_iterated_local_search():
    dag = generate_random_dag(
        num_tasks=20, num_levels=6, edge_probability=0.35, seed=13
    )
    vms = generate_vm_pool(num_vms=5, seed=13)

    heft_result = heft_schedule(dag, vms)
    print(f"HEFT makespan: {heft_result.makespan():.2f}")

    ils_result = iterated_local_search(
        heft_result,
        dag,
        vms,
        max_iterations=1000,
        iterations_per_round=100,
        perturbation_strength=5,
        seed=99,
    )
    print(f"After iterated local search: {ils_result.makespan():.2f}")
    assert is_feasible(ils_result, dag)
    assert ils_result.makespan() <= heft_result.makespan() + 1e-6, (
        "ILS tracks the BEST schedule seen, so the final result must not be worse than HEFT"
    )

    improvement = heft_result.makespan() - ils_result.makespan()
    pct = (improvement / heft_result.makespan()) * 100
    print(f"Improvement: {improvement:.2f} ({pct:.1f}%)\n")


if __name__ == "__main__":
    test_local_search_improves_or_matches_heft()
    test_iterated_local_search()
    print("All Day 3-4 checks passed.")
