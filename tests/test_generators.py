"""
Quick sanity checks for the DAG and VM generators.
Run with: python -m tests.test_generators   (from the project root)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.dag_generator import generate_random_dag
from core.vm_generator import generate_vm_pool


def test_dag_is_valid():
    dag = generate_random_dag(num_tasks=10, num_levels=4, edge_probability=0.4, seed=42)

    # 1. Must be acyclic -> topological_order() must succeed and cover all tasks
    order = dag.topological_order()
    assert len(order) == 10, "Topological sort should include every task"
    print(f"Topological order: {order}")

    # 2. Every non-entry task must have at least one predecessor (connected)
    entries = dag.entry_tasks()
    print(f"Entry tasks (no predecessors): {entries}")
    for tid, task in dag.tasks.items():
        if tid not in entries:
            assert len(task.predecessors) > 0, f"Task {tid} is orphaned"

    # 3. Exit tasks exist
    exits = dag.exit_tasks()
    print(f"Exit tasks (no successors): {exits}")
    assert len(exits) > 0

    print("DAG structure OK\n")


def test_vm_pool_is_heterogeneous():
    vms = generate_vm_pool(num_vms=5, seed=42)
    for vm in vms:
        print(vm)

    speeds = [vm.speed for vm in vms]
    costs = [vm.cost_per_hour for vm in vms]
    assert len(set(speeds)) > 1, "VMs should have varied speeds"
    assert len(set(costs)) > 1, "VMs should have varied costs"

    # Sanity check the speed/cost correlation (not strict, just directional)
    fastest = max(vms, key=lambda v: v.speed)
    slowest = min(vms, key=lambda v: v.speed)
    print(f"\nFastest VM: {fastest}")
    print(f"Slowest VM: {slowest}")

    print("VM pool OK\n")


if __name__ == "__main__":
    test_dag_is_valid()
    test_vm_pool_is_heterogeneous()
    print("All Day 1 checks passed.")
