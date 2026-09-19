"""
Verifies:
  - get_affected_task_ids correctly identifies only not-yet-finished tasks
    on the affected VM (already-finished tasks are untouched)
  - compute_disruption_magnitude sums remaining work sensibly
  - is_significant_change respects the threshold in both directions
  - a NewTaskArrivalEvent is handled distinctly (affects only itself)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.dag_generator import generate_random_dag
from core.dynamic_events import (
    NewTaskArrivalEvent,
    VMFailureEvent,
    VMSlowdownEvent,
    compute_disruption_magnitude,
    get_affected_task_ids,
    is_significant_change,
)
from core.heft_engine import heft_schedule
from core.models import Task
from core.vm_generator import generate_vm_pool


def test_slowdown_affects_only_unfinished_tasks_on_that_vm():
    dag = generate_random_dag(num_tasks=15, num_levels=5, edge_probability=0.4, seed=8)
    vms = generate_vm_pool(num_vms=4, seed=8)
    schedule = heft_schedule(dag, vms)

    # Pick a VM that actually has multiple tasks, and a midpoint time.
    vm_id_counts = {}
    for a in schedule.assignments.values():
        vm_id_counts[a.vm_id] = vm_id_counts.get(a.vm_id, 0) + 1
    target_vm = max(vm_id_counts, key=vm_id_counts.get)

    current_time = schedule.makespan() / 2
    event = VMSlowdownEvent(vm_id=target_vm, time=current_time, speed_multiplier=0.5)

    affected = get_affected_task_ids(dag, schedule, event, current_time)
    print(f"VM {target_vm} slowdown at t={current_time:.1f} affects tasks: {affected}")

    # Manually verify: every affected task must be on target_vm and unfinished;
    # every task on target_vm that's ALREADY finished must NOT be in the set.
    for task_id, a in schedule.assignments.items():
        if a.vm_id == target_vm and a.end_time > current_time:
            assert task_id in affected, (
                f"Task {task_id} should be affected but wasn't flagged"
            )
        if a.vm_id == target_vm and a.end_time <= current_time:
            assert task_id not in affected, (
                f"Already-finished task {task_id} should NOT be affected"
            )

    print("Affected-task detection correct for slowdown.\n")
    return dag, schedule, affected, current_time


def test_disruption_magnitude_and_significance():
    dag, schedule, affected, current_time = (
        test_slowdown_affects_only_unfinished_tasks_on_that_vm()
    )

    magnitude = compute_disruption_magnitude(affected, schedule, current_time)
    print(f"Disruption magnitude: {magnitude:.2f}")
    assert magnitude >= 0

    # A very low threshold should flag this as significant;
    # an absurdly high threshold should not.
    assert is_significant_change(magnitude, threshold=0.01) is True
    assert is_significant_change(magnitude, threshold=1e9) is False
    print("Significance threshold logic behaves correctly in both directions.\n")


def test_new_task_arrival_affects_only_itself():
    dag = generate_random_dag(num_tasks=10, num_levels=4, edge_probability=0.4, seed=3)
    vms = generate_vm_pool(num_vms=3, seed=3)
    schedule = heft_schedule(dag, vms)

    new_task = Task(id=999, computation_cost=25.0)
    event = NewTaskArrivalEvent(time=10.0, task=new_task)

    affected = get_affected_task_ids(dag, schedule, event, current_time=10.0)
    print(f"New task arrival affects: {affected}")
    assert affected == {999}, "A new task arrival should only ever 'affect' itself"

    print("New task arrival handled correctly.\n")


def test_vm_failure_uses_same_logic_as_slowdown():
    dag = generate_random_dag(num_tasks=12, num_levels=4, edge_probability=0.4, seed=11)
    vms = generate_vm_pool(num_vms=3, seed=11)
    schedule = heft_schedule(dag, vms)

    target_vm = vms[0].id
    current_time = schedule.makespan() * 0.3
    event = VMFailureEvent(vm_id=target_vm, time=current_time)

    affected = get_affected_task_ids(dag, schedule, event, current_time)
    print(f"VM {target_vm} failure at t={current_time:.1f} affects tasks: {affected}")

    for task_id, a in schedule.assignments.items():
        if a.vm_id == target_vm and a.end_time > current_time:
            assert task_id in affected

    print("VM failure detection correct.\n")


if __name__ == "__main__":
    test_disruption_magnitude_and_significance()
    test_new_task_arrival_affects_only_itself()
    test_vm_failure_uses_same_logic_as_slowdown()
    print("All Day 7 checks passed.")
