"""
Verifies:
  - VM failure: repaired tasks never end up back on the failed VM, frozen
    tasks are untouched, and the result is fully feasible.
  - Threshold gating: a tiny threshold triggers repair; a huge threshold
    leaves the schedule completely unchanged.
  - New task arrival: the new task gets placed, starts no earlier than
    current_time, and the DAG permanently gains the task.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.dag_generator import generate_random_dag
from core.dynamic_events import NewTaskArrivalEvent, VMFailureEvent, VMSlowdownEvent
from core.heft_engine import heft_schedule
from core.local_search import is_feasible
from core.models import Task
from core.selective_reopt import handle_event, selective_reoptimize
from core.vm_generator import generate_vm_pool


def test_vm_failure_repair():
    dag = generate_random_dag(num_tasks=16, num_levels=5, edge_probability=0.4, seed=9)
    vms = generate_vm_pool(num_vms=4, seed=9)
    schedule = heft_schedule(dag, vms)

    target_vm = vms[0].id
    current_time = schedule.makespan() * 0.4
    event = VMFailureEvent(vm_id=target_vm, time=current_time)

    repaired, new_dag, new_vms, affected = selective_reoptimize(
        dag, vms, schedule, event, current_time
    )

    print(f"VM {target_vm} failed at t={current_time:.1f}. Repaired tasks: {affected}")
    print(
        f"Old makespan: {schedule.makespan():.2f}, new makespan: {repaired.makespan():.2f}"
    )

    # 1. Feasibility must hold after repair.
    assert is_feasible(repaired, new_dag), "Repaired schedule must be feasible"

    # 2. No repaired task should be assigned to the failed VM.
    for task_id in affected:
        assert repaired.assignments[task_id].vm_id != target_vm, (
            f"Task {task_id} was reassigned but ended up on the FAILED VM"
        )

    # 3. The failed VM must be gone from the returned VM pool.
    assert target_vm not in [vm.id for vm in new_vms]

    # 4. Frozen tasks (not in `affected`) must be byte-identical to before.
    for task_id, original in schedule.assignments.items():
        if task_id not in affected:
            r = repaired.assignments[task_id]
            assert r.vm_id == original.vm_id
            assert r.start_time == original.start_time
            assert r.end_time == original.end_time

    print("VM failure repair: feasible, avoids failed VM, frozen tasks untouched.\n")


def test_threshold_gating():
    dag = generate_random_dag(num_tasks=14, num_levels=5, edge_probability=0.4, seed=17)
    vms = generate_vm_pool(num_vms=4, seed=17)
    schedule = heft_schedule(dag, vms)

    current_time = schedule.makespan() * 0.3

    # Pick whichever VM actually has unfinished work at current_time --
    # an arbitrary VM might happen to be idle by this point.
    vm_id_counts = {}
    for a in schedule.assignments.values():
        if a.end_time > current_time:
            vm_id_counts[a.vm_id] = vm_id_counts.get(a.vm_id, 0) + 1
    target_vm = max(vm_id_counts, key=vm_id_counts.get)

    event = VMSlowdownEvent(vm_id=target_vm, time=current_time, speed_multiplier=0.6)

    # Tiny threshold -> should trigger a real repair (schedule changes).
    repaired_low, _, _, affected_low = handle_event(
        dag, vms, schedule, event, current_time, threshold=0.01
    )
    # Huge threshold -> should be ignored entirely (schedule unchanged).
    repaired_high, _, _, affected_high = handle_event(
        dag, vms, schedule, event, current_time, threshold=1e9
    )

    print(
        f"Low threshold  -> repaired {len(affected_low)} tasks, "
        f"makespan {schedule.makespan():.2f} -> {repaired_low.makespan():.2f}"
    )
    print(
        f"High threshold -> repaired {len(affected_high)} tasks, "
        f"makespan {schedule.makespan():.2f} -> {repaired_high.makespan():.2f}"
    )

    assert len(affected_low) > 0, "Tiny threshold should have triggered a repair"
    assert len(affected_high) == 0, "Huge threshold should have ignored the disruption"
    assert repaired_high.makespan() == schedule.makespan(), (
        "Ignored disruption must leave the schedule completely unchanged"
    )

    print("Threshold gating works correctly in both directions.\n")


def test_new_task_arrival_repair():
    dag = generate_random_dag(num_tasks=10, num_levels=4, edge_probability=0.4, seed=23)
    vms = generate_vm_pool(num_vms=3, seed=23)
    schedule = heft_schedule(dag, vms)

    current_time = schedule.makespan() * 0.5
    new_task = Task(id=999, computation_cost=20.0)
    event = NewTaskArrivalEvent(time=current_time, task=new_task)

    repaired, new_dag, new_vms, affected = handle_event(
        dag,
        vms,
        schedule,
        event,
        current_time,
        threshold=1e9,  # threshold ignored for new tasks
    )

    print(
        f"New task arrives at t={current_time:.1f}. Placed: {repaired.assignments.get(999)}"
    )

    assert 999 in repaired.assignments, (
        "New task must always be placed, regardless of threshold"
    )
    assert repaired.assignments[999].start_time >= current_time - 1e-6, (
        "New task can't be scheduled before it actually arrives"
    )
    assert 999 in new_dag.tasks, "New task must permanently join the DAG"
    assert is_feasible(repaired, new_dag)

    print("New task arrival: placed correctly, respects arrival time, feasible.\n")


if __name__ == "__main__":
    test_vm_failure_repair()
    test_threshold_gating()
    test_new_task_arrival_repair()
    print("All Day 8 checks passed.")
