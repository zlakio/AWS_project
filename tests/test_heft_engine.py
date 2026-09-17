"""
Verifies compute_upward_ranks() and heft_schedule() against the exact
hand-worked example from the theory walkthrough:

    T1 --6--> T2 --7--> T4
    T1 --5--> T3 --4--> T4

    costs: T1=9, T2=8, T3=6, T4=5

Expected ranks (computed by hand): T1=35, T2=20, T3=15, T4=5
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.heft_engine import compute_upward_ranks, heft_schedule
from core.models import DAG, VM, Task


def build_hand_worked_dag() -> DAG:
    t1 = Task(id=1, computation_cost=9)
    t2 = Task(id=2, computation_cost=8)
    t3 = Task(id=3, computation_cost=6)
    t4 = Task(id=4, computation_cost=5)

    t1.successors = [2, 3]
    t1.comm_cost = {2: 6, 3: 5}
    t2.predecessors = [1]
    t2.successors = [4]
    t2.comm_cost = {4: 7}
    t3.predecessors = [1]
    t3.successors = [4]
    t3.comm_cost = {4: 4}
    t4.predecessors = [2, 3]

    return DAG(tasks={1: t1, 2: t2, 3: t3, 4: t4})


def test_ranks_match_hand_calculation():
    dag = build_hand_worked_dag()
    compute_upward_ranks(dag)

    expected = {1: 35, 2: 20, 3: 15, 4: 5}
    for task_id, expected_rank in expected.items():
        actual = dag.tasks[task_id].upward_rank
        print(f"T{task_id}: expected={expected_rank}, got={actual}")
        assert actual == expected_rank, f"T{task_id} rank mismatch!"

    print("Ranks match hand calculation exactly.\n")


def test_full_schedule_runs():
    dag = build_hand_worked_dag()
    vms = [
        VM(id=0, speed=1.0, cost_per_hour=1.0),
        VM(id=1, speed=2.0, cost_per_hour=2.0),
    ]

    schedule = heft_schedule(dag, vms)

    # Priority order should be T1, T2, T3, T4 (matches rank order 35>20>15>5)
    for task_id in [1, 2, 3, 4]:
        a = schedule.assignments[task_id]
        print(
            f"T{task_id} -> VM{a.vm_id}, start={a.start_time:.2f}, end={a.end_time:.2f}"
        )

    print(f"\nMakespan: {schedule.makespan():.2f}")

    # Sanity: every task's start time must respect its predecessors
    for task_id, task in dag.tasks.items():
        a = schedule.assignments[task_id]
        for pred_id in task.predecessors:
            pred_a = schedule.assignments[pred_id]
            assert a.start_time >= pred_a.end_time, (
                f"T{task_id} starts before predecessor T{pred_id} finishes!"
            )

    print("All precedence constraints respected.\n")


if __name__ == "__main__":
    test_ranks_match_hand_calculation()
    test_full_schedule_runs()
    print("All Day 2 checks passed.")
