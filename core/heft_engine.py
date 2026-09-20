"""
Classical HEFT (Heterogeneous Earliest Finish Time) scheduling engine.

Two phases, matching the theory:
    Phase 1: compute_upward_ranks()  -> priority for every task
    Phase 2: heft_schedule()         -> assign each task to a VM using
                                         insertion-based earliest-finish-time
"""

from typing import Dict, List, Tuple

from core.models import DAG, VM, Assignment, Schedule, Task


def compute_upward_ranks(dag: DAG) -> None:
    """
    Fills in task.upward_rank for every task in the DAG.

    rank(t) = computation_cost(t) + max over successors s of:
                  [ comm_cost(t, s) + rank(s) ]
    (0 for the max-term if t has no successors.)

    Must be computed bottom-up: a task's rank depends on its successors'
    ranks, so we process tasks in REVERSE topological order (exit tasks
    first, entry tasks last).
    """
    # topological_order() gives entry -> exit order; reverse it to get
    # exit -> entry, which is exactly the "bottom-up" order rank needs.
    for task_id in reversed(dag.topological_order()):
        task = dag.tasks[task_id]

        if not task.successors:
            task.upward_rank = task.computation_cost
            continue

        best_downstream = max(
            task.comm_cost[succ_id] + dag.tasks[succ_id].upward_rank
            for succ_id in task.successors
        )
        task.upward_rank = task.computation_cost + best_downstream


def _data_ready_time(
    task: Task,
    vm: VM,
    dag: DAG,
    schedule: Schedule,
) -> float:
    """
    The earliest time `task` could possibly START on `vm`, based purely on
    its predecessors — ignoring vm's own busy schedule for a moment.

    For each predecessor:
      - if the predecessor ran on the SAME vm, its data is already local
        -> no communication delay, just wait for it to finish.
      - if the predecessor ran on a DIFFERENT vm, we must wait for its
        finish time PLUS the communication cost to transfer data over.

    Returns 0.0 if the task has no predecessors (it's an entry task and can
    start immediately, subject only to VM availability).
    """
    if not task.predecessors:
        return 0.0

    ready_times = []
    for pred_id in task.predecessors:
        pred_assignment = schedule.assignments[pred_id]
        if pred_assignment.vm_id == vm.id:
            ready_times.append(pred_assignment.end_time)
        else:
            pred_task = dag.tasks[pred_id]
            comm_delay = pred_task.comm_cost[task.id]
            ready_times.append(pred_assignment.end_time + comm_delay)

    return max(ready_times)


def _earliest_finish_time_on_vm(
    task: Task,
    vm: VM,
    dag: DAG,
    schedule: Schedule,
    min_start: float = 0.0,
) -> Tuple[float, float]:
    """
    Finds the earliest (start_time, end_time) for `task` on `vm`, using
    INSERTION-based scheduling: check every gap between vm's already-
    scheduled tasks (plus the gap before the first one, and after the
    last one) and use the first gap the task actually fits into.

    `min_start` is a floor on when the task may start -- 0.0 (the default)
    changes nothing for normal HEFT scheduling. It matters for DYNAMIC
    REPAIR (Day 8): when re-scheduling a task after a disruption at
    simulation time T, the task obviously can't be backdated to before T,
    even if an earlier gap would otherwise fit it.

    Returns (start_time, end_time).
    """
    ready_time = max(_data_ready_time(task, vm, dag, schedule), min_start)
    duration = vm.exec_time(task)
    intervals = schedule.vm_busy_intervals(vm.id)

    if not intervals:
        # VM is completely empty -> task starts as soon as its data is ready.
        start = ready_time
        return start, start + duration

    # Check the gap BEFORE the first scheduled task.
    first_start = intervals[0][0]
    if ready_time + duration <= first_start:
        return ready_time, ready_time + duration

    # Check gaps BETWEEN consecutive scheduled tasks.
    for i in range(len(intervals) - 1):
        gap_start = max(ready_time, intervals[i][1])  # can't start before data is ready
        gap_end = intervals[i + 1][0]
        if gap_start + duration <= gap_end:
            return gap_start, gap_start + duration

    # No gap fits -> schedule after the LAST task on this VM.
    last_end = intervals[-1][1]
    start = max(ready_time, last_end)
    return start, start + duration


def heft_schedule(dag: DAG, vms: List[VM]) -> Schedule:
    """
    Full classical HEFT algorithm:
      1. Compute upward ranks for every task (Phase 1).
      2. Process tasks in descending rank order, assigning each to
         whichever VM gives the smallest earliest-finish-time (Phase 2).
    """
    compute_upward_ranks(dag)

    # Highest rank first = highest priority first.
    priority_order = sorted(
        dag.tasks.values(), key=lambda t: t.upward_rank, reverse=True
    )

    schedule = Schedule()

    for task in priority_order:
        best_vm_id = None
        best_start, best_end = None, float("inf")

        for vm in vms:
            start, end = _earliest_finish_time_on_vm(task, vm, dag, schedule)
            if end < best_end:
                best_vm_id = vm.id
                best_start, best_end = start, end

        schedule.assignments[task.id] = Assignment(
            task_id=task.id,
            vm_id=best_vm_id,
            start_time=best_start,
            end_time=best_end,
        )

    return schedule
