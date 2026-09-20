"""
Selective re-optimization (Day 8): given a disruption, repair ONLY the
affected portion of a running schedule, freezing everything else exactly
as it was.

Pipeline for one event:
    1. Find directly-affected tasks (Day 7's get_affected_task_ids).
    2. Expand to include every DOWNSTREAM task too (a delayed/moved task
       can push its successors later, even if they weren't on the
       disrupted VM themselves).
    3. Freeze every other task's assignment unchanged.
    4. Re-run insertion-based scheduling (same mechanism as HEFT) but
       ONLY for the affected set, in priority order, never scheduling
       anything before `current_time`.
"""

from typing import Set, Tuple

from core.dynamic_events import (
    DynamicEvent,
    NewTaskArrivalEvent,
    VMFailureEvent,
    VMSlowdownEvent,
    compute_disruption_magnitude,
    get_affected_task_ids,
    is_significant_change,
)
from core.heft_engine import _earliest_finish_time_on_vm, compute_upward_ranks
from core.models import DAG, VM, Assignment, Schedule, Task


def get_descendant_closure(dag: DAG, task_ids: Set[int]) -> Set[int]:
    """All tasks in `task_ids`, PLUS every task reachable by following
    successor edges from any of them (their downstream dependents)."""
    result = set(task_ids)
    frontier = list(task_ids)
    while frontier:
        current = frontier.pop()
        for succ_id in dag.tasks[current].successors:
            if succ_id not in result:
                result.add(succ_id)
                frontier.append(succ_id)
    return result


def selective_reoptimize(
    dag: DAG,
    vms: list,
    schedule: Schedule,
    event: DynamicEvent,
    current_time: float,
) -> Tuple[Schedule, DAG, list, Set[int]]:
    """
    Repairs only the part of the schedule impacted by `event`.

    Returns (new_schedule, updated_dag, updated_vms, repaired_task_ids):
    the dag/vms are returned too because some events PERMANENTLY change
    them (a failed VM is gone for good; a slowdown persists; a new task
    permanently joins the DAG) -- the caller should keep using these
    going forward, not the originals.
    """
    directly_affected = get_affected_task_ids(dag, schedule, event, current_time)

    # --- Apply the event's permanent effect on the DAG / VM pool first ---
    if isinstance(event, NewTaskArrivalEvent):
        dag = DAG(
            tasks=dict(dag.tasks)
        )  # shallow copy so the original DAG object is untouched
        dag.tasks[event.task.id] = event.task
        vms_for_repair = list(vms)
    elif isinstance(event, VMFailureEvent):
        vms_for_repair = [vm for vm in vms if vm.id != event.vm_id]
    elif isinstance(event, VMSlowdownEvent):
        vms_for_repair = [
            VM(
                id=vm.id,
                speed=vm.speed * event.speed_multiplier,
                cost_per_hour=vm.cost_per_hour,
            )
            if vm.id == event.vm_id
            else vm
            for vm in vms
        ]
    else:
        vms_for_repair = list(vms)

    # --- Expand to include downstream dependents of the directly affected tasks ---
    affected = get_descendant_closure(dag, directly_affected)

    # --- Freeze every task NOT in the affected set ---
    new_schedule = Schedule()
    for task_id, assignment in schedule.assignments.items():
        if task_id not in affected:
            new_schedule.assignments[task_id] = Assignment(
                assignment.task_id,
                assignment.vm_id,
                assignment.start_time,
                assignment.end_time,
            )

    # --- Repair the affected tasks, in priority order, same insertion
    #     logic as HEFT, but never scheduled before current_time ---
    compute_upward_ranks(dag)
    priority_order = sorted(
        affected, key=lambda tid: dag.tasks[tid].upward_rank, reverse=True
    )

    for task_id in priority_order:
        task = dag.tasks[task_id]
        best_vm_id, best_start, best_end = None, None, float("inf")

        for vm in vms_for_repair:
            start, end = _earliest_finish_time_on_vm(
                task, vm, dag, new_schedule, min_start=current_time
            )
            if end < best_end:
                best_vm_id, best_start, best_end = vm.id, start, end

        new_schedule.assignments[task_id] = Assignment(
            task_id, best_vm_id, best_start, best_end
        )

    return new_schedule, dag, vms_for_repair, affected


def handle_event(
    dag: DAG,
    vms: list,
    schedule: Schedule,
    event: DynamicEvent,
    current_time: float,
    threshold: float,
) -> Tuple[Schedule, DAG, list, Set[int]]:
    """
    Top-level entry point tying Day 7 (detection) and Day 8 (repair)
    together: only actually repairs the schedule if the change detector
    says the disruption is significant enough. Otherwise returns the
    schedule completely unchanged -- the whole point of the change
    detector is to avoid wasting repair effort on disruptions too small
    to matter.

    A NewTaskArrivalEvent always triggers repair regardless of threshold,
    since the new task must be placed somewhere no matter how "small" the
    disruption -- there's no such thing as ignoring a task that needs to run.
    """
    if isinstance(event, NewTaskArrivalEvent):
        return selective_reoptimize(dag, vms, schedule, event, current_time)

    directly_affected = get_affected_task_ids(dag, schedule, event, current_time)
    magnitude = compute_disruption_magnitude(directly_affected, schedule, current_time)

    if not is_significant_change(magnitude, threshold):
        return schedule, dag, vms, set()  # ignore -- too small to bother with

    return selective_reoptimize(dag, vms, schedule, event, current_time)
