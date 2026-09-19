"""
Local search (Iterated Local Search) refinement on top of a HEFT schedule.

Three mutation operators:
    - move:    reassign one task to a different VM
    - swap:    two tasks (on different VMs) trade VMs
    - reorder: two tasks on the SAME VM swap their relative execution order

Each mutation is applied to a COPY of the schedule, then checked for
feasibility before being considered. Infeasible candidates are discarded.
"""

import random
from typing import Callable, List

from core.heft_engine import _earliest_finish_time_on_vm
from core.models import DAG, VM, Assignment, Schedule, Task

EPSILON = 1e-6  # tolerance for floating point start/end time comparisons


# ---------------------------------------------------------------------------
# Feasibility checking
# ---------------------------------------------------------------------------


def is_feasible(schedule: Schedule, dag: DAG) -> bool:
    """
    A schedule is feasible if:
      1. Every task starts no earlier than its predecessors' data-ready time
         (same-VM predecessor: after its end_time; different-VM predecessor:
         after end_time + comm_cost).
      2. No two tasks scheduled on the same VM have overlapping time ranges.
    """
    # --- 1. Precedence constraints ---
    for task_id, task in dag.tasks.items():
        assignment = schedule.assignments[task_id]
        for pred_id in task.predecessors:
            pred_assignment = schedule.assignments[pred_id]
            required_ready_time = pred_assignment.end_time
            if pred_assignment.vm_id != assignment.vm_id:
                required_ready_time += dag.tasks[pred_id].comm_cost[task_id]
            if assignment.start_time < required_ready_time - EPSILON:
                return False

    # --- 2. No overlapping tasks on the same VM ---
    vm_ids = {a.vm_id for a in schedule.assignments.values()}
    for vm_id in vm_ids:
        intervals = schedule.vm_busy_intervals(vm_id)
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0] + EPSILON:
                return False

    return True


# ---------------------------------------------------------------------------
# Mutation operators
# Each returns a NEW Schedule (the input schedule is never modified) or,
# if the mutation isn't applicable (e.g. only one VM exists), returns an
# unchanged copy.
# ---------------------------------------------------------------------------


def mutate_move(schedule: Schedule, dag: DAG, vms: List[VM]) -> Schedule:
    """Move a random task onto a different, randomly chosen VM."""
    new_schedule = schedule.copy()
    vm_by_id = {vm.id: vm for vm in vms}

    task_id = random.choice(list(dag.tasks.keys()))
    task = dag.tasks[task_id]
    current_vm_id = new_schedule.assignments[task_id].vm_id

    candidate_vms = [vm for vm in vms if vm.id != current_vm_id]
    if not candidate_vms:
        return new_schedule  # only one VM exists -> no-op

    new_vm = random.choice(candidate_vms)

    del new_schedule.assignments[task_id]
    start, end = _earliest_finish_time_on_vm(task, new_vm, dag, new_schedule)
    new_schedule.assignments[task_id] = Assignment(task_id, new_vm.id, start, end)
    return new_schedule


def mutate_swap(schedule: Schedule, dag: DAG, vms: List[VM]) -> Schedule:
    """Pick two tasks on DIFFERENT VMs and swap which VM each runs on.

    The two tasks must NOT be directly connected by a dependency edge --
    if they were, deleting both assignments before reinserting either would
    break the predecessor lookup used by insertion-based scheduling.
    """
    new_schedule = schedule.copy()
    vm_by_id = {vm.id: vm for vm in vms}

    task_ids = list(dag.tasks.keys())
    if len(task_ids) < 2:
        return new_schedule

    candidates = [
        (t1, t2)
        for t1 in task_ids
        for t2 in task_ids
        if t1 < t2
        and new_schedule.assignments[t1].vm_id != new_schedule.assignments[t2].vm_id
        and t2 not in dag.tasks[t1].successors
        and t1 not in dag.tasks[t2].successors
    ]
    if not candidates:
        return new_schedule  # no eligible pair -> no-op

    t1_id, t2_id = random.choice(candidates)
    a1, a2 = new_schedule.assignments[t1_id], new_schedule.assignments[t2_id]
    vm1, vm2 = vm_by_id[a1.vm_id], vm_by_id[a2.vm_id]
    task1, task2 = dag.tasks[t1_id], dag.tasks[t2_id]

    del new_schedule.assignments[t1_id]
    del new_schedule.assignments[t2_id]

    # t1 goes to vm2, t2 goes to vm1 -- recompute each via insertion logic,
    # since execution time changes with VM speed.
    s1, e1 = _earliest_finish_time_on_vm(task1, vm2, dag, new_schedule)
    new_schedule.assignments[t1_id] = Assignment(t1_id, vm2.id, s1, e1)

    s2, e2 = _earliest_finish_time_on_vm(task2, vm1, dag, new_schedule)
    new_schedule.assignments[t2_id] = Assignment(t2_id, vm1.id, s2, e2)

    return new_schedule


def mutate_reorder(schedule: Schedule, dag: DAG, vms: List[VM]) -> Schedule:
    """Pick a VM with >=2 tasks on it and re-insert two of them in swapped
    order, potentially changing which one runs first.

    The two tasks must NOT be directly connected by a dependency edge, for
    the same reason as in mutate_swap.
    """
    new_schedule = schedule.copy()
    vm_by_id = {vm.id: vm for vm in vms}

    # Group currently-assigned task ids by VM
    tasks_per_vm = {}
    for task_id, a in new_schedule.assignments.items():
        tasks_per_vm.setdefault(a.vm_id, []).append(task_id)

    eligible_pairs = []
    for vm_id, tids in tasks_per_vm.items():
        for t1 in tids:
            for t2 in tids:
                if (
                    t1 < t2
                    and t2 not in dag.tasks[t1].successors
                    and t1 not in dag.tasks[t2].successors
                ):
                    eligible_pairs.append((vm_id, t1, t2))

    if not eligible_pairs:
        return new_schedule  # nothing eligible -> no-op

    vm_id, t1_id, t2_id = random.choice(eligible_pairs)
    vm = vm_by_id[vm_id]
    task1, task2 = dag.tasks[t1_id], dag.tasks[t2_id]

    del new_schedule.assignments[t1_id]
    del new_schedule.assignments[t2_id]

    # Re-insert in the OPPOSITE order from before, to actually try a
    # different arrangement rather than reproducing the same one.
    s2, e2 = _earliest_finish_time_on_vm(task2, vm, dag, new_schedule)
    new_schedule.assignments[t2_id] = Assignment(t2_id, vm.id, s2, e2)

    s1, e1 = _earliest_finish_time_on_vm(task1, vm, dag, new_schedule)
    new_schedule.assignments[t1_id] = Assignment(t1_id, vm.id, s1, e1)

    return new_schedule


MUTATION_OPERATORS: List[Callable] = [mutate_move, mutate_swap, mutate_reorder]


# ---------------------------------------------------------------------------
# Local search loop
# ---------------------------------------------------------------------------


def local_search(
    initial_schedule: Schedule,
    dag: DAG,
    vms: List[VM],
    max_iterations: int = 500,
    seed: int = None,
) -> Schedule:
    """
    Greedy local search: repeatedly try a random mutation, keep it only if
    it's feasible AND improves makespan. Never accepts a worse schedule.
    """
    if seed is not None:
        random.seed(seed)

    current = initial_schedule
    for _ in range(max_iterations):
        operator = random.choice(MUTATION_OPERATORS)
        candidate = operator(current, dag, vms)

        if not is_feasible(candidate, dag):
            continue
        if candidate.makespan() < current.makespan() - EPSILON:
            current = candidate

    return current


def iterated_local_search(
    initial_schedule: Schedule,
    dag: DAG,
    vms: List[VM],
    max_iterations: int = 1000,
    iterations_per_round: int = 100,
    perturbation_strength: int = 5,
    seed: int = None,
) -> Schedule:
    """
    Runs local_search() in rounds. After each round, applies a burst of
    `perturbation_strength` random mutations WITHOUT the improve-or-reject
    check (accepting them even if feasible-but-worse), to shake the
    schedule out of a local optimum before the next round of local search.

    Keeps track of the best feasible schedule seen across all rounds,
    since a perturbation round can temporarily make things worse.
    """
    if seed is not None:
        random.seed(seed)

    current = initial_schedule
    best = initial_schedule
    rounds = max(1, max_iterations // iterations_per_round)

    for round_num in range(rounds):
        current = local_search(current, dag, vms, iterations_per_round)
        if current.makespan() < best.makespan() - EPSILON:
            best = current

        # Perturb: apply a few random mutations, keep only if still feasible
        # (accept even if worse -- that's the point of a "kick").
        for _ in range(perturbation_strength):
            operator = random.choice(MUTATION_OPERATORS)
            candidate = operator(current, dag, vms)
            if is_feasible(candidate, dag):
                current = candidate

    return best


def cost_aware_local_search(
    initial_schedule: Schedule,
    dag: DAG,
    vms: List[VM],
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    max_iterations: int = 500,
    seed: int = None,
):
    """
    Same greedy local search loop as local_search(), but accepts/rejects
    mutations based on the multi-objective Score (makespan + cost +
    imbalance) instead of raw makespan alone. This is what makes the
    search "cost-aware" -- it can now accept a mutation that makes
    makespan slightly WORSE, if the cost/imbalance improvement outweighs
    it in the combined score.

    Returns (best_schedule, tracker) -- the tracker is returned too since
    it holds the running min/max ranges, useful for reporting/plots later.
    """
    # Local import to avoid a circular import at module load time
    # (cost_model doesn't need to import local_search, so this keeps the
    # dependency one-directional).
    from core.cost_model import ObjectiveTracker, compute_raw_objectives

    if seed is not None:
        random.seed(seed)

    tracker = ObjectiveTracker()
    current = initial_schedule

    for _ in range(max_iterations):
        operator = random.choice(MUTATION_OPERATORS)
        candidate = operator(current, dag, vms)

        if not is_feasible(candidate, dag):
            continue

        # Get RAW (un-normalized) objective values for both schedules first,
        # then feed BOTH into the tracker before normalizing either one.
        # This guarantees current and candidate are compared under the
        # exact same min/max range -- computing one score, then the other,
        # against a tracker that changed in between would silently compare
        # them on different scales.
        current_makespan, current_cost, current_imbalance = compute_raw_objectives(
            current, dag, vms
        )
        cand_makespan, cand_cost, cand_imbalance = compute_raw_objectives(
            candidate, dag, vms
        )

        tracker.observe(current_makespan, current_cost, current_imbalance)
        tracker.observe(cand_makespan, cand_cost, cand_imbalance)

        current_score = (
            alpha * tracker.normalize_makespan(current_makespan)
            + beta * tracker.normalize_cost(current_cost)
            + gamma * tracker.normalize_imbalance(current_imbalance)
        )
        candidate_score = (
            alpha * tracker.normalize_makespan(cand_makespan)
            + beta * tracker.normalize_cost(cand_cost)
            + gamma * tracker.normalize_imbalance(cand_imbalance)
        )

        if candidate_score < current_score - EPSILON:
            current = candidate

    return current, tracker
