"""
Cost model and multi-objective scoring for DCA-HEFT-ILS.

Adds a second (and third) objective alongside makespan:
    - cost:      total $ to run the schedule, based on each task's VM
    - imbalance: how unevenly work is spread across the VM pool

Score(S) = alpha * NormMakespan(S) + beta * NormCost(S) + gamma * Imbalance(S)

Lower Score = better. This REPLACES the plain "smaller makespan" acceptance
rule used in local_search.py -- see cost_aware_local_search() below.
"""

import statistics
from typing import Dict, List

from core.models import DAG, VM, Schedule

# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def compute_task_cost(
    task_id: int, schedule: Schedule, dag: DAG, vm_by_id: Dict[int, VM]
) -> float:
    """Cost of running one task = its execution duration on its assigned VM
    times that VM's cost_per_hour. (Time units are assumed consistent with
    cost_per_hour's "per hour" -- e.g. both in hours, or both scaled the
    same way; for a course project it's fine to treat this as abstract
    "cost units" as long as it's used consistently.)
    """
    assignment = schedule.assignments[task_id]
    vm = vm_by_id[assignment.vm_id]
    duration = assignment.end_time - assignment.start_time
    return duration * vm.cost_per_hour


def compute_schedule_cost(schedule: Schedule, dag: DAG, vms: List[VM]) -> float:
    """Total cost across every task in the schedule."""
    vm_by_id = {vm.id: vm for vm in vms}
    return sum(
        compute_task_cost(task_id, schedule, dag, vm_by_id)
        for task_id in schedule.assignments
    )


# ---------------------------------------------------------------------------
# Utilization / imbalance
# ---------------------------------------------------------------------------


def compute_vm_busy_times(schedule: Schedule, vms: List[VM]) -> Dict[int, float]:
    """Total busy time (sum of task durations) per VM. VMs with zero tasks
    assigned still appear in the result with busy_time = 0 -- an idle VM
    IS part of what makes a schedule imbalanced."""
    busy = {vm.id: 0.0 for vm in vms}
    for assignment in schedule.assignments.values():
        busy[assignment.vm_id] += assignment.end_time - assignment.start_time
    return busy


def compute_imbalance(schedule: Schedule, vms: List[VM]) -> float:
    """Standard deviation of busy time across VMs. A schedule that piles
    everything onto one VM while others sit idle has HIGH imbalance, even
    if its makespan happens to be good."""
    busy_times = list(compute_vm_busy_times(schedule, vms).values())
    if len(busy_times) < 2:
        return 0.0
    return statistics.pstdev(busy_times)


def compute_average_utilization(schedule: Schedule, vms: List[VM]) -> float:
    """Average, across all VMs, of (busy_time / makespan) -- i.e. what
    fraction of the schedule's total wall-clock time each VM actually
    spends doing work, expressed as a percentage.

    Classical HEFT always picks the fastest available machine for a task,
    which tends to pile work onto a few fast VMs while others sit mostly
    idle: good makespan, poor utilization of the pool you're paying for.
    This is a scale-free way to show that a cost-/balance-aware schedule
    uses the whole VM pool more effectively, not just that it costs less.
    """
    makespan = schedule.makespan()
    if makespan <= 0 or not vms:
        return 0.0
    busy_times = compute_vm_busy_times(schedule, vms)
    utilizations = [busy / makespan for busy in busy_times.values()]
    return statistics.mean(utilizations) * 100.0


def compute_load_balance_index(schedule: Schedule, vms: List[VM]) -> float:
    """Coefficient of variation (stdev / mean) of per-VM busy time.

    Unlike compute_imbalance() -- a raw stdev in time units, which isn't
    comparable across DAGs/VM pools of different scale -- this is
    dimensionless: 0.0 means perfectly balanced load across every VM, and
    it grows the more lopsided the distribution is, regardless of how big
    the workload happens to be. That makes it usable for averaging across
    many random trials of different sizes, which raw imbalance is not.
    """
    busy_times = list(compute_vm_busy_times(schedule, vms).values())
    if len(busy_times) < 2:
        return 0.0
    mean_busy = statistics.mean(busy_times)
    if mean_busy <= 0:
        return 0.0
    return statistics.pstdev(busy_times) / mean_busy


# ---------------------------------------------------------------------------
# Normalization + combined score
# ---------------------------------------------------------------------------


class ObjectiveTracker:
    """
    Tracks the running min/max of each objective seen so far during a
    search, so raw values (makespan in time units, cost in $, imbalance in
    time units) can be rescaled to a comparable 0-1 range before combining.

    Without this, objectives with very different natural scales (e.g.
    makespan ~100, cost ~$20) would contribute wildly unequal amounts to
    the combined score regardless of the chosen weights.
    """

    def __init__(self):
        self.min_makespan = float("inf")
        self.max_makespan = float("-inf")
        self.min_cost = float("inf")
        self.max_cost = float("-inf")
        self.min_imbalance = float("inf")
        self.max_imbalance = float("-inf")

    def observe(self, makespan: float, cost: float, imbalance: float) -> None:
        self.min_makespan = min(self.min_makespan, makespan)
        self.max_makespan = max(self.max_makespan, makespan)
        self.min_cost = min(self.min_cost, cost)
        self.max_cost = max(self.max_cost, cost)
        self.min_imbalance = min(self.min_imbalance, imbalance)
        self.max_imbalance = max(self.max_imbalance, imbalance)

    @staticmethod
    def _normalize(value: float, lo: float, hi: float) -> float:
        if hi - lo < 1e-9:
            return 0.0  # every candidate seen so far is identical on this objective
        return (value - lo) / (hi - lo)

    def normalize_makespan(self, value: float) -> float:
        return self._normalize(value, self.min_makespan, self.max_makespan)

    def normalize_cost(self, value: float) -> float:
        return self._normalize(value, self.min_cost, self.max_cost)

    def normalize_imbalance(self, value: float) -> float:
        return self._normalize(value, self.min_imbalance, self.max_imbalance)


def compute_raw_objectives(schedule: Schedule, dag: DAG, vms: List[VM]):
    """Returns (makespan, cost, imbalance) without touching any tracker --
    used when two schedules need to be normalized under the exact same
    range for a fair comparison (see cost_aware_local_search)."""
    makespan = schedule.makespan()
    cost = compute_schedule_cost(schedule, dag, vms)
    imbalance = compute_imbalance(schedule, vms)
    return makespan, cost, imbalance


def compute_score(
    schedule: Schedule,
    dag: DAG,
    vms: List[VM],
    tracker: ObjectiveTracker,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> float:
    """
    Score(S) = alpha * NormMakespan(S) + beta * NormCost(S) + gamma * NormImbalance(S)

    Lower is better. IMPORTANT: call tracker.observe(...) with this
    schedule's raw values BEFORE computing its normalized score, so the
    tracker's min/max range always includes the value being normalized.
    """
    makespan = schedule.makespan()
    cost = compute_schedule_cost(schedule, dag, vms)
    imbalance = compute_imbalance(schedule, vms)

    tracker.observe(makespan, cost, imbalance)

    return (
        alpha * tracker.normalize_makespan(makespan)
        + beta * tracker.normalize_cost(cost)
        + gamma * tracker.normalize_imbalance(imbalance)
    )
