"""
Core data structures for the DCA-HEFT-ILS scheduler.

These are the building blocks every other module (HEFT engine, local search,
cost model, dynamic simulator) will import and use.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class Task:
    """A single task/node in the workflow DAG.

    computation_cost is expressed in an abstract unit (e.g. "million
    instructions"). Actual execution time depends on which VM it runs on:
        exec_time = computation_cost / vm.speed
    """
    id: int
    computation_cost: float
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    # comm_cost[successor_id] = data transfer cost if this task and that
    # successor run on DIFFERENT VMs. If they run on the SAME VM, communication
    # cost is treated as 0 (a standard HEFT assumption).
    comm_cost: Dict[int, float] = field(default_factory=dict)

    # Filled in later by the HEFT engine — not part of the task's identity,
    # just a convenient place to cache the computed priority.
    upward_rank: Optional[float] = None

    def __repr__(self):
        return f"Task({self.id}, cost={self.computation_cost:.1f})"


@dataclass
class VM:
    """A single virtual machine in the heterogeneous VM pool."""
    id: int
    speed: float          # processing capacity, e.g. MIPS. Higher = faster.
    cost_per_hour: float  # cloud billing rate

    def exec_time(self, task: Task) -> float:
        """Time (in the same time unit as computation_cost/speed) for this
        VM to execute the given task."""
        return task.computation_cost / self.speed

    def __repr__(self):
        return f"VM({self.id}, speed={self.speed:.1f}, cost=${self.cost_per_hour:.2f}/hr)"


@dataclass
class DAG:
    """A workflow: a collection of tasks with dependency edges."""
    tasks: Dict[int, Task]

    def entry_tasks(self) -> List[int]:
        """Tasks with no predecessors — valid starting points."""
        return [t.id for t in self.tasks.values() if not t.predecessors]

    def exit_tasks(self) -> List[int]:
        """Tasks with no successors — the DAG's "sink" nodes."""
        return [t.id for t in self.tasks.values() if not t.successors]

    def topological_order(self) -> List[int]:
        """Standard Kahn's-algorithm topological sort.
        HEFT doesn't strictly require this (it uses upward rank for
        priority), but it's useful for validation and for the dynamic
        simulator later."""
        in_degree = {tid: len(t.predecessors) for tid, t in self.tasks.items()}
        queue = [tid for tid, d in in_degree.items() if d == 0]
        order = []
        while queue:
            tid = queue.pop(0)
            order.append(tid)
            for succ_id in self.tasks[tid].successors:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)
        if len(order) != len(self.tasks):
            raise ValueError("DAG contains a cycle — this should never happen "
                              "if generated correctly.")
        return order


@dataclass
class Assignment:
    """Where and when a single task runs."""
    task_id: int
    vm_id: int
    start_time: float
    end_time: float


@dataclass
class Schedule:
    """A complete mapping of every task to a VM + time slot.

    This is the object that HEFT produces, local search mutates, and the
    cost/scoring functions evaluate.
    """
    assignments: Dict[int, Assignment] = field(default_factory=dict)

    def makespan(self) -> float:
        """Total completion time = latest end_time across all tasks."""
        if not self.assignments:
            return 0.0
        return max(a.end_time for a in self.assignments.values())

    def vm_busy_intervals(self, vm_id: int) -> List[Tuple[float, float]]:
        """All (start, end) intervals currently occupied on a given VM,
        sorted by start time. Used by the insertion-based scheduling logic
        to find gaps."""
        intervals = [
            (a.start_time, a.end_time)
            for a in self.assignments.values() if a.vm_id == vm_id
        ]
        return sorted(intervals)

    def copy(self) -> "Schedule":
        """Deep-ish copy — needed by local search, which mutates a candidate
        schedule and must be able to reject it and fall back to the current
        one."""
        new_assignments = {
            tid: Assignment(a.task_id, a.vm_id, a.start_time, a.end_time)
            for tid, a in self.assignments.items()
        }
        return Schedule(assignments=new_assignments)
