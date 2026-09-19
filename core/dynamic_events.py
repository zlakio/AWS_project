"""
Dynamic event simulation for DCA-HEFT-ILS.

Three event types a running schedule can experience:
    - VMSlowdownEvent:      a VM's processing speed drops
    - VMFailureEvent:       a VM goes down entirely
    - NewTaskArrivalEvent:  a new task shows up mid-execution

Simplifying assumption used throughout (standard for this kind of
simulation, and worth stating explicitly in your report): a task that has
ALREADY FINISHED by `current_time` is untouched by any event. A task that
is currently running or not yet started on the AFFECTED VM is treated as
needing to be rescheduled from scratch at `current_time` -- we don't model
partial/resumable execution, since that adds real complexity for little
extra insight in a course project.
"""

from dataclasses import dataclass
from typing import List, Set, Union

from core.models import DAG, Schedule, Task


@dataclass
class VMSlowdownEvent:
    vm_id: int
    time: float
    speed_multiplier: float  # e.g. 0.5 = VM is now half as fast


@dataclass
class VMFailureEvent:
    vm_id: int
    time: float


@dataclass
class NewTaskArrivalEvent:
    time: float
    task: Task  # predecessors should reference task ids that already exist in the DAG


DynamicEvent = Union[VMSlowdownEvent, VMFailureEvent, NewTaskArrivalEvent]


# ---------------------------------------------------------------------------
# Affected-task detection
# ---------------------------------------------------------------------------


def get_affected_task_ids(
    dag: DAG,
    schedule: Schedule,
    event: DynamicEvent,
    current_time: float,
) -> Set[int]:
    """
    Returns the set of task ids that need to be reconsidered because of
    this event. A task counts as affected if:
      - it's on the event's VM (for slowdown/failure), AND
      - it hasn't finished yet at current_time (end_time > current_time)
        -- already-completed work is never undone.

    For a NewTaskArrivalEvent, the only "affected" task is the new task
    itself (it doesn't exist in any prior schedule yet, so nothing else
    needs to move because of it -- Day 8's repair logic decides where to
    slot it in).
    """
    if isinstance(event, NewTaskArrivalEvent):
        return {event.task.id}

    affected = set()
    for task_id, assignment in schedule.assignments.items():
        if assignment.vm_id == event.vm_id and assignment.end_time > current_time:
            affected.add(task_id)
    return affected


# ---------------------------------------------------------------------------
# Change significance ("is this worth reacting to?")
# ---------------------------------------------------------------------------


def compute_disruption_magnitude(
    affected_task_ids: Set[int],
    schedule: Schedule,
    current_time: float,
) -> float:
    """
    A simple magnitude metric: total REMAINING scheduled work (in time
    units) that's now in question, summed across all affected tasks.

    remaining_work(t) = max(0, original_end_time(t) - current_time)

    Using "remaining work" rather than just "number of affected tasks"
    matters: 1 affected task with 50 units left to run is a much bigger
    disruption than 5 affected tasks that were each about to finish
    anyway.
    """
    total = 0.0
    for task_id in affected_task_ids:
        # A brand-new task (from NewTaskArrivalEvent) has no prior
        # assignment to measure against -- treat its own duration
        # separately in the caller if needed; here we only sum tasks
        # that already had a scheduled end_time.
        if task_id in schedule.assignments:
            end_time = schedule.assignments[task_id].end_time
            total += max(0.0, end_time - current_time)
    return total


def is_significant_change(magnitude: float, threshold: float) -> bool:
    """
    The actual "change detector": decides whether a disruption is big
    enough to trigger selective re-optimization (Day 8), or small enough
    to just ignore and let the existing schedule play out.

    `threshold` is a tunable parameter -- too low and you re-optimize on
    every tiny blip (wasteful); too high and you ignore disruptions that
    genuinely hurt the schedule. This is a deliberate design knob to
    discuss in your report, similar to alpha/beta/gamma for the score.
    """
    return magnitude > threshold
