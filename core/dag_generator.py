"""
Random layered DAG generator.

Approach:
  1. Split `num_tasks` tasks into levels (level 0 = entry tasks).
  2. Only allow edges from a lower level to a strictly higher level.
     This guarantees the result is acyclic by construction — no need to
     detect/reject cycles after the fact.
  3. Randomize computation cost per task and communication cost per edge.

Tune `num_levels` and `edge_probability` to control dependency density,
as called for in the experimental design (varying "dependency densities").
"""

import random
from typing import List
from core.models import Task, DAG


def generate_random_dag(
    num_tasks: int,
    num_levels: int,
    edge_probability: float = 0.3,
    min_computation_cost: float = 10.0,
    max_computation_cost: float = 100.0,
    min_comm_cost: float = 1.0,
    max_comm_cost: float = 20.0,
    seed: int = None,
) -> DAG:
    """
    Args:
        num_tasks: total number of tasks in the workflow.
        num_levels: how many "layers" to spread tasks across. More levels =
            a longer, narrower DAG (more sequential); fewer levels = a
            wider, flatter DAG (more parallelism).
        edge_probability: probability of drawing an edge from a task in
            level L to a task in level L+1..num_levels-1. Higher = denser
            dependency graph.
        seed: set for reproducible experiments (important for your
            "repeat with multiple random seeds" requirement).

    Returns:
        A DAG with no cycles, where every non-entry task has at least one
        predecessor (so the graph is actually connected, not just acyclic).
    """
    if seed is not None:
        random.seed(seed)

    if num_levels > num_tasks:
        raise ValueError("num_levels cannot exceed num_tasks")

    # --- Step 1: assign each task to a level ---
    # Guarantee at least one task per level, then scatter the rest randomly.
    task_ids = list(range(num_tasks))
    levels: List[List[int]] = [[] for _ in range(num_levels)]
    for i in range(num_levels):
        levels[i].append(task_ids[i])
    for tid in task_ids[num_levels:]:
        levels[random.randint(0, num_levels - 1)].append(tid)

    # --- Step 2: create Task objects ---
    tasks = {
        tid: Task(
            id=tid,
            computation_cost=random.uniform(min_computation_cost, max_computation_cost),
        )
        for tid in task_ids
    }

    # --- Step 3: wire edges level -> later level only (acyclic by construction) ---
    for level_idx in range(num_levels - 1):
        for src_id in levels[level_idx]:
            # candidate targets = every task in a strictly later level
            candidates = [tid for later in levels[level_idx + 1:] for tid in later]
            for dst_id in candidates:
                if random.random() < edge_probability:
                    comm = random.uniform(min_comm_cost, max_comm_cost)
                    tasks[src_id].successors.append(dst_id)
                    tasks[src_id].comm_cost[dst_id] = comm
                    tasks[dst_id].predecessors.append(src_id)

    # --- Step 4: safety net — connect any task that ended up with no
    # predecessor AND isn't in level 0 (edge_probability may have produced
    # orphans). Attach it to a random task in the immediately preceding
    # non-empty level so the DAG stays connected. ---
    for level_idx in range(1, num_levels):
        for tid in levels[level_idx]:
            if not tasks[tid].predecessors:
                prev_levels = [l for l in levels[:level_idx] if l]
                if prev_levels:
                    src_id = random.choice(random.choice(prev_levels))
                    comm = random.uniform(min_comm_cost, max_comm_cost)
                    tasks[src_id].successors.append(tid)
                    tasks[src_id].comm_cost[tid] = comm
                    tasks[tid].predecessors.append(src_id)

    return DAG(tasks=tasks)
