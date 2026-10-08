"""Bridge between the Python scheduler (core/) and the browser dashboards.

Run:  python server.py
Open: http://localhost:5000          (new dashboard  -> dashboard.html)
      http://localhost:5000/classic  (old page       -> frontend.html)

Why the lock?  core/ seeds Python's *global* random generator, so two requests
running at once would corrupt each other's "reproducible" results.  Every heavy
endpoint takes LOCK so they run one at a time.
"""
import json
import math
import random
import statistics
import threading
import time

from flask import Flask, Response, jsonify, request, send_from_directory

from core.benchmark import run_dynamic_benchmark, run_static_benchmark
from core.cost_model import (
    compute_average_utilization,
    compute_load_balance_index,
    compute_schedule_cost,
    compute_task_cost,
)
from core.dag_generator import generate_random_dag
from core.heft_engine import (
    _earliest_finish_time_on_vm,
    compute_upward_ranks,
    heft_schedule,
)
from core.local_search import (
    EPSILON,
    MUTATION_OPERATORS,
    cost_aware_local_search,
    is_feasible,
    local_search,
)
from core.models import Assignment, Schedule
from core.vm_generator import generate_vm_pool

app = Flask(__name__)
LOCK = threading.Lock()

MAX_TASKS, MAX_VMS = 120, 16
TRACE_POINTS = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vms_to_json(vms):
    return [{"id": v.id, "speed": round(v.speed, 2),
             "cost_per_hour": round(v.cost_per_hour, 2)} for v in vms]


def schedule_to_json(schedule, dag, vms):
    """Gantt data + headline metrics + per-VM workload for one schedule."""
    vm_by_id = {vm.id: vm for vm in vms}
    busy = {v.id: 0.0 for v in vms}
    count = {v.id: 0 for v in vms}
    for a in schedule.assignments.values():
        busy[a.vm_id] += a.end_time - a.start_time
        count[a.vm_id] += 1
    makespan = schedule.makespan()
    return {
        "tasks": [
            {
                "id": a.task_id,
                "vm": a.vm_id,
                "start": round(a.start_time, 2),
                "end": round(a.end_time, 2),
                "cost": round(compute_task_cost(a.task_id, schedule, dag, vm_by_id), 2),
            }
            for a in sorted(schedule.assignments.values(), key=lambda a: a.task_id)
        ],
        "makespan": round(makespan, 2),
        "cost": round(compute_schedule_cost(schedule, dag, vms), 2),
        "utilization": round(compute_average_utilization(schedule, vms), 2),
        "loadBalance": round(compute_load_balance_index(schedule, vms), 4),
        "vmBusy": [round(busy[v.id], 2) for v in vms],
        "vmTasks": [count[v.id] for v in vms],
        "vmUtil": [round(busy[v.id] / makespan, 4) if makespan > 0 else 0.0 for v in vms],
    }


def build_instance(num_tasks, num_vms, seed):
    levels = max(3, min(14, round(math.sqrt(num_tasks)) + 1))
    dag = generate_random_dag(num_tasks, levels, edge_probability=0.3, seed=seed)
    vms = generate_vm_pool(num_vms, seed=seed, max_speed=8.0)
    return dag, vms


def round_robin_schedule(dag, vms):
    """Naive baseline: deal tasks to VMs in turn, in dependency order."""
    schedule = Schedule()
    for i, task_id in enumerate(dag.topological_order()):
        vm = vms[i % len(vms)]
        start, end = _earliest_finish_time_on_vm(dag.tasks[task_id], vm, dag, schedule)
        schedule.assignments[task_id] = Assignment(task_id, vm.id, start, end)
    return schedule


def iteration_budget(num_tasks):
    """Shrink the search budget as graphs grow so a click stays responsive."""
    return max(300, min(1000, int(1000 * (40 / max(num_tasks, 1)) ** 1.5)))


def traced_ils(initial, dag, vms, iterations, seed, per_round=100, kick=5):
    """Iterated local search, same algorithm as core.local_search.iterated_local_search,
    but it also records (a) the best makespan after every iteration and
    (b) per-operator attempts / feasible / accepted counts."""
    random.seed(seed)
    ops = {op.__name__.replace("mutate_", ""): {"attempts": 0, "feasible": 0, "accepted": 0}
           for op in MUTATION_OPERATORS}
    current = best = initial
    trace = [initial.makespan()]
    rounds = max(1, iterations // per_round)

    for _ in range(rounds):
        for _ in range(per_round):
            op = random.choice(MUTATION_OPERATORS)
            stats = ops[op.__name__.replace("mutate_", "")]
            stats["attempts"] += 1
            candidate = op(current, dag, vms)
            if is_feasible(candidate, dag):
                stats["feasible"] += 1
                if candidate.makespan() < current.makespan() - EPSILON:
                    current = candidate
                    stats["accepted"] += 1
            trace.append(min(best.makespan(), current.makespan()))
        if current.makespan() < best.makespan() - EPSILON:
            best = current
        for _ in range(kick):  # perturbation: accept feasible-but-worse moves
            op = random.choice(MUTATION_OPERATORS)
            candidate = op(current, dag, vms)
            if is_feasible(candidate, dag):
                current = candidate

    if current.makespan() < best.makespan() - EPSILON:
        best = current
    trace.append(best.makespan())
    return best, trace, ops


def downsample(values, points=TRACE_POINTS):
    if len(values) <= points:
        return [round(v, 2) for v in values]
    step = (len(values) - 1) / (points - 1)
    return [round(values[round(i * step)], 2) for i in range(points)]


def find_improving_seed(num_tasks, num_vms, start_seed, iterations, tries=30, min_gain=0.5):
    """Scan seeds until the local search beats HEFT by at least `min_gain` %."""
    for k in range(tries):
        seed = start_seed + k
        dag, vms = build_instance(num_tasks, num_vms, seed)
        heft = heft_schedule(dag, vms)
        best, _, _ = traced_ils(heft, dag, vms, iterations, seed)
        gain = (heft.makespan() - best.makespan()) / heft.makespan() * 100
        if gain >= min_gain:
            return seed, k + 1
    return None, tries


def run_pipeline(num_tasks, num_vms, seed, beta, iterations):
    stages = []

    def timed(label, fn):
        t0 = time.perf_counter()
        value = fn()
        stages.append({"label": label, "ms": round((time.perf_counter() - t0) * 1000, 2)})
        return value

    dag, vms = timed("Generate workflow", lambda: build_instance(num_tasks, num_vms, seed))
    naive = timed("Round-robin", lambda: round_robin_schedule(dag, vms))
    timed("Rank tasks", lambda: compute_upward_ranks(dag))
    heft = timed("HEFT", lambda: heft_schedule(dag, vms))
    ils, trace, ops = timed(
        "Iterated local search", lambda: traced_ils(heft, dag, vms, iterations, seed)
    )
    cost_aware, _ = timed(
        "Cost-aware search",
        lambda: cost_aware_local_search(
            heft, dag, vms,
            alpha=(1.0 - beta) * 0.8, beta=beta, gamma=(1.0 - beta) * 0.2,
            max_iterations=max(200, iterations // 2), seed=seed,
        ),
    )

    return {
        "instance": {
            "num_tasks": num_tasks,
            "num_vms": num_vms,
            "num_edges": sum(len(t.successors) for t in dag.tasks.values()),
            "seed": seed,
            "beta": beta,
            "iterations": iterations,
        },
        "vms": vms_to_json(vms),
        "stages": stages,
        "naive": schedule_to_json(naive, dag, vms),
        "heft": schedule_to_json(heft, dag, vms),
        "ils": schedule_to_json(ils, dag, vms),
        "cost_aware": schedule_to_json(cost_aware, dag, vms),
        "trace": downsample(trace),
        "traceLength": len(trace),
        "operators": [{"name": k, **v} for k, v in ops.items()],
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


@app.route("/classic")
def classic():
    return send_from_directory(".", "frontend.html")


# ---------------------------------------------------------------------------
# New dashboard API
# ---------------------------------------------------------------------------

@app.route("/api/schedule")
def api_schedule():
    """Generate a workflow and schedule it five ways. Query params:
    tasks, vms, seed (blank = random), beta, find=1 (scan for a seed where search helps)."""
    try:
        num_tasks = max(5, min(MAX_TASKS, int(request.args.get("tasks", 30))))
        num_vms = max(2, min(MAX_VMS, int(request.args.get("vms", 6))))
        beta = max(0.0, min(0.95, float(request.args.get("beta", 0.4))))
        seed_arg = request.args.get("seed", "").strip()
        seed = int(seed_arg) if seed_arg else random.randrange(10_000)
    except ValueError:
        return jsonify({"error": "tasks, vms, seed and beta must be numbers"}), 400

    iterations = iteration_budget(num_tasks)
    scan = None
    with LOCK:
        if request.args.get("find") == "1":
            found, tries = find_improving_seed(num_tasks, num_vms, seed, iterations)
            scan = {"found": found is not None, "tries": tries}
            if found is not None:
                seed = found
        result = run_pipeline(num_tasks, num_vms, seed, beta, iterations)
    if scan:
        result["scan"] = scan
    return jsonify(result)


# ---------------------------------------------------------------------------
# Averaged benchmarks (same shapes as STATIC / DYNAMIC / TRADEOFF)
# ---------------------------------------------------------------------------

def _avg(rows, key):
    return statistics.mean(r[key] for r in rows)


@app.route("/api/benchmark/static")
def benchmark_static():
    trials = int(request.args.get("trials", 30))
    seed_base = int(request.args.get("seed", 1000))
    with LOCK:
        rows = run_static_benchmark(
            num_trials=trials, num_tasks=20, num_levels=6, num_vms=5,
            ls_iterations=500, seed_base=seed_base,
        )
    return jsonify({
        "trials": len(rows),
        "methods": [
            {
                "key": key,
                "makespan": round(_avg(rows, f"{key}_makespan"), 2),
                "cost": round(_avg(rows, f"{key}_cost"), 2),
                "utilization": round(_avg(rows, f"{key}_utilization"), 2),
                "loadBalance": round(_avg(rows, f"{key}_load_balance"), 4),
            }
            for key in ("heft", "ls", "cost_aware")
        ],
    })


@app.route("/api/benchmark/dynamic")
def benchmark_dynamic():
    trials = int(request.args.get("trials", 20))
    seed_base = int(request.args.get("seed", 2000))
    with LOCK:
        rows = run_dynamic_benchmark(
            num_trials=trials, num_tasks=25, num_levels=7, num_vms=6, seed_base=seed_base,
        )
    return jsonify({
        "trials": len(rows),
        "methods": [
            {
                "key": "selective",
                "tasksTouched": round(_avg(rows, "tasks_touched_selective"), 2),
                "repairMs": round(_avg(rows, "selective_time_sec") * 1000, 4),
                "stability": round(_avg(rows, "stability_selective"), 1),
                "makespan": round(_avg(rows, "selective_makespan"), 2),
            },
            {
                "key": "naive",
                "tasksTouched": round(_avg(rows, "tasks_touched_naive"), 2),
                "repairMs": round(_avg(rows, "naive_time_sec") * 1000, 4),
                "stability": round(_avg(rows, "stability_naive"), 1),
                "makespan": round(_avg(rows, "naive_makespan"), 2),
            },
        ],
    })


@app.route("/api/benchmark/tradeoff")
def benchmark_tradeoff():
    """Sweep the cost weight beta -> one (makespan, cost) point per beta.
    Mirrors core/plots.py (re-implemented so matplotlib isn't required)."""
    num_dags = int(request.args.get("dags", 8))
    seed_base = int(request.args.get("seed", 3000))
    betas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]

    with LOCK:
        instances = []
        for i in range(num_dags):
            seed = seed_base + i
            dag = generate_random_dag(num_tasks=18, num_levels=6, edge_probability=0.35, seed=seed)
            vms = generate_vm_pool(num_vms=5, seed=seed, max_speed=8.0)
            instances.append((dag, vms, heft_schedule(dag, vms)))

        points = []
        for beta in betas:
            makespans, costs = [], []
            for dag, vms, heft_result in instances:
                result, _ = cost_aware_local_search(
                    heft_result, dag, vms,
                    alpha=(1.0 - beta) * 0.8, beta=beta, gamma=(1.0 - beta) * 0.2,
                    max_iterations=400, seed=42,
                )
                makespans.append(result.makespan())
                costs.append(compute_schedule_cost(result, dag, vms))
            points.append({"beta": beta,
                           "makespan": round(statistics.mean(makespans), 2),
                           "cost": round(statistics.mean(costs), 2)})
    return jsonify(points)


# ---------------------------------------------------------------------------
# Old endpoints (still used by /classic)
# ---------------------------------------------------------------------------

@app.route("/api/run")
def run():
    """One-shot: HEFT, HEFT+LS and HEFT+Cost-ILS on a fresh random instance."""
    seed = int(request.args.get("seed", 1))
    dag = generate_random_dag(
        num_tasks=int(request.args.get("tasks", 14)),
        num_levels=int(request.args.get("levels", 5)),
        edge_probability=0.35, seed=seed,
    )
    vms = generate_vm_pool(num_vms=int(request.args.get("vms", 4)), seed=seed, max_speed=8.0)
    iters = int(request.args.get("iters", 500))
    with LOCK:
        heft = heft_schedule(dag, vms)
        ls = local_search(heft, dag, vms, max_iterations=iters, seed=seed)
        ca, _ = cost_aware_local_search(
            heft, dag, vms, alpha=0.5, beta=float(request.args.get("beta", 0.4)),
            gamma=0.1, max_iterations=iters, seed=seed,
        )
    return jsonify({
        "vms": vms_to_json(vms),
        "heft": schedule_to_json(heft, dag, vms),
        "ls": schedule_to_json(ls, dag, vms),
        "cost_aware": schedule_to_json(ca, dag, vms),
    })


@app.route("/api/stream")
def stream():
    """Live: run local search in small rounds and push each result (SSE)."""
    seed = int(request.args.get("seed", 1))
    dag = generate_random_dag(
        num_tasks=int(request.args.get("tasks", 14)),
        num_levels=int(request.args.get("levels", 5)),
        edge_probability=0.35, seed=seed,
    )
    vms = generate_vm_pool(num_vms=int(request.args.get("vms", 4)), seed=seed, max_speed=8.0)
    rounds = int(request.args.get("rounds", 30))
    per_round = int(request.args.get("per_round", 20))

    def gen():
        with LOCK:
            current = heft_schedule(dag, vms)
        first = {"round": 0, "vms": vms_to_json(vms), **schedule_to_json(current, dag, vms)}
        yield f"data: {json.dumps(first)}\n\n"
        for r in range(1, rounds + 1):
            with LOCK:
                current = local_search(current, dag, vms, max_iterations=per_round, seed=seed + r)
            yield f"data: {json.dumps({'round': r, **schedule_to_json(current, dag, vms)})}\n\n"
            time.sleep(0.15)
        yield "event: done\ndata: {}\n\n"

    return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
