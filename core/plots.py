"""
Generates report-ready plots (Days 14-15 territory) from the benchmark
suite built on Day 9. Run directly: python -m core.plots

Produces three PNGs in the `plots/` folder:
    1. static_comparison.png     -- HEFT vs LS vs Cost-ILS (makespan & cost)
    2. cost_makespan_tradeoff.png -- Pareto-style curve as beta varies
    3. dynamic_comparison.png    -- selective repair vs naive reschedule
"""

import os
import statistics

import matplotlib

matplotlib.use("Agg")  # no GUI needed -- just save files
import matplotlib.pyplot as plt

from core.benchmark import run_dynamic_benchmark, run_static_benchmark
from core.cost_model import compute_schedule_cost
from core.dag_generator import generate_random_dag
from core.heft_engine import heft_schedule
from core.local_search import cost_aware_local_search
from core.vm_generator import generate_vm_pool

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot 1: Static comparison bar chart
# ---------------------------------------------------------------------------


def plot_static_comparison(results):
    approaches = ["HEFT only", "HEFT + LS", "HEFT + Cost-ILS"]
    makespans = [
        statistics.mean(r["heft_makespan"] for r in results),
        statistics.mean(r["ls_makespan"] for r in results),
        statistics.mean(r["cost_aware_makespan"] for r in results),
    ]
    costs = [
        statistics.mean(r["heft_cost"] for r in results),
        statistics.mean(r["ls_cost"] for r in results),
        statistics.mean(r["cost_aware_cost"] for r in results),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.bar(approaches, makespans, color=["#4C72B0", "#55A868", "#C44E52"])
    ax1.set_title("Average Makespan")
    ax1.set_ylabel("Makespan (time units)")
    ax1.tick_params(axis="x", rotation=15)

    ax2.bar(approaches, costs, color=["#4C72B0", "#55A868", "#C44E52"])
    ax2.set_title("Average Cost")
    ax2.set_ylabel("Cost ($)")
    ax2.tick_params(axis="x", rotation=15)

    fig.suptitle("Static Comparison: HEFT vs Local Search vs Cost-Aware ILS")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "static_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 2: Cost-vs-makespan trade-off curve (sweeping beta)
# ---------------------------------------------------------------------------


def plot_cost_makespan_tradeoff(num_dags: int = 8, betas=None):
    """
    For a fixed set of random DAGs, sweeps the cost weight (beta) from
    "makespan-focused" to "cost-focused" and plots the resulting average
    (makespan, cost) point for each beta -- this traces out the trade-off
    curve directly, which is the clearest possible visual evidence for
    Novelty A.
    """
    if betas is None:
        betas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]

    dags_and_vms = []
    for i in range(num_dags):
        seed = 3000 + i
        dag = generate_random_dag(
            num_tasks=18, num_levels=6, edge_probability=0.35, seed=seed
        )
        vms = generate_vm_pool(num_vms=5, seed=seed, max_speed=8.0)
        dags_and_vms.append((dag, vms, heft_schedule(dag, vms)))

    curve_points = []
    for beta in betas:
        alpha = (1.0 - beta) * 0.8  # keep most of the non-cost weight on makespan
        gamma = (1.0 - beta) * 0.2  # small imbalance weight throughout
        makespans, costs = [], []
        for dag, vms, heft_result in dags_and_vms:
            result, _ = cost_aware_local_search(
                heft_result,
                dag,
                vms,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                max_iterations=400,
                seed=42,
            )
            makespans.append(result.makespan())
            costs.append(compute_schedule_cost(result, dag, vms))
        curve_points.append((beta, statistics.mean(makespans), statistics.mean(costs)))
        print(
            f"beta={beta:.2f} -> avg makespan={statistics.mean(makespans):.2f}, "
            f"avg cost={statistics.mean(costs):.2f}"
        )

    fig, ax = plt.subplots(figsize=(6, 5))
    xs = [p[1] for p in curve_points]  # makespan
    ys = [p[2] for p in curve_points]  # cost
    ax.plot(xs, ys, marker="o", color="#C44E52")
    for beta, ms, cost in curve_points:
        ax.annotate(
            f"β={beta:.2f}",
            (ms, cost),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.set_xlabel("Average Makespan")
    ax.set_ylabel("Average Cost ($)")
    ax.set_title("Cost vs Makespan Trade-off as Cost Weight (β) Increases")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cost_makespan_tradeoff.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plot 3: Selective repair vs naive full reschedule
# ---------------------------------------------------------------------------


def plot_dynamic_comparison(results):
    labels = ["Selective", "Naive (full)"]
    tasks_touched = [
        statistics.mean(r["tasks_touched_selective"] for r in results),
        statistics.mean(r["tasks_touched_naive"] for r in results),
    ]
    repair_times = [
        statistics.mean(r["selective_time_sec"] for r in results)
        * 1000,  # ms for readability
        statistics.mean(r["naive_time_sec"] for r in results) * 1000,
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    ax1.bar(labels, tasks_touched, color=["#55A868", "#C44E52"])
    ax1.set_title("Avg Tasks Touched per Repair")
    ax1.set_ylabel("Number of Tasks")

    ax2.bar(labels, repair_times, color=["#55A868", "#C44E52"])
    ax2.set_title("Avg Repair Time")
    ax2.set_ylabel("Time (ms)")

    fig.suptitle("Selective Re-optimization vs Naive Full Reschedule")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "dynamic_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    _ensure_output_dir()

    print("Running static benchmark for Plot 1...")
    static_results = run_static_benchmark(num_trials=20)
    plot_static_comparison(static_results)

    print("\nSweeping beta for Plot 2 (this takes a bit longer)...")
    plot_cost_makespan_tradeoff()

    print("\nRunning dynamic benchmark for Plot 3...")
    dynamic_results = run_dynamic_benchmark(num_trials=15)
    plot_dynamic_comparison(dynamic_results)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
