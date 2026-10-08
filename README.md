Things to know about demo
Live demo uses simulated data; run server.py locally for the real optimizer

# DCA-HEFT-ILS

**Dynamic Cost-Aware HEFT Scheduling Optimizer with Iterative Local Search**

A cloud workflow scheduler that starts with classical HEFT, refines it with local search, and extends it with cost-aware multi-objective scoring and dynamic selective re-optimization — built as a cloud computing course project.

## What this is

Workflow scheduling assigns a set of dependent tasks (a DAG) onto a pool of heterogeneous VMs (different speeds, different costs) to minimize completion time. Classical HEFT solves this well but has three limitations this project addresses:

| Limitation of HEFT | What this project adds |
|---|---|
| Only optimizes makespan, ignores cost | **Cost-aware multi-objective scoring** — trades a bit of speed for real cost savings when it's worth it |
| One-shot: never revisits a decision | **Dynamic event simulation** — VM slowdown, VM failure, new task arrival, mid-execution |
| Would have to rebuild the whole schedule on any change | **Selective re-optimization** — repairs only the affected region, freezing everything else |

> **Scope note:** dynamic behavior and VM pricing are *simulated* for experimentation — this does not claim to reproduce real AWS/Azure/GCP billing or infrastructure behavior.

## Architecture

```
DAG + VM pool
      |
      v
  HEFT Engine  --(upward rank + insertion-based EFT)-->  baseline schedule
      |
      v
  Local Search / ILS  --(move / swap / reorder mutations)-->  refined schedule
      |
      v
  Cost-Aware Scoring  --(makespan + cost + imbalance, weighted)-->  cost-aware schedule
      |
      v
  Dynamic Monitor  --(detects VM slowdown / failure / new tasks)-->  change detector
      |
      v
  Selective Re-optimization  --(repairs only affected tasks)-->  adapted schedule
```

## Project structure

```
AWS_Project/
├── core/
│   ├── models.py            # Task, VM, DAG, Schedule, Assignment data structures
│   ├── dag_generator.py     # Random layered DAG generator (acyclic by construction)
│   ├── vm_generator.py      # Heterogeneous VM pool generator (speed/cost correlated)
│   ├── heft_engine.py       # Classical HEFT: upward rank + insertion-based EFT scheduling
│   ├── local_search.py      # Mutation operators + (cost-aware) Iterated Local Search
│   ├── cost_model.py        # Cost/imbalance computation + multi-objective scoring
│   ├── dynamic_events.py    # Event types + affected-task detection + change significance
│   ├── selective_reopt.py   # Repairs only the disrupted part of a running schedule
│   ├── benchmark.py         # Static + dynamic benchmark suites, with baselines
│   └── plots.py             # Generates report-ready comparison/trade-off plots
├── tests/                   # One test module per core component, all passing
├── notes/                   # Day-by-day concept notes (theory + implementation)
├── plots/                   # Generated PNG figures (after running plots.py)
├── dashboard.html            # Interactive dashboard (Gantt view + live metrics)
└── venv/                    # Local virtual environment (not committed)
```

## Setup

```
python -m venv venv
venv\Scripts\activate          # Windows
pip install matplotlib
```

No other dependencies — everything else is the Python standard library.

## Usage

**Run all tests** (from the project root, one module at a time):
```
python -m tests.test_generators
python -m tests.test_heft_engine
python -m tests.test_local_search
python -m tests.test_cost_model
python -m tests.test_dynamic_events
python -m tests.test_selective_reopt
```

**Run the benchmark suite** (prints comparison tables to the console):
```
python -m core.benchmark
```

**Generate report-ready plots** (saved to `plots/`):
```
python -m core.plots
```

**View the dashboard:** open `frontend.html` directly in a browser (self-contained, no server needed).

## Key results

- **Cost-makespan trade-off**: sweeping the cost weight (β) from 0 → 0.95 traces a clean trade-off curve — cost dropped ~4.5% (517 → 494) as makespan rose from ~80 → ~167 time units, confirming the multi-objective scoring works as designed.
- **Selective re-optimization vs naive full reschedule**: selective repair touches ~⅓ as many tasks (5.4 vs 15.2 on average) and runs ~1.8-1.9x faster, with only a ~0.8% difference in resulting makespan.
- **Local search**: plain local search can get stuck at a HEFT-produced local optimum with zero improvement; the Iterated (perturbation-based) version escapes this, achieving up to ~4.7% improvement in isolated testing — motivating the "Iterated" part of the design.

## Novelty summary

> HEFT gives a static schedule; lightweight local search can refine the assignment — that's the existing baseline. This project adds a **dynamic, cost-aware extension**: the scheduler evaluates makespan, estimated VM cost, and utilization together; detects meaningful runtime changes; and selectively re-optimizes only the affected tasks instead of rebuilding the entire schedule from scratch.

## References

1. H. Topcuoglu, S. Hariri, and M.-Y. Wu, "Performance-effective and low-complexity task scheduling for heterogeneous computing," *IEEE Transactions on Parallel and Distributed Systems*, vol. 13, no. 3, pp. 260–274, 2002. — the original paper introducing HEFT and CPOP; defines the upward-rank priority scheme and insertion-based earliest-finish-time scheduling used as the baseline in this project.
