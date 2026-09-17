"""
Heterogeneous VM pool generator.

Design choice: speed and cost are positively correlated (faster VM = more
expensive), with some random noise — this mirrors how real cloud instance
tiers are priced (e.g. AWS compute-optimized vs burstable instances) and is
exactly what makes the cost-vs-makespan trade-off in your objective function
(Score = alpha*Makespan + beta*Cost + gamma*Imbalance) meaningful. If speed
and cost were uncorrelated or inversely correlated, "cost-aware" scheduling
would be trivial (always cheaper AND faster) — a fair simulation needs the
tension
"""

import random
from typing import List
from core.models import VM


def generate_vm_pool(
    num_vms: int,
    min_speed: float = 1.0,
    max_speed: float = 5.0,
    base_cost_per_speed_unit: float = 0.5,
    cost_noise: float = 0.15,
    seed: int = None,
) -> List[VM]:
    """
    Args:
        num_vms: pool size.
        min_speed / max_speed: range of VM processing speeds.
        base_cost_per_speed_unit: roughly, cost_per_hour ~= speed * this
            value, so faster VMs cost proportionally more.
        cost_noise: fractional random noise applied on top of the base
            cost, so the correlation isn't perfectly linear (real cloud
            pricing isn't either — e.g. spot vs on-demand tiers).
        seed: set for reproducibility.

    Returns:
        A list of VM objects with heterogeneous speed/cost.
    """
    if seed is not None:
        random.seed(seed)

    vms = []
    for vm_id in range(num_vms):
        speed = random.uniform(min_speed, max_speed)
        base_cost = speed * base_cost_per_speed_unit
        noise_factor = 1.0 + random.uniform(-cost_noise, cost_noise)
        cost = round(base_cost * noise_factor, 3)
        vms.append(VM(id=vm_id, speed=round(speed, 3), cost_per_hour=cost))

    return vms
