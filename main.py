# IEE598 Mini-Project 1: Simple GA with a base-10 genotype.
# Warren Babb
# 2/15/2026
# This Python script was written with the help of Codex.

#!/usr/bin/env python
"""
Mini-Project 1: Simple GA with a base-10 genotype.

This script solves:
    maximize f(x) = x * sin(10*pi*x) + 1
    subject to x in [-0.5, 1)

What this file does:
1. Implements a base-10 GA from scratch (sign + 4 digits, with crossover + mutation).
2. Implements a mutation-only GA with a single continuous gene.
3. Creates the Question 1 plots in ./outputs.
4. Runs repeated trials and prints a concise Question 2 comparison.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Keep matplotlib cache inside the project so plotting works in restricted envs.
MPLCONFIGDIR = os.path.join(os.getcwd(), ".mplconfig")
CACHE_DIR = os.path.join(os.getcwd(), ".cache")
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


X_MIN = -0.5
X_MAX = 1.0
X_MAX_EXCLUSIVE = float(np.nextafter(np.float64(X_MAX), np.float64(-np.inf)))
PLACE_VALUES = np.array([0.1, 0.01, 0.001, 0.0001], dtype=float)


def objective(x: np.ndarray | float) -> np.ndarray | float:
    """Objective function from the assignment."""
    return x * np.sin(10.0 * np.pi * x) + 1.0


@dataclass
class Base10GAConfig:
    population_size: int = 80
    generations: int = 120
    tournament_size: int = 3
    crossover_prob: float = 0.9
    mutation_prob: float = 0.08  # per-gene mutation probability
    elitism_count: int = 1


@dataclass
class ContinuousGAConfig:
    population_size: int = 80
    generations: int = 120
    tournament_size: int = 3
    mutation_prob: float = 0.9  # per-individual mutation probability
    mutation_std: float = 0.06
    elitism_count: int = 1


@dataclass
class GARunResult:
    best_fitness_history: np.ndarray
    worst_fitness_history: np.ndarray
    avg_fitness_history: np.ndarray
    best_x_history: np.ndarray
    final_population_x: np.ndarray
    final_population_fitness: np.ndarray
    best_x: float
    best_fitness: float
    best_generation: int
    first_hit_generation: Optional[int]


def estimate_true_optimum(num_points: int = 600_001) -> tuple[float, float]:
    """Dense sampling approximation of the continuous optimum."""
    xs = np.linspace(X_MIN, X_MAX_EXCLUSIVE, num_points)
    fs = objective(xs)
    idx = int(np.argmax(fs))
    return float(xs[idx]), float(fs[idx])


def best_discrete_base10_optimum() -> tuple[float, float]:
    """
    Exact best fitness reachable using sign + 4 decimal digits (step 1e-4).
    This is useful for evaluating how close the discrete GA is to the best
    representable value in its own encoding.
    """
    pos = np.arange(0, 10_000, dtype=float) / 10_000.0
    cand = np.concatenate([pos, -pos])
    cand = cand[(cand >= X_MIN) & (cand < X_MAX)]
    cand = np.unique(cand)
    fit = objective(cand)
    idx = int(np.argmax(fit))
    return float(cand[idx]), float(fit[idx])


def decode_base10_population(signs: np.ndarray, digits: np.ndarray) -> np.ndarray:
    """Decode sign + 4 digits into real x values."""
    abs_values = digits.astype(float) @ PLACE_VALUES
    return signs.astype(float) * abs_values


def encode_real_to_base10(x: float) -> tuple[int, np.ndarray]:
    """
    Encode real x into (sign, [d1,d2,d3,d4]) with 1e-4 precision.
    Constraints are enforced by clipping to the feasible range.
    """
    x_clipped = float(np.clip(x, X_MIN, X_MAX_EXCLUSIVE))
    sign = -1 if x_clipped < 0 else 1
    abs_int = int(round(abs(x_clipped) * 10_000))
    abs_int = max(0, min(abs_int, 9_999))
    digit_str = f"{abs_int:04d}"
    digits = np.array([int(ch) for ch in digit_str], dtype=np.int16)
    return sign, digits


def initialize_base10_population(
    pop_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a feasible base-10 population."""
    signs = np.empty(pop_size, dtype=np.int8)
    digits = np.empty((pop_size, 4), dtype=np.int16)
    filled = 0

    while filled < pop_size:
        batch = max(32, 3 * (pop_size - filled))
        trial_signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=batch)
        trial_digits = rng.integers(0, 10, size=(batch, 4), dtype=np.int16)
        trial_x = decode_base10_population(trial_signs, trial_digits)
        valid_mask = (trial_x >= X_MIN) & (trial_x < X_MAX)

        valid_signs = trial_signs[valid_mask]
        valid_digits = trial_digits[valid_mask]
        take = min(pop_size - filled, valid_signs.size)
        if take > 0:
            signs[filled : filled + take] = valid_signs[:take]
            digits[filled : filled + take] = valid_digits[:take]
            filled += take

    return signs, digits


def repair_base10_population(signs: np.ndarray, digits: np.ndarray) -> None:
    """
    In-place repair to keep each genome feasible after crossover/mutation.
    """
    x = decode_base10_population(signs, digits)
    invalid = np.where((x < X_MIN) | (x >= X_MAX))[0]
    for idx in invalid:
        new_sign, new_digits = encode_real_to_base10(float(x[idx]))
        signs[idx] = new_sign
        digits[idx] = new_digits


def pack_genomes(signs: np.ndarray, digits: np.ndarray) -> np.ndarray:
    return np.column_stack((signs, digits)).astype(np.int16)


def unpack_genomes(genomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return genomes[:, 0].astype(np.int8), genomes[:, 1:].astype(np.int16)


def tournament_select_indices(
    fitness: np.ndarray,
    num_select: int,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tournament selection for maximization."""
    candidates = rng.integers(0, fitness.size, size=(num_select, tournament_size))
    row_winner = np.argmax(fitness[candidates], axis=1)
    return candidates[np.arange(num_select), row_winner]


def one_point_crossover(
    parent_genomes: np.ndarray,
    crossover_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One-point crossover on genome [sign, d1, d2, d3, d4]."""
    children = parent_genomes.copy()
    genome_len = children.shape[1]

    for i in range(0, children.shape[0] - 1, 2):
        if rng.random() < crossover_prob:
            point = int(rng.integers(1, genome_len))
            p1 = children[i].copy()
            p2 = children[i + 1].copy()
            children[i, point:] = p2[point:]
            children[i + 1, point:] = p1[point:]

    return children


def mutate_base10_genomes(
    genomes: np.ndarray,
    mutation_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-gene mutation. Sign flips, digits are re-sampled from 0..9."""
    signs, digits = unpack_genomes(genomes)

    sign_mask = rng.random(signs.size) < mutation_prob
    signs[sign_mask] *= -1

    digit_mask = rng.random(digits.shape) < mutation_prob
    random_digits = rng.integers(0, 10, size=digits.shape, dtype=np.int16)
    digits[digit_mask] = random_digits[digit_mask]

    repair_base10_population(signs, digits)
    return pack_genomes(signs, digits)


def run_base10_ga(
    config: Base10GAConfig,
    rng: np.random.Generator,
    target_fitness: Optional[float] = None,
) -> GARunResult:
    """Run base-10 GA with crossover + mutation."""
    signs, digits = initialize_base10_population(config.population_size, rng)

    best_hist: list[float] = []
    worst_hist: list[float] = []
    avg_hist: list[float] = []
    best_x_hist: list[float] = []

    best_fitness_overall = -np.inf
    best_x_overall = 0.0
    best_generation_overall = 0
    first_hit_generation: Optional[int] = None

    for gen in range(config.generations + 1):
        x = decode_base10_population(signs, digits)
        fitness = objective(x)

        best_idx = int(np.argmax(fitness))
        worst_idx = int(np.argmin(fitness))
        gen_best = float(fitness[best_idx])
        gen_worst = float(fitness[worst_idx])
        gen_avg = float(np.mean(fitness))
        gen_best_x = float(x[best_idx])

        best_hist.append(gen_best)
        worst_hist.append(gen_worst)
        avg_hist.append(gen_avg)
        best_x_hist.append(gen_best_x)

        if gen_best > best_fitness_overall:
            best_fitness_overall = gen_best
            best_x_overall = gen_best_x
            best_generation_overall = gen

        if (
            target_fitness is not None
            and first_hit_generation is None
            and gen_best >= target_fitness
        ):
            first_hit_generation = gen

        if gen == config.generations:
            break

        elite_count = max(0, min(config.elitism_count, config.population_size - 1))
        genomes = pack_genomes(signs, digits)

        if elite_count > 0:
            elite_idx = np.argpartition(fitness, -elite_count)[-elite_count:]
            elites = genomes[elite_idx]
        else:
            elites = np.empty((0, genomes.shape[1]), dtype=np.int16)

        num_children = config.population_size - elite_count
        num_parents = num_children if num_children % 2 == 0 else num_children + 1
        parent_idx = tournament_select_indices(
            fitness, num_parents, config.tournament_size, rng
        )
        parents = genomes[parent_idx]
        children = one_point_crossover(parents, config.crossover_prob, rng)
        children = mutate_base10_genomes(children, config.mutation_prob, rng)
        children = children[:num_children]

        next_genomes = np.vstack([elites, children]) if elite_count > 0 else children
        signs, digits = unpack_genomes(next_genomes)

    final_x = decode_base10_population(signs, digits)
    final_fitness = objective(final_x)

    return GARunResult(
        best_fitness_history=np.array(best_hist),
        worst_fitness_history=np.array(worst_hist),
        avg_fitness_history=np.array(avg_hist),
        best_x_history=np.array(best_x_hist),
        final_population_x=final_x,
        final_population_fitness=final_fitness,
        best_x=best_x_overall,
        best_fitness=best_fitness_overall,
        best_generation=best_generation_overall,
        first_hit_generation=first_hit_generation,
    )


def run_continuous_mutation_only_ga(
    config: ContinuousGAConfig,
    rng: np.random.Generator,
    target_fitness: Optional[float] = None,
) -> GARunResult:
    """Single-gene GA with mutation only (no crossover)."""
    population = rng.uniform(X_MIN, X_MAX_EXCLUSIVE, size=config.population_size)

    best_hist: list[float] = []
    worst_hist: list[float] = []
    avg_hist: list[float] = []
    best_x_hist: list[float] = []

    best_fitness_overall = -np.inf
    best_x_overall = 0.0
    best_generation_overall = 0
    first_hit_generation: Optional[int] = None

    for gen in range(config.generations + 1):
        fitness = objective(population)

        best_idx = int(np.argmax(fitness))
        worst_idx = int(np.argmin(fitness))
        gen_best = float(fitness[best_idx])
        gen_worst = float(fitness[worst_idx])
        gen_avg = float(np.mean(fitness))
        gen_best_x = float(population[best_idx])

        best_hist.append(gen_best)
        worst_hist.append(gen_worst)
        avg_hist.append(gen_avg)
        best_x_hist.append(gen_best_x)

        if gen_best > best_fitness_overall:
            best_fitness_overall = gen_best
            best_x_overall = gen_best_x
            best_generation_overall = gen

        if (
            target_fitness is not None
            and first_hit_generation is None
            and gen_best >= target_fitness
        ):
            first_hit_generation = gen

        if gen == config.generations:
            break

        elite_count = max(0, min(config.elitism_count, config.population_size - 1))
        if elite_count > 0:
            elite_idx = np.argpartition(fitness, -elite_count)[-elite_count:]
            elites = population[elite_idx]
        else:
            elites = np.empty(0, dtype=float)

        num_children = config.population_size - elite_count
        parent_idx = tournament_select_indices(
            fitness, num_children, config.tournament_size, rng
        )
        children = population[parent_idx].copy()  # no crossover; copy selected parents

        mutate_mask = rng.random(num_children) < config.mutation_prob
        if np.any(mutate_mask):
            children[mutate_mask] += rng.normal(
                0.0, config.mutation_std, size=int(np.sum(mutate_mask))
            )
        children = np.clip(children, X_MIN, X_MAX_EXCLUSIVE)

        population = np.concatenate([elites, children]) if elite_count > 0 else children

    final_fitness = objective(population)

    return GARunResult(
        best_fitness_history=np.array(best_hist),
        worst_fitness_history=np.array(worst_hist),
        avg_fitness_history=np.array(avg_hist),
        best_x_history=np.array(best_x_hist),
        final_population_x=population.copy(),
        final_population_fitness=final_fitness.copy(),
        best_x=best_x_overall,
        best_fitness=best_fitness_overall,
        best_generation=best_generation_overall,
        first_hit_generation=first_hit_generation,
    )


def save_q1_plots(
    result: GARunResult,
    true_opt_x: float,
    true_opt_f: float,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    gens = np.arange(result.best_fitness_history.size)

    # Plot 1: best, worst, average fitness by generation.
    plt.figure(figsize=(9, 5))
    plt.plot(gens, result.best_fitness_history, label="Best fitness", linewidth=2)
    plt.plot(gens, result.avg_fitness_history, label="Average fitness", linewidth=2)
    plt.plot(gens, result.worst_fitness_history, label="Worst fitness", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Base-10 GA: Fitness Statistics Per Generation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "q1_fitness_history.png"), dpi=180)
    plt.close()

    # Plot 2: best individual x by generation.
    plt.figure(figsize=(9, 5))
    plt.plot(gens, result.best_x_history, color="tab:orange", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Best x")
    plt.title("Base-10 GA: Best Individual Per Generation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "q1_best_individual_history.png"), dpi=180)
    plt.close()

    # Plot 3: objective curve + terminal population + GA best.
    x_curve = np.linspace(X_MIN, X_MAX_EXCLUSIVE, 3000)
    y_curve = objective(x_curve)

    plt.figure(figsize=(9, 5))
    plt.plot(x_curve, y_curve, color="black", linewidth=2, label="f(x)")
    plt.scatter(
        result.final_population_x,
        result.final_population_fitness,
        s=24,
        alpha=0.55,
        color="tab:blue",
        label="Terminal population",
    )
    plt.scatter(
        [result.best_x],
        [result.best_fitness],
        s=120,
        marker="*",
        color="tab:red",
        label="Best GA solution",
        zorder=4,
    )
    plt.scatter(
        [true_opt_x],
        [true_opt_f],
        s=80,
        marker="X",
        color="tab:green",
        label="Estimated continuous optimum",
        zorder=4,
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Objective Function and Base-10 GA Solution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "q1_function_and_solution.png"), dpi=180)
    plt.close()


def summarize_hits(hit_list: list[Optional[int]], total_trials: int) -> tuple[float, float, float]:
    hits = np.array([h for h in hit_list if h is not None], dtype=float)
    success_rate = float(hits.size / total_trials)
    mean_hit = float(np.mean(hits)) if hits.size > 0 else float("nan")
    median_hit = float(np.median(hits)) if hits.size > 0 else float("nan")
    return success_rate, mean_hit, median_hit


def run_comparison_trials(
    base_cfg: Base10GAConfig,
    continuous_cfg: ContinuousGAConfig,
    trials: int,
    target_fitness: float,
    seed: int,
    output_dir: str,
) -> dict[str, float]:
    """
    Run repeated experiments for Question 2 and save a mean trajectory plot.
    """
    base_hits: list[Optional[int]] = []
    cont_hits: list[Optional[int]] = []
    base_final: list[float] = []
    cont_final: list[float] = []
    base_histories: list[np.ndarray] = []
    cont_histories: list[np.ndarray] = []

    for i in range(trials):
        ss = np.random.SeedSequence(seed + i)
        seed_base, seed_cont = ss.spawn(2)
        rng_base = np.random.default_rng(seed_base)
        rng_cont = np.random.default_rng(seed_cont)

        base_result = run_base10_ga(base_cfg, rng_base, target_fitness=target_fitness)
        cont_result = run_continuous_mutation_only_ga(
            continuous_cfg, rng_cont, target_fitness=target_fitness
        )

        base_hits.append(base_result.first_hit_generation)
        cont_hits.append(cont_result.first_hit_generation)
        base_final.append(base_result.best_fitness)
        cont_final.append(cont_result.best_fitness)
        base_histories.append(base_result.best_fitness_history)
        cont_histories.append(cont_result.best_fitness_history)

    base_success, base_mean_hit, base_median_hit = summarize_hits(base_hits, trials)
    cont_success, cont_mean_hit, cont_median_hit = summarize_hits(cont_hits, trials)

    base_final_arr = np.array(base_final)
    cont_final_arr = np.array(cont_final)
    base_hist_mean = np.mean(np.vstack(base_histories), axis=0)
    cont_hist_mean = np.mean(np.vstack(cont_histories), axis=0)

    os.makedirs(output_dir, exist_ok=True)
    gens = np.arange(base_hist_mean.size)
    plt.figure(figsize=(9, 5))
    plt.plot(gens, base_hist_mean, linewidth=2, label="Base-10 GA (mean best fitness)")
    plt.plot(
        gens,
        cont_hist_mean,
        linewidth=2,
        label="Mutation-only GA (mean best fitness)",
        color="tab:orange",
    )
    plt.xlabel("Generation")
    plt.ylabel("Mean best fitness across trials")
    plt.title("Question 2: Mean Best-Fitness Trajectories")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "q2_mean_best_fitness_comparison.png"), dpi=180)
    plt.close()

    return {
        "base_success_rate": base_success,
        "base_mean_hit_gen": base_mean_hit,
        "base_median_hit_gen": base_median_hit,
        "base_mean_final_best": float(np.mean(base_final_arr)),
        "base_std_final_best": float(np.std(base_final_arr)),
        "cont_success_rate": cont_success,
        "cont_mean_hit_gen": cont_mean_hit,
        "cont_median_hit_gen": cont_median_hit,
        "cont_mean_final_best": float(np.mean(cont_final_arr)),
        "cont_std_final_best": float(np.std(cont_final_arr)),
    }


def fmt_hit(value: float) -> str:
    return "not reached" if np.isnan(value) else f"{value:.2f}"


def print_analysis(
    q1_result: GARunResult,
    true_opt_x: float,
    true_opt_f: float,
    discrete_opt_x: float,
    discrete_opt_f: float,
    target_fitness: float,
    q2_stats: dict[str, float],
    output_dir: str,
) -> None:
    print("\n=== Mini-Project 1 Results ===")
    print("Objective: maximize f(x) = x*sin(10*pi*x) + 1 for x in [-0.5, 1)")
    print(f"Estimated continuous optimum: x = {true_opt_x:.6f}, f(x) = {true_opt_f:.6f}")
    print(
        "Best value representable by sign+4-digit encoding: "
        f"x = {discrete_opt_x:.4f}, f(x) = {discrete_opt_f:.6f}"
    )

    print("\nQuestion 1 (Base-10 GA) summary")
    print(
        f"Best GA solution: x = {q1_result.best_x:.4f}, "
        f"f(x) = {q1_result.best_fitness:.6f}, "
        f"first seen at generation {q1_result.best_generation}"
    )
    print(
        f"Gap to estimated continuous optimum: {true_opt_f - q1_result.best_fitness:.6e}"
    )
    print(
        f"Gap to best discrete-encoding optimum: {discrete_opt_f - q1_result.best_fitness:.6e}"
    )
    print(
        "Generated plots:\n"
        f"  - {os.path.join(output_dir, 'q1_fitness_history.png')}\n"
        f"  - {os.path.join(output_dir, 'q1_best_individual_history.png')}\n"
        f"  - {os.path.join(output_dir, 'q1_function_and_solution.png')}"
    )

    print("\nQuestion 2 (Mutation-only GA vs Base-10 GA) summary")
    print(
        "Target fitness used for efficiency comparison: "
        f"f(x) >= {target_fitness:.6f}"
    )
    print(
        "Base-10 GA: "
        f"success rate = {100.0 * q2_stats['base_success_rate']:.1f}% | "
        f"mean hit generation = {fmt_hit(q2_stats['base_mean_hit_gen'])} | "
        f"median hit generation = {fmt_hit(q2_stats['base_median_hit_gen'])} | "
        f"mean final best = {q2_stats['base_mean_final_best']:.6f} +/- "
        f"{q2_stats['base_std_final_best']:.6f}"
    )
    print(
        "Mutation-only GA: "
        f"success rate = {100.0 * q2_stats['cont_success_rate']:.1f}% | "
        f"mean hit generation = {fmt_hit(q2_stats['cont_mean_hit_gen'])} | "
        f"median hit generation = {fmt_hit(q2_stats['cont_median_hit_gen'])} | "
        f"mean final best = {q2_stats['cont_mean_final_best']:.6f} +/- "
        f"{q2_stats['cont_std_final_best']:.6f}"
    )
    print(
        "Comparison plot:\n"
        f"  - {os.path.join(output_dir, 'q2_mean_best_fitness_comparison.png')}"
    )

    if q2_stats["base_mean_hit_gen"] < q2_stats["cont_mean_hit_gen"] or np.isnan(
        q2_stats["cont_mean_hit_gen"]
    ):
        efficiency_msg = "Base-10 GA reached the target faster on average."
    elif q2_stats["cont_mean_hit_gen"] < q2_stats["base_mean_hit_gen"] or np.isnan(
        q2_stats["base_mean_hit_gen"]
    ):
        efficiency_msg = "Mutation-only GA reached the target faster on average."
    else:
        efficiency_msg = "Both methods were similar in average target-hit generation."

    if q2_stats["base_mean_final_best"] > q2_stats["cont_mean_final_best"]:
        quality_msg = "For the same generation budget, Base-10 GA had better final quality."
    elif q2_stats["cont_mean_final_best"] > q2_stats["base_mean_final_best"]:
        quality_msg = (
            "For the same generation budget, Mutation-only GA had better final quality."
        )
    else:
        quality_msg = "For the same generation budget, both had similar final quality."

    print("\nInterpretation")
    print(f"- {efficiency_msg}")
    print(f"- {quality_msg}")
    print(
        "- This aligns with expected behavior: crossover helps broader search early, while "
        "mutation-only behaves more like local random perturbation."
    )


def main() -> None:
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    seed = 598
    rng_q1 = np.random.default_rng(seed)

    base_cfg = Base10GAConfig(
        population_size=80,
        generations=120,
        tournament_size=3,
        crossover_prob=0.9,
        mutation_prob=0.08,
        elitism_count=1,
    )
    cont_cfg = ContinuousGAConfig(
        population_size=80,
        generations=120,
        tournament_size=3,
        mutation_prob=0.9,
        mutation_std=0.06,
        elitism_count=1,
    )

    true_opt_x, true_opt_f = estimate_true_optimum()
    discrete_opt_x, discrete_opt_f = best_discrete_base10_optimum()

    # Question 1 run and plots.
    q1_result = run_base10_ga(base_cfg, rng_q1)
    save_q1_plots(q1_result, true_opt_x, true_opt_f, output_dir)

    # Question 2 multi-trial comparison.
    # A "solution" is defined as reaching near-optimal discrete fitness.
    target_fitness = discrete_opt_f - 1e-4
    q2_stats = run_comparison_trials(
        base_cfg=base_cfg,
        continuous_cfg=cont_cfg,
        trials=40,
        target_fitness=target_fitness,
        seed=10_000,
        output_dir=output_dir,
    )

    print_analysis(
        q1_result=q1_result,
        true_opt_x=true_opt_x,
        true_opt_f=true_opt_f,
        discrete_opt_x=discrete_opt_x,
        discrete_opt_f=discrete_opt_f,
        target_fitness=target_fitness,
        q2_stats=q2_stats,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
