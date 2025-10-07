# file: optimizer/pokemon_party_optimizer.py
"""
Genetic Algorithm for Pokemon Party Optimization (Set-based, Original Score)
- Order-invariant representation (sets of size 6, canonicalized as sorted lists)
- Set-uniform crossover, per-gene mutation
- Jaccard diversity with immigrant injection
- ORIGINAL fitness: sum over columns of min across party (NaN->0), no scaling/soft-min

Enhancements to escape local minima:
- Fitness sharing (niching) using Jaccard distance; selection uses shared fitness
- Deterministic crowding replacement (children compete with most-similar parent)
- Adaptive mutation rate & tournament pressure based on diversity/stagnation
- Soft partial restarts & hypermutation bursts
- Optional novelty bonus shaping (distance-based)
- Fitness caching
"""

import pandas as pd
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
import logging
import json
from pathlib import Path
import datetime
import os
import matplotlib.pyplot as plt
from collections import deque

# ------------------------------ Utilities ------------------------------

def canonical(ind: List[int]) -> List[int]:
    """Keep order irrelevant (why: set representation)."""
    return sorted(ind)

def jaccard_distance(a: List[int], b: List[int]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return 1.0 - (len(A & B) / len(A | B))

def sample_population_diversity(population: List[List[int]], max_pairs: int = 2000) -> float:
    n = len(population)
    if n < 2:
        return 1.0
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = []
        seen = set()
        while len(pairs) < max_pairs:
            i, j = random.randrange(n), random.randrange(n)
            if i == j or (i, j) in seen or (j, i) in seen:
                continue
            seen.add((i, j))
            pairs.append((i, j))
    dists = [jaccard_distance(population[i], population[j]) for i, j in pairs]
    return float(np.mean(dists)) if dists else 1.0

# ------------------------------ Optimizer ------------------------------

class PokemonPartyOptimizer:
    """
    Set-based GA for optimizing 6-Pokemon parties under restrictions.
    Minimization of ORIGINAL score: sum over columns of min across the 6 Pokémon (NaN->0).
    """

    def __init__(self,
                 battle_results_df: pd.DataFrame,
                 restrictions: Dict[str, List[int]] = None,
                 population_size: int = 500,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.85,
                 elitism_count: int = 4,
                 max_generations: int = 100000,
                 diversity_threshold: float = 0.45,
                 restart_threshold: int = 2000,
                 config_path: str = None,
                 tournament_size: int = 2,
                 immigrant_fraction: float = 0.1,
                 # NEW: exploration/adaptation knobs
                 rng_seed: Optional[int] = None,
                 adaptive: bool = True,
                 min_mutation: float = 0.02,
                 max_mutation: float = 0.35,
                 anneal_window: int = 250,   # controls how fast we adapt
                 crowding: bool = True,
                 fitness_sharing: bool = True,
                 sigma_share: float = 0.5,   # Jaccard radius for sharing (0..1)
                 alpha_share: float = 1.0,
                 novelty_weight: float = 0.0, # 0 disables novelty shaping
                 k_novelty: int = 15,
                 soft_restart_fraction: float = 0.25,
                 hypermutation_stag: int = 600,   # kicks in after this stagnation
                 immigrant_on_low_diversity: float = 0.15):
        # RNG
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        self.df = battle_results_df.copy()
        self.population_size = population_size
        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.max_generations = max_generations
        self.diversity_threshold = diversity_threshold
        self.restart_threshold = restart_threshold
        self.tournament_size = tournament_size
        self.immigrant_fraction = immigrant_fraction

        # NEW knobs
        self.adaptive = adaptive
        self.min_mutation = min_mutation
        self.max_mutation = max_mutation
        self.anneal_window = max(1, anneal_window)
        self.use_crowding = crowding
        self.use_fitness_sharing = fitness_sharing
        self.sigma_share = max(1e-6, min(1.0, sigma_share))
        self.alpha_share = max(0.1, alpha_share)
        self.novelty_weight = max(0.0, novelty_weight)
        self.k_novelty = max(1, k_novelty)
        self.soft_restart_fraction = max(0.0, min(0.9, soft_restart_fraction))
        self.hypermutation_stag = max(1, hypermutation_stag)
        self.immigrant_on_low_diversity = immigrant_on_low_diversity

        self.evo_id_pp = list(self.df['evo_id_pp'])
        self.battle_columns = [c for c in self.df.columns if c != 'evo_id_pp']
        self.battle_matrix = self.df[self.battle_columns].values

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.restrictions = self._process_restrictions(restrictions or {}, config_path)
        if self.restrictions:
            self.logger.info(f"Loaded {len(self.restrictions)} restriction groups")

        # History & state
        self.fitness_history: List[float] = []
        self.best_party_history: List[List[int]] = []
        self.diversity_history: List[float] = []
        self.restart_count = 0
        self.stagnation_count = 0
        self.current_diversity = 1.0

        # Cache
        self._fitness_cache: Dict[Tuple[int, ...], float] = {}

        # Small rolling windows for adaptivity
        self._recent_best = deque(maxlen=50)
        self._recent_div = deque(maxlen=50)

    # --------- Restrictions ---------

    def _process_restrictions(self, restrictions: Dict[str, List[int]], config_path: str = None) -> List[List[int]]:
        groups = []
        if not restrictions:
            return groups
        for _, evo_ids in restrictions.items():
            if not (isinstance(evo_ids, list) and len(evo_ids) >= 2):
                continue
            indices = []
            for evo_id in evo_ids:
                for i, evo_id_pp_val in enumerate(self.evo_id_pp):
                    base_evo_id = int(str(evo_id_pp_val).split('_')[0])
                    if base_evo_id == evo_id:
                        indices.append(i)
            if len(indices) >= 2:
                groups.append(indices)
        return groups

    def _violates_restrictions(self, individual: List[int]) -> bool:
        if not self.restrictions:
            return False
        s = set(individual)
        for group in self.restrictions:
            if len(s.intersection(group)) >= 2:
                return True
        return False

    # --------- Individual creation / repair ---------

    def _valid_candidate(self, party: List[int]) -> bool:
        if len(party) != 6:
            return False
        if len(set(party)) != 6:
            return False
        if self._violates_restrictions(party):
            return False
        return True

    def create_individual(self) -> List[int]:
        tries = 0
        while tries < 200:
            cand = random.sample(range(len(self.evo_id_pp)), 6)
            cand = canonical(cand)
            if self._valid_candidate(cand):
                return cand
            tries += 1
        return self._create_valid_individual()

    def _create_valid_individual(self) -> List[int]:
        available = list(range(len(self.evo_id_pp)))
        random.shuffle(available)
        chosen: List[int] = []
        for v in available:
            if len(chosen) == 6:
                break
            trial = canonical(chosen + [v])
            if self._valid_candidate(trial):
                chosen = trial
        if len(chosen) < 6:
            pool = [v for v in range(len(self.evo_id_pp)) if v not in chosen]
            random.shuffle(pool)
            for v in pool:
                trial = canonical(chosen + [v])
                if not self._violates_restrictions(trial):
                    chosen = trial
                if len(chosen) == 6:
                    break
        if len(chosen) < 6:
            pool = [v for v in range(len(self.evo_id_pp)) if v not in chosen]
            while len(chosen) < 6 and pool:
                chosen.append(pool.pop())
            chosen = canonical(chosen[:6])
        return chosen

    def create_initial_population(self) -> List[List[int]]:
        return [self.create_individual() for _ in range(self.population_size)]

    def _repair(self, individual: List[int]) -> List[int]:
        ind = canonical(individual)
        if self._valid_candidate(ind):
            return ind
        ind = list(sorted(set(ind)))
        while len(ind) < 6:
            candidates = [i for i in range(len(self.evo_id_pp)) if i not in ind]
            random.shuffle(candidates)
            placed = False
            for c in candidates:
                trial = canonical(ind + [c])
                if not self._violates_restrictions(trial):
                    ind = trial
                    placed = True
                    break
            if not placed and candidates:
                ind.append(candidates[0])
                ind = canonical(ind)
        for _ in range(20):
            if not self._violates_restrictions(ind):
                break
            pos = random.randrange(6)
            current = ind[pos]
            candidates = [i for i in range(len(self.evo_id_pp)) if i not in ind]
            random.shuffle(candidates)
            for c in candidates:
                trial = ind.copy()
                trial[pos] = c
                trial = canonical(trial)
                if not self._violates_restrictions(trial):
                    ind = trial
                    break
        return canonical(ind)

    # --------- Fitness (ORIGINAL SCORE) ---------

    def _fitness_key(self, individual: List[int]) -> Tuple[int, ...]:
        return tuple(canonical(individual))

    def calculate_fitness(self, individual: List[int]) -> float:
        """Lower is better. Large penalty on violations."""
        key = self._fitness_key(individual)
        if key in self._fitness_cache:
            return self._fitness_cache[key]
        ind = list(key)
        party_data = self.battle_matrix[ind, :]          # [6, n_cols]
        min_values = np.nanmin(party_data, axis=0)       # per column min across 6
        min_values = np.nan_to_num(min_values, nan=0.0)
        score = float(np.sum(min_values))
        if self._violates_restrictions(ind):
            score += 1e12  # enforce feasibility
        self._fitness_cache[key] = score
        return score

    # --------- Selection scoring (sharing & novelty) ---------

    def _pairwise_dist_matrix(self, population: List[List[int]]) -> np.ndarray:
        n = len(population)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = jaccard_distance(population[i], population[j])
                D[i, j] = D[j, i] = d
        return D

    def _fitness_sharing(self,
                         raw_scores: np.ndarray,
                         D: np.ndarray) -> np.ndarray:
        """Shared fitness: raw / niche_count; we minimize shared fitness."""
        n = len(raw_scores)
        sigma = self.sigma_share
        alpha = self.alpha_share
        # sharing function: 1 - (d/sigma)^alpha for d < sigma else 0
        S = np.maximum(0.0, 1.0 - (D / sigma) ** alpha)
        # include self-sharing; avoid div by zero
        niche_sizes = np.sum(S, axis=1)
        niche_sizes = np.maximum(1.0, niche_sizes)
        shared = raw_scores / niche_sizes
        return shared

    def _novelty_bonus(self, D: np.ndarray) -> np.ndarray:
        """Avg distance to k nearest neighbors."""
        if self.novelty_weight <= 0.0:
            return np.zeros(D.shape[0], dtype=float)
        n = D.shape[0]
        k = min(self.k_novelty, max(1, n - 1))
        sorted_d = np.sort(D, axis=1)[:, 1:k + 1]  # skip self (0)
        avgk = np.mean(sorted_d, axis=1)
        return avgk * self.novelty_weight

    def _selection_scores(self, population: List[List[int]], raw_scores: List[float]) -> np.ndarray:
        raw = np.asarray(raw_scores, dtype=float)
        if not self.use_fitness_sharing and self.novelty_weight <= 0.0:
            return raw
        D = self._pairwise_dist_matrix(population)
        shared = self._fitness_sharing(raw, D) if self.use_fitness_sharing else raw
        novelty = self._novelty_bonus(D)
        # since we minimize: lower is better; subtract novelty to prefer diversity
        return shared - novelty

    # --------- Selection / Crossover / Mutation ---------

    def _current_tournament_size(self) -> int:
        """Reduce pressure when stagnating (why: exploration)."""
        if not self.adaptive:
            return self.tournament_size
        # scale 2..base using recent stagnation
        base = max(2, self.tournament_size)
        # more stagnation -> smaller k
        if self.stagnation_count == 0:
            return base
        k = base - min(base - 2, int(self.stagnation_count / (self.anneal_window // 2 + 1)))
        return max(2, k)

    def tournament_selection(self, population: List[List[int]], selection_scores: List[float]) -> List[int]:
        k = self._current_tournament_size()
        idxs = random.sample(range(len(population)), k)
        # minimize selection_scores
        best_idx = min(idxs, key=lambda i: selection_scores[i])
        return population[best_idx].copy()

    def set_uniform_crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        if random.random() > self.crossover_rate:
            return canonical(p1), canonical(p2)
        U = list(set(p1) | set(p2))
        def make_child() -> List[int]:
            chosen = [v for v in U if random.random() < 0.5]
            chosen = list(set(chosen))
            if len(chosen) > 6:
                chosen = random.sample(chosen, 6)
            elif len(chosen) < 6:
                remaining = [v for v in range(len(self.evo_id_pp)) if v not in chosen]
                random.shuffle(remaining)
                for r in remaining:
                    chosen.append(r)
                    if len(chosen) == 6:
                        break
            return canonical(chosen)
        c1, c2 = make_child(), make_child()
        c1, c2 = self._repair(c1), self._repair(c2)
        return c1, c2

    def mutate(self, individual: List[int], extra_replacements: int = 0, mutation_rate: Optional[float] = None) -> List[int]:
        """Per-gene mutation; extra replacements when diversity low."""
        rate = self.mutation_rate if mutation_rate is None else mutation_rate
        ind = individual.copy()
        s = set(ind)
        for i in range(6):
            if random.random() < rate:
                current = ind[i]
                candidates = [v for v in range(len(self.evo_id_pp)) if v not in s]
                random.shuffle(candidates)
                for c in candidates:
                    trial = ind.copy()
                    trial[i] = c
                    trial = canonical(trial)
                    if not self._violates_restrictions(trial):
                        s.discard(current)
                        s.add(c)
                        ind = trial
                        break
        for _ in range(extra_replacements):
            pos = random.randrange(6)
            current = ind[pos]
            candidates = [v for v in range(len(self.evo_id_pp)) if v not in s]
            random.shuffle(candidates)
            for c in candidates:
                trial = ind.copy()
                trial[pos] = c
                trial = canonical(trial)
                if not self._violates_restrictions(trial):
                    s.discard(current)
                    s.add(c)
                    ind = trial
                    break
        return canonical(ind)

    # --------- Diversity / Immigration / Evolution ---------

    def inject_immigrants(self, population: List[List[int]], fitness_scores: List[float], fraction: float) -> List[List[int]]:
        count = max(1, int(len(population) * fraction))
        order = np.argsort(fitness_scores)  # ascending
        worst = list(order[-count:])
        new_pop = population.copy()
        for idx in worst:
            imm = self.create_individual()
            attempts = 0
            while attempts < 50 and any(imm == ex for ex in new_pop):
                imm = self.create_individual()
                attempts += 1
            new_pop[idx] = imm
        return new_pop

    def _soft_partial_restart(self, population: List[List[int]], raw_scores: List[float]) -> List[List[int]]:
        """Reinit a fraction of worst non-elites. Clears cache entries impacted."""
        if self.soft_restart_fraction <= 0.0:
            return population
        keep = max(0, self.elitism_count)
        order = np.argsort(raw_scores)  # best first
        survivors = [population[i] for i in order[:keep]]
        # replace bottom fraction
        n_replace = int(self.population_size * self.soft_restart_fraction)
        n_replace = max(keep, n_replace)
        to_replace_idxs = order[-n_replace:]
        for _ in to_replace_idxs:
            survivors.append(self.create_individual())
        # fill with remaining best of the middle to preserve size
        middle = [population[i] for i in order[keep:-n_replace]] if n_replace < len(order) - keep else []
        new_pop = survivors + middle
        new_pop = new_pop[:self.population_size]
        # clear cache partially (why: structure changed significantly)
        self._fitness_cache = {}
        return new_pop

    def evolve(self) -> Tuple[List[int], float]:
        global_best_individual: Optional[List[int]] = None
        global_best_fitness = np.inf
        self.logger.info(f"Starting evolution | pop={self.population_size} gens={self.max_generations} restag={self.restart_threshold}")

        max_restarts = 3
        for restart in range(max_restarts + 1):
            if restart > 0:
                self.logger.info(f"=== RESTART {restart} ===")
                self.restart_count += 1

            population = self.create_initial_population()
            self.stagnation_count = 0
            self._fitness_cache = {}

            for generation in range(self.max_generations):
                raw_scores = [self.calculate_fitness(ind) for ind in population]
                selection_scores = self._selection_scores(population, raw_scores).tolist()

                diversity = sample_population_diversity(population)
                self.current_diversity = diversity
                self.diversity_history.append(diversity)

                best_idx = int(np.argmin(raw_scores))
                best_fitness = float(raw_scores[best_idx])
                best_individual = population[best_idx].copy()

                improved = best_fitness < global_best_fitness - 1e-12
                if improved:
                    global_best_fitness = best_fitness
                    global_best_individual = best_individual.copy()
                    self.stagnation_count = 0
                else:
                    self.stagnation_count += 1

                self.fitness_history.append(best_fitness)
                self.best_party_history.append(best_individual)
                self._recent_best.append(best_fitness)
                self._recent_div.append(diversity)

                if generation % 100 == 0 or generation == self.max_generations - 1:
                    avg_fit = float(np.mean(raw_scores))
                    self.logger.info(f"Gen {generation:4d} | Best={best_fitness:.6f} Global={global_best_fitness:.6f} "
                                     f"Avg={avg_fit:.6f} Div(J)={diversity:.3f} Stag={self.stagnation_count} "
                                     f"Mut={self.mutation_rate:.3f} Tourn={self._current_tournament_size()}")

                # --- Adaptive controls ---
                extra_repl = 0
                if self.adaptive:
                    # widen exploration when diversity low
                    if diversity < self.diversity_threshold:
                        self.mutation_rate = min(self.max_mutation, self.mutation_rate * 1.25)
                        self.immigrant_fraction = max(self.immigrant_fraction, self.immigrant_on_low_diversity)
                        extra_repl = 1
                    else:
                        # anneal mutation slowly toward base/min
                        target = max(self.min_mutation, self.base_mutation_rate)
                        self.mutation_rate = self.mutation_rate - (self.mutation_rate - target) / self.anneal_window

                    # hypermutation bursts when stagnating long
                    if self.stagnation_count >= self.hypermutation_stag:
                        self.mutation_rate = self.max_mutation
                        extra_repl = max(extra_repl, 2)

                # Immigrants on low diversity
                if diversity < self.diversity_threshold:
                    population = self.inject_immigrants(population, raw_scores, self.immigrant_fraction)
                    # refresh scores post-immigration for replacement step
                    raw_scores = [self.calculate_fitness(ind) for ind in population]
                    selection_scores = self._selection_scores(population, raw_scores).tolist()

                # Soft partial restart on very long stagnation
                if self.stagnation_count and self.stagnation_count % self.restart_threshold == 0:
                    self.logger.info(f"Soft partial restart at gen {generation} (stagnation={self.stagnation_count})")
                    population = self._soft_partial_restart(population, raw_scores)
                    continue  # next gen with fresh pop

                # --- Next generation (elitism + crowding replacement) ---
                new_population: List[List[int]] = []

                # Small elitism
                elite_indices = list(np.argsort(raw_scores)[:self.elitism_count])
                for idx in elite_indices:
                    new_population.append(population[idx].copy())

                # build mating pool using selection scores (minimize)
                while len(new_population) < self.population_size:
                    p1 = self.tournament_selection(population, selection_scores)
                    p2 = self.tournament_selection(population, selection_scores)
                    c1, c2 = self.set_uniform_crossover(p1, p2)
                    c1 = self.mutate(c1, extra_replacements=extra_repl)
                    c2 = self.mutate(c2, extra_replacements=extra_repl)

                    if self.use_crowding:
                        # deterministic crowding: child competes with nearest parent
                        d11 = jaccard_distance(p1, c1)
                        d12 = jaccard_distance(p1, c2)
                        d21 = jaccard_distance(p2, c1)
                        d22 = jaccard_distance(p2, c2)

                        # match to minimize total distance
                        if d11 + d22 <= d12 + d21:
                            match = [(p1, c1), (p2, c2)]
                        else:
                            match = [(p1, c2), (p2, c1)]

                        for parent, child in match:
                            parent_f = self.calculate_fitness(parent)
                            child_f = self.calculate_fitness(child)
                            # replace only if strictly better (why: maintain niches)
                            winner = child if child_f < parent_f else parent
                            new_population.append(winner)
                            if len(new_population) >= self.population_size:
                                break
                    else:
                        new_population.extend([c1, c2])

                population = [canonical(ind) for ind in new_population[:self.population_size]]

                # Hard restart guard: if soft restart didn't help significantly, allow loop restart
                if self.stagnation_count >= self.restart_threshold:
                    self.logger.info(f"Stagnation {self.stagnation_count} >= {self.restart_threshold} → hard restart break")
                    break

            # stop using further hard restarts if last phase improved
            if self.stagnation_count < self.restart_threshold // 2:
                break

        self.logger.info(f"Optimization complete | Restarts used: {self.restart_count} | Best fitness (original score): {global_best_fitness:.6f}")
        return global_best_individual or [], global_best_fitness

    # --------- Reporting ---------

    def get_party_details(self, individual: List[int]) -> pd.DataFrame:
        evo_id_pp_values = [self.evo_id_pp[i] for i in individual]
        party_data = self.battle_matrix[individual, :]
        details = []
        for i, evo_idx in enumerate(individual):
            details.append({
                'Position': i + 1,
                'Evo_ID_PP': self.evo_id_pp[evo_idx],
                'Avg_Performance': np.nanmean(party_data[i, :]),
                'Min_Performance': np.nanmin(party_data[i, :]),
                'Max_Performance': np.nanmax(party_data[i, :])
            })
        return pd.DataFrame(details)

# ------------------------------ API ------------------------------

def estimate_best_party_multi_gen(config_path: str = None, thorough: bool = True, **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Run party optimization for multiple generations as specified in config.
    Returns a dictionary with generation results keyed by 'gen_X' where X is the generation number.
    """
    # Load config
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")
            config = {}
    else:
        config = {}
    
    # Get generations to run (default to [1,2,3])
    gen_config = config.get('gen', [1,2,3])
    generations_to_run = gen_config if isinstance(gen_config, list) else [gen_config]
    
    # Determine paths
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    intermediate_files_dir = script_dir.parent / 'intermediate_files'
    
    all_results = {}
    
    for target_gen in generations_to_run:
        battle_results_path = intermediate_files_dir / f'battle_results_gen{target_gen}.csv'
        
        if not battle_results_path.exists():
            logging.warning(f"Battle results file not found at {battle_results_path}, skipping generation {target_gen}")
            continue
        
        try:
            logging.info(f"Running optimization for generation {target_gen}")
            results = estimate_best_party(
                battle_results_path=str(battle_results_path),
                config_path=config_path,
                thorough=thorough,
                **kwargs
            )
            all_results[f'gen_{target_gen}'] = results
            logging.info(f"Generation {target_gen} optimization completed with fitness: {results['best_fitness']:.6f}")
            
        except Exception as e:
            logging.error(f"Error during generation {target_gen} optimization: {e}")
            continue
    
    return all_results

def estimate_best_party(battle_results_path: str, config_path: str = None, thorough: bool = True, **kwargs) -> Dict[str, Any]:
    df = pd.read_csv(battle_results_path)

    restrictions = None
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                restrictions = config.get('restrictions', {})
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")

    optimizer = PokemonPartyOptimizer(df, restrictions=restrictions, config_path=config_path, **kwargs)
    best_party, best_fitness = optimizer.evolve()
    best_evo_id_pp = [optimizer.evo_id_pp[i] for i in best_party] if best_party else []

    party_details = optimizer.get_party_details(best_party) if best_party else pd.DataFrame()

    results = {
        'best_party_indices': best_party,
        'best_evo_id_pp': best_evo_id_pp,
        'best_fitness': best_fitness,
        'party_details': party_details,
        'fitness_history': optimizer.fitness_history,
        'optimization_generations': len(optimizer.fitness_history)
    }

    _save_results_to_file(results, config_path, optimizer)
    return results

def _save_results_to_file(results: Dict[str, Any], config_path: str = None, optimizer: PokemonPartyOptimizer = None):
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results")
        if not results_dir.exists():
            project_root = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd()
            results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"best_party_{timestamp}.txt"
        filepath = results_dir / filename

        config_settings = {}
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_settings = json.load(f)
            except Exception as e:
                logging.warning(f"Could not load config from {config_path}: {e}")

        gen_config = config_settings.get('gen', [1, 2, 3])
        generations_to_try = gen_config if isinstance(gen_config, list) else [gen_config]

        pokemon_stats_df = None
        for gen in generations_to_try:
            stats_path = f'data/gen_{gen}/stats_gen_{gen}.csv'
            if not Path(stats_path).exists():
                project_root = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
                stats_path = project_root / f'data/gen_{gen}/stats_gen_{gen}.csv'
            try:
                gen_stats_df = pd.read_csv(stats_path)
                pokemon_stats_df = gen_stats_df if pokemon_stats_df is None else pd.concat([pokemon_stats_df, gen_stats_df], ignore_index=True)
                logging.info(f"Loaded {len(gen_stats_df)} Pokemon stats from gen {gen}")
            except Exception as e:
                logging.warning(f"Could not load Pokemon stats from gen {gen} at {stats_path}: {e}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"POKEMON PARTY OPTIMIZATION RESULTS\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")

            f.write(f"FINAL FITNESS SCORE (original sum of mins): {results['best_fitness']:.6f}\n\n")

            f.write("CHOSEN POKEMON PARTY:\n")
            f.write("-" * 40 + "\n")
            for i, evo_id_pp in enumerate(results['best_evo_id_pp'], 1):
                base_evo_id = int(str(evo_id_pp).split('_')[0])
                pokemon_info = "Unknown Pokemon"
                pokemon_types = "Unknown"
                if pokemon_stats_df is not None:
                    matching = pokemon_stats_df[pokemon_stats_df['evo_id'] == base_evo_id]
                    if not matching.empty:
                        unique_pokemon = matching.drop_duplicates(subset=['pokemon'])
                        names = unique_pokemon['pokemon'].tolist()
                        types_list = unique_pokemon['types'].tolist()
                        if len(names) == 1:
                            pokemon_info = names[0]
                            pokemon_types = types_list[0]
                        else:
                            pokemon_info = " -> ".join(names)
                            pokemon_types = types_list[0]
                f.write(f"{i}. {pokemon_info} (Evo ID: {base_evo_id})\n")
                f.write(f"   Types: {pokemon_types}\n")
                f.write(f"   Evo ID PP: {evo_id_pp}\n\n")

            f.write("\nCONFIGURATION SETTINGS:\n")
            f.write("-" * 40 + "\n")
            if config_settings:
                for key, value in config_settings.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write("No configuration settings loaded\n")

            f.write("\nOPTIMIZATION DETAILS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Generations: {results['optimization_generations']}\n")
            f.write(f"Best Fitness Score: {results['best_fitness']:.6f}\n")
            if results['fitness_history']:
                initial_fitness = results['fitness_history'][0]
                improvement = initial_fitness - results['best_fitness']
                pct = (improvement / initial_fitness * 100) if initial_fitness > 0 else 0
                f.write(f"Initial Fitness (gen0 best): {initial_fitness:.6f}\n")
                f.write(f"Improvement: {improvement:.6f} ({pct:.2f}%)\n")

            f.write("\n" + "="*60 + "\n")

        print(f"\nResults saved to: {filepath}")

        if optimizer is not None:
            try:
                png_filepath = filepath.with_suffix('.png')
                _create_party_performance_visualization(results, optimizer, png_filepath)
                print(f"Visualization saved to: {png_filepath}")
                
                unique_png_filepath = filepath.with_name(filepath.stem + "_unique_best").with_suffix('.png')
                _create_unique_best_performance_visualization(results, optimizer, unique_png_filepath)
                print(f"Unique best performance visualization saved to: {unique_png_filepath}")
            except Exception as viz_error:
                logging.error(f"Error creating visualization: {viz_error}")
                print(f"Warning: Could not create visualization: {viz_error}")

    except Exception as e:
        logging.error(f"Error saving results to file: {e}")
        print(f"Warning: Could not save results to file: {e}")

def _create_party_performance_visualization(results: Dict[str, Any], optimizer: PokemonPartyOptimizer, png_filepath: Path):
    """Visualize ORIGINAL per-encounter mins from raw values (why: match score semantics)."""
    best_party_indices = results['best_party_indices']
    if not best_party_indices:
        return
    party_battle_data = optimizer.battle_matrix[best_party_indices, :]
    min_values = np.nanmin(party_battle_data, axis=0)
    min_values = np.nan_to_num(min_values, nan=0.0)

    # Names
    pokemon_names = []
    best_evo_id_pp = results['best_evo_id_pp']
    pokemon_stats_df = None
    for gen in [1, 2, 3]:
        stats_path = f'data/gen_{gen}/stats_gen_{gen}.csv'
        if not Path(stats_path).exists():
            project_root = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
            stats_path = project_root / f'data/gen_{gen}/stats_gen_{gen}.csv'
        try:
            gen_stats_df = pd.read_csv(stats_path)
            pokemon_stats_df = gen_stats_df if pokemon_stats_df is None else pd.concat([pokemon_stats_df, gen_stats_df], ignore_index=True)
        except Exception:
            pass

    for evo_id_pp in best_evo_id_pp:
        base_evo_id = int(str(evo_id_pp).split('_')[0])
        if pokemon_stats_df is not None:
            matching = pokemon_stats_df[pokemon_stats_df['evo_id'] == base_evo_id]
            if not matching.empty:
                item_evolutions = matching[matching['evo_item'].notna() & (matching['evo_item'] != '')]
                if not item_evolutions.empty:
                    final_evo = item_evolutions.loc[item_evolutions['evo_lvl'].idxmax()]
                else:
                    final_evo = matching.loc[matching['evo_lvl'].idxmax()]
                pokemon_names.append(f"{final_evo['pokemon']} ({evo_id_pp})")
            else:
                pokemon_names.append(f"Evo {evo_id_pp}")
        else:
            pokemon_names.append(f"Evo {evo_id_pp}")

    individual_pokemon_data = np.nan_to_num(party_battle_data, nan=0.0)

    def value_to_color(value: float):
        if np.isnan(value) or value >= 1_000_000_000:
            return 'black'
        if value > 1:
            return 'yellow'
        if value >= 0:
            ratio = max(0.0, min(1.0, value))
            return (ratio, 0, 1 - ratio)
        return 'blue'

    fig, axes = plt.subplots(7, 1, figsize=(16, 12), sharex=True)
    n_encounters = len(min_values)
    x_positions = np.arange(n_encounters)

    for i in range(6):
        vals = individual_pokemon_data[i, :]
        colors = [value_to_color(v) for v in vals]
        axes[i].bar(x_positions, height=1, width=1.0, color=colors, edgecolor='none')
        axes[i].set_xlim(-0.5, n_encounters - 0.5)
        axes[i].set_ylim(0, 1)
        label = pokemon_names[i] if i < len(pokemon_names) else f'Pokemon {i+1}'
        axes[i].set_ylabel(label, fontsize=10, rotation=0, ha='right', va='center')
        axes[i].set_yticks([])
        axes[i].set_xticks([])

    min_colors = [value_to_color(v) for v in min_values]
    axes[6].bar(x_positions, height=1, width=1.0, color=min_colors, edgecolor='none')
    axes[6].set_xlim(-0.5, n_encounters - 0.5)
    axes[6].set_ylim(0, 1)
    axes[6].set_ylabel('Party Min', fontsize=10, rotation=0, ha='right', va='center')
    axes[6].set_yticks([])
    axes[6].set_xticks(np.arange(0, n_encounters, max(1, n_encounters // 10)))
    axes[6].set_xlabel('Encounter ID', fontsize=12)

    fig.suptitle('Party Performance Across Encounters (Original Raw Values)', fontsize=14, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(png_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def _create_unique_best_performance_visualization(results: Dict[str, Any], optimizer: PokemonPartyOptimizer, png_filepath: Path):
    """Visualize only values where one pokemon uniquely outperforms all others in the party for that encounter."""
    best_party_indices = results['best_party_indices']
    if not best_party_indices:
        return
    party_battle_data = optimizer.battle_matrix[best_party_indices, :]
    
    # Names
    pokemon_names = []
    best_evo_id_pp = results['best_evo_id_pp']
    pokemon_stats_df = None
    for gen in [1, 2, 3]:
        stats_path = f'data/gen_{gen}/stats_gen_{gen}.csv'
        if not Path(stats_path).exists():
            project_root = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
            stats_path = project_root / f'data/gen_{gen}/stats_gen_{gen}.csv'
        try:
            gen_stats_df = pd.read_csv(stats_path)
            pokemon_stats_df = gen_stats_df if pokemon_stats_df is None else pd.concat([pokemon_stats_df, gen_stats_df], ignore_index=True)
        except Exception:
            pass

    for evo_id_pp in best_evo_id_pp:
        base_evo_id = int(str(evo_id_pp).split('_')[0])
        if pokemon_stats_df is not None:
            matching = pokemon_stats_df[pokemon_stats_df['evo_id'] == base_evo_id]
            if not matching.empty:
                item_evolutions = matching[matching['evo_item'].notna() & (matching['evo_item'] != '')]
                if not item_evolutions.empty:
                    final_evo = item_evolutions.loc[item_evolutions['evo_lvl'].idxmax()]
                else:
                    final_evo = matching.loc[matching['evo_lvl'].idxmax()]
                pokemon_names.append(f"{final_evo['pokemon']} ({evo_id_pp})")
            else:
                pokemon_names.append(f"Evo {evo_id_pp}")
        else:
            pokemon_names.append(f"Evo {evo_id_pp}")

    individual_pokemon_data = np.nan_to_num(party_battle_data, nan=0.0)
    n_encounters = party_battle_data.shape[1]

    unique_best_data = np.full_like(individual_pokemon_data, np.nan)
    
    for encounter_idx in range(n_encounters):
        encounter_values = individual_pokemon_data[:, encounter_idx]
        min_value = np.nanmin(encounter_values)
        min_count = np.sum(encounter_values == min_value)
        if min_count == 1:
            best_pokemon_idx = np.argmin(encounter_values)
            unique_best_data[best_pokemon_idx, encounter_idx] = encounter_values[best_pokemon_idx]

    def value_to_color(value: float):
        if np.isnan(value):
            return (1, 1, 1, 0)
        if value >= 1_000_000_000:
            return 'black'
        if value > 1:
            return 'yellow'
        if value >= 0:
            ratio = max(0.0, min(1.0, value))
            return (ratio, 0, 1 - ratio)
        return 'blue'

    fig, axes = plt.subplots(6, 1, figsize=(16, 10), sharex=True)
    x_positions = np.arange(n_encounters)

    for i in range(6):
        vals = unique_best_data[i, :]
        colors = [value_to_color(v) for v in vals]
        for j, (val, color) in enumerate(zip(vals, colors)):
            if not np.isnan(val):
                axes[i].bar(j, height=1, width=1.0, color=color, edgecolor='none')
        
        axes[i].set_xlim(-0.5, n_encounters - 0.5)
        axes[i].set_ylim(0, 1)
        label = pokemon_names[i] if i < len(pokemon_names) else f'Pokemon {i+1}'
        axes[i].set_ylabel(label, fontsize=10, rotation=0, ha='right', va='center')
        axes[i].set_yticks([])
        axes[i].set_xticks([])

    axes[5].set_xticks(np.arange(0, n_encounters, max(1, n_encounters // 10)))
    axes[5].set_xlabel('Encounter ID', fontsize=12)

    fig.suptitle('Unique Best Performance Across Encounters\n(Only showing bars where one Pokemon uniquely outperforms all others)', fontsize=14, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(png_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# ------------------------------ Sample Data (for testing) ------------------------------

def create_sample_data(n_variants: int = 50, n_battles: int = 100) -> pd.DataFrame:
    evo_id_pp_values = list(range(1, n_variants + 1))
    data = {'evo_id_pp': evo_id_pp_values}
    for battle in range(1, n_battles + 1):
        values = np.random.beta(2, 5, n_variants)
        outlier_mask = np.random.random(n_variants) < 0.05
        values[outlier_mask] = np.random.uniform(100, 1_000_000, np.sum(outlier_mask))
        nan_mask = np.random.random(n_variants) < 0.02
        values[nan_mask] = np.nan
        data[str(battle)] = values
    return pd.DataFrame(data)

# ------------------------------ CLI ------------------------------

if __name__ == "__main__":
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    config_file_path = script_dir.parent / 'config' / 'config.json'

    if not config_file_path.exists():
        print(f"Error: Config file not found at {config_file_path}")
        exit(1)

    try:
        with open(config_file_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        exit(1)

    print(f"Loaded configuration from: {config_file_path}")
    print(f"Config: {config}")

    gen_config = config.get('gen', [1,2,3])
    generations_to_run = gen_config if isinstance(gen_config, list) else [gen_config]
    
    print(f"Generations to optimize: {generations_to_run}")
    
    intermediate_files_dir = script_dir.parent / 'intermediate_files'
    
    all_results = {}
    use_thorough = True
    
    for gen_idx, target_gen in enumerate(generations_to_run):
        print(f"\n{'='*60}")
        print(f"OPTIMIZING GENERATION {target_gen} ({gen_idx + 1}/{len(generations_to_run)})")
        print(f"{'='*60}")
        
        battle_results_path = intermediate_files_dir / f'battle_results_gen{target_gen}.csv'

        if not battle_results_path.exists():
            print(f"Error: Battle results file not found at {battle_results_path}")
            print("Available files in intermediate_files:")
            if intermediate_files_dir.exists():
                for file in intermediate_files_dir.glob('battle_results_gen*.csv'):
                    print(f"  - {file.name}")
            print(f"Skipping generation {target_gen}")
            continue

        print(f"Loading battle results from: {battle_results_path}")
        df = pd.read_csv(battle_results_path)
        print(f"Battle results shape: {df.shape}")
        print(f"Evo ID PP entries: {len(df)}, Battle scenarios: {len(df.columns) - 1}")
        print("\nData preview:")
        print(df.head())

        print(f"\nRunning set-based genetic algorithm optimization for Generation {target_gen} (ORIGINAL score)...")
        print("Using thorough optimization parameters for production")

        try:
            results = estimate_best_party(
                battle_results_path=str(battle_results_path),
                config_path=str(config_file_path),
                thorough=use_thorough
            )
            
            all_results[f'gen_{target_gen}'] = results
            
            print(f"\nGeneration {target_gen} optimization completed!")
            print(f"Best fitness score (MINIMIZED, original): {results['best_fitness']:.6f}")
            print(f"Best party (evo_id_pp): {results['best_evo_id_pp']}")
            print(f"Total generations: {results['optimization_generations']}")

            print("\nParty Performance Details:")
            if not results['party_details'].empty:
                print(results['party_details'].round(4))
            else:
                print("No details available.")

            if results['fitness_history']:
                initial_fitness = results['fitness_history'][0]
                final_fitness = results['best_fitness']
                improvement = initial_fitness - final_fitness
                improvement_pct = (improvement / initial_fitness * 100) if initial_fitness > 0 else 0
                print(f"\nFitness improvement: {initial_fitness:.6f} -> {final_fitness:.6f}")
                print(f"Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")

        except Exception as e:
            print(f"Error during generation {target_gen} optimization: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    
    if all_results:
        print("Results for all generations:")
        for gen_key, gen_results in all_results.items():
            gen_num = gen_key.split('_')[1]
            print(f"Generation {gen_num}: Best fitness = {gen_results['best_fitness']:.6f}")
        print(f"\nAll results have been saved to the results directory.")
    else:
        print("No successful optimizations completed.")
        exit(1)
