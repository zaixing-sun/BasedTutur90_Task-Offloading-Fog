import random
import numpy as np
from core.env import Env
from core.task import Task
# Assume the following functions have been defined (or imported):
# dominates, crowding_distance, non_dominated_sort,
# tournament_selection, crossover, mutate, select_next_generation

class Individual:
    def __init__(self, weights, obs_type=["cpu", "buffer", "bw"]):
        self.weights = weights
        self.obs_type = obs_type

    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)
        
    def _make_observation(self, env: Env, task: Task, obs_type=["cpu", "buffer", "bw"]):
        """
        Returns a flat observation vector.
        For instance, returns free CPU frequency for each node combined with free bandwidth per link.
        """
        if env is None:
            raise ValueError("Environment must be provided.")
        obs = []
        if "cpu" in obs_type:
            cpu_obs = [env.scenario.get_node(node_name).free_cpu_freq 
                       for node_name in env.scenario.get_nodes()]
            obs += cpu_obs
            
        if "buffer" in obs_type:
            buffer_obs = [env.scenario.get_node(node_name).buffer_free_size()
                          for node_name in env.scenario.get_nodes()]
            obs += buffer_obs
            
        if "bw" in obs_type:
            bw_obs = [env.scenario.get_link(link_name[0], link_name[1]).free_bandwidth
                      for link_name in env.scenario.get_links()]
            obs += bw_obs

        return np.array(obs)

    def act(self, env, task):
        """
        Compute an observation vector, then use successive matrix multiplications 
        (with ReLU between hidden layers) to produce scores for each node.
        """
        obs = self._make_observation(env, task, self.obs_type)
        for i in range(len(self.weights)):
            obs = np.dot(obs, self.weights[i])
            if i < len(self.weights) - 1:
                obs = Individual.ReLU(obs)
        return np.argmax(obs), obs


class NPGAPolicy:
    def __init__(self, env, config):
        self.config = config
        self.env = env

        self.obs_type = config["model"]["obs_type"]
        self.d_model = config["model"]["d_model"]
        self.n_layers = config["model"]["n_layers"]
        
        # Compute observation dimension using a helper (similar to Individual._make_observation).
        self.n_observations = len(self._make_observation(self.env, None, self.obs_type))
        
        self.num_actions = len(self.env.scenario.node_id2name)
        
        # Initialize population as a list of weight matrices (each individual is a list of matrices)
        self.population = [self.genenerate_individual() 
                           for _ in range(config["training"]["pop_size"])]
    
    def _make_observation(self, env, task, obs_type):
        if env is None:
            raise ValueError("Environment must be provided to determine observation size.")
        obs = []
        if "cpu" in obs_type:
            cpu_obs = [env.scenario.get_node(node_name).free_cpu_freq 
                       for node_name in env.scenario.get_nodes()]
            obs += cpu_obs
        if "buffer" in obs_type:
            buffer_obs = [env.scenario.get_node(node_name).buffer_free_size()
                          for node_name in env.scenario.get_nodes()]
            obs += buffer_obs
        if "bw" in obs_type:
            bw_obs = [env.scenario.get_link(link_name[0], link_name[1]).free_bandwidth
                      for link_name in env.scenario.get_links()]
            obs += bw_obs
        return obs

    def genenerate_individual(self):
        """
        Generate a new individual with random weights.
        """
        if self.n_layers < 1:
            raise ValueError("The number of layers must be at least 1.")
        elif self.n_layers == 1:
            weights = [np.random.rand(self.n_observations, self.num_actions)]
        elif self.n_layers == 2:
            weights = [np.random.rand(self.n_observations, self.d_model), 
                       np.random.rand(self.d_model, self.num_actions)]
        else:
            weights = [np.random.rand(self.n_observations, self.d_model)]
            for i in range(self.n_layers - 2):
                weights.append(np.random.rand(self.d_model, self.d_model))
            weights.append(np.random.rand(self.d_model, self.num_actions))
        return weights
    
    def individuals(self):
        """
        Wrap the current population into Individual objects.
        """
        return [Individual(weights, self.obs_type) for weights in self.population]
            
    def best_individual(self, fitness):
        """
        Select the best individual via scalarization.
        Here we use a weighted sum where higher success rate is better and 
        lower latency and power are better.
        """
        lambda_param = self.config["training"].get("latency_weight", 0.1)
        mu_param = self.config["training"].get("power_weight", 0.1)
        scores = [sr - lambda_param * l - mu_param * e for (sr, l, e) in fitness]
        best_idx = np.argmax(scores)
        return fitness[best_idx]
    def dominates(obj1, obj2):
        """
        Check if objective vector obj1 dominates obj2 (assuming minimization).
        """
        better_or_equal = all(a <= b for a, b in zip(obj1, obj2))
        strictly_better = any(a < b for a, b in zip(obj1, obj2))
        return better_or_equal and strictly_better
    
    def crowding_distance(fitness_list):
        """
        Compute crowding distance for a list of objective vectors.
        """
        num_individuals = len(fitness_list)
        if num_individuals == 0:
            return []
        distances = [0.0] * num_individuals
        num_objectives = len(fitness_list[0])
        for m in range(num_objectives):
            values = [fit[m] for fit in fitness_list]
            sorted_indices = sorted(range(num_individuals), key=lambda i: values[i])
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            for i in range(1, num_individuals - 1):
                if max(values) - min(values) == 0:
                    diff = 0
                else:
                    diff = (values[sorted_indices[i+1]] - values[sorted_indices[i-1]]) / (max(values) - min(values))
                distances[sorted_indices[i]] += diff
        return distances
    
    def crossover(parent1, parent2):
        """
        Perform arithmetic crossover between two parent weight matrices.
        """
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
    
    def update(self, fitness):
        """
        Update the population using NSGA-II based multi-objective selection.
        
        Parameters:
          fitness: list of tuples (success_rate, avg_latency, avg_power) for the current population.
          
        Note:
          Since NSGA operators assume minimization, we convert the objectives as:
              obj = (-success_rate, avg_latency, avg_power)
          Offspring are generated via tournament selection, arithmetic crossover, and mutation.
          (For demonstration purposes, offspring fitness is simulated by perturbing a random parent's fitness.)
        """
        pop_size = len(self.population)
        # Convert parent fitness to minimization objectives.
        parent_obj = [(-sr, l, e) for sr, l, e in fitness]
        
        offspring = []
        offspring_fitness = []
        for _ in range(pop_size):
            # Select two parents via tournament selection.
            parent1 = tournament_selection(self.population, parent_obj, niche_radius=0.1)
            parent2 = tournament_selection(self.population, parent_obj, niche_radius=0.1)
            # Perform arithmetic crossover for each weight matrix.
            child = []
            for w1, w2 in zip(parent1, parent2):
                alpha = random.random()
                # The provided crossover returns two children, but here we generate one.
                child_w = self.crossover(w1, w2)[0]
                child.append(child_w)
            # Apply mutation to the child's weight matrices.
            mutated_child = [mutate(w, self.config["training"].get("mutation_rate", 0.1)) for w in child]
            offspring.append(mutated_child)
            # For demonstration, simulate offspring fitness by taking a random parent's fitness and adding a small noise.
            parent_fitness_sample = random.choice(fitness)
            noise = (random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01))
            offspring_fitness.append(tuple(p + n for p, n in zip(parent_fitness_sample, noise)))
        
        offspring_obj = [(-sr, l, e) for sr, l, e in offspring_fitness]
        
        # Combine current population (parents) with offspring.
        combined_population = self.population + offspring
        combined_obj = parent_obj + offspring_obj
        
        # Select next generation using NSGA-II selection (non-dominated sorting and crowding distance).
        new_population, _ = select_next_generation(combined_population, combined_obj, pop_size)
        self.population = new_population
