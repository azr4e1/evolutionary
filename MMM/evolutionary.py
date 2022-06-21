# implement evolutionary algorithm
from abc import ABC, abstractmethod
import logging
import numpy as np


class BaseIndividual(ABC):
    """
    Template class for generation's individuals
    """
    def __init__(self, init_params=None):
        if init_params.get("value", None) is not None:
            self.value = init_params["value"]
        else:
            self.value = self._random_init(init_params)

    @abstractmethod
    def pair(self, other, pair_params):
        pass

    @abstractmethod
    def mutate(self, mutate_params):
        pass

    @abstractmethod
    def _random_init(self, init_params):
        pass


class Population:
    def __init__(self, size, fitness, individual_class, init_params):
        self.fitness = fitness
        self.individuals = [individual_class(init_params=init_params) for _ in range(size)]
        self.individuals.sort(key=lambda x: self.fitness(x))

    def replace(self, new_individuals):
        size = len(self.individuals)
        self.individuals.extend(new_individuals)
        self.individuals.sort(key=lambda x: self.fitness(x))
        self.individuals = self.individuals[-size:]

    def get_parents(self, n_offsprings):
        mothers = self.individuals[-2 * n_offsprings::2]
        fathers = self.individuals[-2 * n_offsprings + 1::2]

        return mothers, fathers


class Evolution:
    def __init__(self, pool_size, fitness, individual_class, n_offsprings, pair_params, mutate_params, init_params, history=25):
        self.pair_params = pair_params
        self.mutate_params = mutate_params
        self.mutate_params_copy = mutate_params.copy()
        self.pool = Population(pool_size, fitness, individual_class, init_params)
        self.n_offsprings = n_offsprings
        self.history = history

    def step(self, generation: int):
        mothers, fathers = self.pool.get_parents(self.n_offsprings)
        offsprings = []

        for mother, father in zip(mothers, fathers):
            offspring = mother.pair(father, self.pair_params)
            offspring.mutate(self.boost(generation))
            offsprings.append(offspring)

        self.pool.replace(offsprings)

    def boost(self, generation):
        """
        Every 100 generations,
        if there is no significant improvement,
        double the rate of mutation
        """
        if not generation % self.history:
            rel_diff = abs((self.century_gen[-1] - self.century_gen[-2]) / self.century_gen[-2])
            if rel_diff < 10:
                length_rate = len(self.mutate_params_copy['rate'])
                boost_factor = self.mutate_params.get("boost_factor", 20)
                self.mutate_params_copy['rate'] = self.mutate_params['rate'] * np.random.uniform(1, boost_factor, (1, length_rate))
            else:
                self.mutate_params_copy['rate'] = self.mutate_params['rate']
        else:
            self.mutate_params_copy['rate'] = self.mutate_params['rate']

        return self.mutate_params_copy


    def champion(self):
        return self.pool.individuals[-1]


    def __call__(self, max_generations, verbose: bool = False):
        try:
            self.century_gen = [self.pool.fitness(self.champion())]

            if verbose:
                logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(message)s")
            if isinstance(max_generations, dict):
                generation = 1
                while True:
                    score = self.pool.fitness(self.champion())
                    if not generation % self.history:
                        self.century_gen.append(score)
                    self.step(generation)
                    if verbose:
                        logging.debug(f"Generation n.: {generation}, fitness score: {score}")
                    if score >= max_generations.get("min_score", 100):
                        break
                    generation += 1

            else:
                for generation in range(1, max_generations+1):
                    if not generation % self.history:
                        self.century_gen.append(score)
                    self.step(generation)
                    score = self.pool.fitness(self.champion())
                    if verbose:
                        logging.debug(f"Generation n.: {generation}, fitness score: {score}")
        except KeyboardInterrupt:
            print("\nAborting!\n")
        except ValueError as e:
            print(f"Value error: {e.args[0]}")
