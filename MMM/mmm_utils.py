import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from evolutionary import BaseIndividual

def adstock(df: pd.DataFrame, decay):
    df_adstock = df.copy()
    df_adstock += df_adstock.shift(periods=1, axis=0, fill_value=0) * decay

    return df_adstock

def exp_decay(df:pd.DataFrame, slope):
    df_decay = df.copy()
    df_decay /= slope
    df_decay = 1 - np.exp(df_decay)

    return df_decay

def transformation(df: pd.DataFrame, decay, slope):
    df_adstock = adstock(df, decay)
    df_decay = exp_decay(df_adstock, slope)

    return df_decay


def fitness(individual):
    data = individual.data
    target = individual.target
    decay = individual.value[0,:]
    slope = individual.value[1,:]

    X = transformation(data, decay, slope)
    reg = LinearRegression().fit(X, target)

    return reg.score(X, target)


class ChildrenOfBodom(BaseIndividual):

    def pair(self, other, pair_params):
        child = ChildrenOfBodom({"value": pair_params["alpha"] * self.value + (1 - pair_params["alpha"]) * other.value})
        # child's inheritance
        child.data = self.data
        child.target = self.target

        return child

    def mutate(self, mutate_params):
        self.value += np.random.normal(0, mutate_params['rate'], (2, mutate_params['n_channels'])).T

        xx, yy = np.meshgrid(range(2), range(mutate_params["n_channels"]))
        for i in zip(xx.flatten(), yy.flatten()):
            if self.value[i] < mutate_params['lower_bounds'][i[0]]:
                self.value[i] = mutate_params['lower_bounds'][i[0]]
            elif self.value[i] > mutate_params['upper_bounds'][i[0]]:
                self.value[i] = mutate_params['upper_bounds'][i[0]]

    def _random_init(self, init_params):
        self.data = init_params["data"]
        self.target = init_params["target"]

        stack = []
        for i in range(init_params["n_channels"]):
            stack.append(np.random.uniform(init_params["lower_bounds"], init_params["upper_bounds"]).reshape(2,1))

        values = np.hstack(stack)

        return values


def fitness_solver(individual):
    data = individual.data
    target = individual.target
    decay = individual.value[0]
    slope = individual.value[1]
    beta = individual.value[2]
    base = individual.value[3, 0]

    X = transformation(data, decay, slope)
    y = base + np.dot(X, beta)
    score = r2_score(target, y)

    return score


class Solver(BaseIndividual):

    def pair(self, other, pair_params):
        child = Solver({"value": pair_params["alpha"] * self.value + (1 - pair_params["alpha"]) * other.value})
        # child's inheritance
        child.data = self.data
        child.target = self.target

        return child

    def mutate(self, mutate_params):
        mutation = np.random.normal(0, mutate_params['rate'], (mutate_params['n_channels'], 4)).T
        mutation[3] = mutation[3, 0]
        self.value += mutation

        xx, yy = np.meshgrid(range(4), range(mutate_params["n_channels"]))
        for i in zip(xx.flatten(), yy.flatten()):
            if self.value[i] < mutate_params['lower_bounds'][i[0]]:
                self.value[i] = mutate_params['lower_bounds'][i[0]]
            elif self.value[i] > mutate_params['upper_bounds'][i[0]]:
                self.value[i] = mutate_params['upper_bounds'][i[0]]

    def _random_init(self, init_params):
        self.data = init_params["data"]
        self.target = init_params["target"]

        stack = []
        for i in range(init_params["n_channels"]):
            stack.append(np.random.uniform(init_params["lower_bounds"], init_params["upper_bounds"]).reshape(4,1))

        values = np.hstack(stack)
        values[3] = values[3,0]

        return values
