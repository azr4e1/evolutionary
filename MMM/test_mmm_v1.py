import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from evolutionary import Evolution
from mmm_utils import *
import time


# TODO
# run a regression first and then use the output as initial parameters for evol.

df = pd.read_csv("https://raw.githubusercontent.com/flyandlure/datasets/master/marketing_mix.csv")
# reorder the columns
cols = set(df.columns)
df2 = df.loc[:, ["week", "sales"] + list(cols.difference(set(["week", "sales"])))]

X = df2.iloc[:, 2:]
y = df2["sales"]

scaler = StandardScaler()
scaler.fit(X)
X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

n_channels = len(X_scaled.columns)

evo1 = Evolution(
    pool_size=15, fitness=fitness, individual_class=ChildrenOfBodom, n_offsprings=5,
    pair_params={'alpha': 0.5},
    mutate_params={'lower_bounds': [0, 1], 'upper_bounds': [1, 1000], 'rate': [0.25, 100], 'n_channels': n_channels},
    init_params={'lower_bounds': [0, 1], 'upper_bounds': [1, 1000], 'n_channels': n_channels, 'data': X_scaled, 'target': y}
)
time_start = time.time()
evo1({"min_score": 0.7}, verbose=True)
time_end = time.time()
print(f"\nIt took {time_end - time_start:.2f} seconds")

init_value = evo1.champion().value

input("Press any key to continue to second stage: ")

evo2 = Evolution(
    pool_size=15, fitness=fitness_solver, individual_class=Solver, n_offsprings=5,
    pair_params={'alpha': 0.5},
    mutate_params={'lower_bounds': [0, 1, 10, 0], 'upper_bounds': upper_bounds, 'rate': 0.001 * upper_bounds, 'n_channels': n_channels},
    init_params={'n_channels': n_channels, 'data': X_scaled, 'target': y}
)

time_start = time.time()
evo({"min_score": 0.9}, verbose=True)
time_end = time.time()

champion = evo.champion()
decay = champion.value[0]
slope = champion.value[1]
beta = champion.value[2]
base = champion.value[3, 0]


X_transformed = transformation(X_scaled, decay, slope)
prediction = base + np.dot(X_transformed, beta)
values = pd.DataFrame(np.vstack([champion.value]), index = ["decay", "slope", "coefficients", "base"], columns=X_scaled.columns)
print("Hyperparameters:")
print(values)
print(f"\nIt took {time_end - time_start:.2f} seconds")
