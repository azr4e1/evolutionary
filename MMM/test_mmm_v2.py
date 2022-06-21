import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from evolutionary import Evolution
from mmm_utils import *

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
evo = Evolution(
    pool_size=10, fitness=fitness, individual_class=ChildrenOfBodom, n_offsprings=3,
    pair_params={'alpha': 0.5},
    mutate_params={'lower_bounds': [0, 0], 'upper_bounds': [1, 1000], 'rate': 0.25, 'n_channels': n_channels},
    init_params={'lower_bounds': [0, 0], 'upper_bounds': [1, 1000], 'n_channels': n_channels, 'data': X_scaled, 'target': y}
)

n_epochs = 100
evo(n_epochs, verbose=True)

champion = evo.champion()
decay = champion.value[0]
slope = champion.value[1]


X_transformed = transformation(X_scaled, decay, slope)
reg = LinearRegression().fit(X_transformed, y)
values = pd.DataFrame(np.vstack([champion.value, reg.coef_]), index = ["decay", "slope", "coefficients"], columns=X_scaled.columns)
print("Hyperparameters:")
print(values)
print("intercept:", reg.intercept_)