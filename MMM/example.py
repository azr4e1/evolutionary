from evolutionary import Evolution, Individual


def fitness(opt):
    return -opt.value[0] * (opt.value[0] - 1) * (opt.value[0] - 2) * (opt.value[0] - 3) * (opt.value[0] - 4)


evo = Evolution(
    pool_size=10, fitness=fitness, individual_class=Individual, n_offsprings=3,
    pair_params={'alpha': 0.5},
    mutate_params={'lower_bound': 0, 'upper_bound': 4, 'rate': 0.25, 'dim': 1},
    init_params={'lower_bound': 0, 'upper_bound': 4, 'dim': 1},
)
n_epochs = 50

evo(n_epochs, verbose=True)

print(evo.champion().value)