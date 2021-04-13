from models import *
from utils import *
from populations import *

n_samples = 250
act_selector = None
# filename = 'dataset_mnist.pkl'
filename = 'dataset_fmnist.pkl'
with open(filename, 'rb') as f:
        x_cal, y_cal, x_exp, y_exp, x_left, y_left = pickle.load(f)

# ---- Set "global" variables
# -- Genetic algorithm parameters
n_gen = 300
poptype = 'pixels'

if poptype == 'sources':
    probabilities = {'xy': 0.5, 'curr': 0.5, 'zero': 0.05} # 0.5 0.5
elif poptype == 'pixels':
    probabilities = {'mut': 0.1, 'zero': 0.01} # 0.1 0.01

n_best = 50
n_imm = 100

# set n_mut according to power-law
fk = zipf_distributed(n_best, 0.6) # 0.6
distribution = set_lower_to_one(fk)
n_mut = np.sum(distribution)

n_individuals = n_mut + n_best + n_imm


# -- Individual parameters
n_sources = 15
img_dim = 28
n_filters = 1


# -- Set models
# model_name = 'lindseyNet_1_3'
model_name = 'lindseyNet_fmnist_1_3'
model = load_model(model_name)

input_to_layer_model = create_input_to_layer_model(
    model=model,
    layer_list=[8]
)

layer_to_layer_model = create_layer_to_layer_model(
    model=model,
    layer1=4,
    layer2=8
)

layer_to_output_model = create_layer_to_output_model(
    model=model,
    layer=4
)

min_error_history = np.zeros((n_samples, n_gen))
out_error_history = np.zeros((n_samples, n_gen))
guessed_cat_history = np.zeros((n_samples, n_gen))

target_cat = np.zeros(n_samples)
guessed_cat = np.zeros(n_samples)
# ---- Simulation starts
for ss in range(n_samples):
    count = 0
    input = np.expand_dims(x_exp[ss, :, :], axis=0)
    target_act = input_to_layer_model.predict(input)
    target_out = model.predict(input)
    target_cat[ss] = int(np.argmax(target_out[0, :]))

    # ---
    print('Start evolution')
    if poptype == 'sources':
        pop = SourcePopulation(n_individuals, n_sources, img_dim)
    elif poptype == 'pixels':
        pop = PixelPopulation(n_individuals, img_dim, n_filters)
    pop.generate_random_population()
    pop.produce_errors_from_population(layer_to_layer_model, target_act, act_selector)

    min_error = np.min(pop.errors)
    best_ind = pop.extract_individuals(np.argmin(pop.errors))
    best_ind._data = np.expand_dims(best_ind.data, axis=0)
    if poptype == 'sources':
        images = best_ind.convert_to_pixels(n_filters)
        output = layer_to_output_model.predict(images)
    elif poptype == 'pixels':
        output = layer_to_output_model.predict(best_ind._data)
    temp_cat = np.argmax(output[0, :])
    out_error = np.sqrt(np.sum((output[0, :] - target_out[0, :])**2))

    min_error_history[ss, 0] = min_error
    out_error_history[ss, 0] = out_error
    guessed_cat_history[ss, 0] = temp_cat

    print('Sample', ss, 'generation ', 0, ' error: ', min_error)

    for i in range(1, n_gen):
        pop_best = pop.select_best_individuals(n_best)
        pop_mut = generate_population_for_mutation(pop_best, distribution, count)
        pop_mut = pop_mut.generate_mutated_population(probabilities)
        pop_mut.produce_errors_from_population(layer_to_layer_model, target_act, act_selector)

        pop_imm = pop.copy_template(n_imm)
        pop_imm.generate_random_population()
        pop_imm.produce_errors_from_population(layer_to_layer_model, target_act, act_selector)

        pop = join_populations([pop_best, pop_mut, pop_imm])
        min_error = np.min(pop.errors)
        best_ind = pop.extract_individuals(np.argmin(pop.errors))
        best_ind._data = np.expand_dims(best_ind.data, axis=0)
        if poptype == 'sources':
            images = best_ind.convert_to_pixels(n_filters)
            output = layer_to_output_model.predict(images)
        elif poptype == 'pixels':
            output = layer_to_output_model.predict(best_ind._data)
        temp_cat = np.argmax(output[0, :])
        out_error = np.sqrt(np.sum((output[0, :] - target_out[0, :])**2))

        min_error_history[ss, i] = min_error
        out_error_history[ss, i] = out_error
        guessed_cat_history[ss, i] = temp_cat

        if not(i % 25):
            print('Sample', ss, 'generation ', i, ' error: ', min_error)

    guessed_cat[ss] = temp_cat
    print(target_cat[ss])
    print(guessed_cat[ss])

history = {
    'target_cat': target_cat,
    'min_error': min_error_history,
    'out_error': out_error_history,
    'guessed_cat': guessed_cat_history
}

filename = 'draft01_static_landscape_' + poptype + '.pkl'
with open(filename, 'wb') as f:
    pickle.dump(history, f)
