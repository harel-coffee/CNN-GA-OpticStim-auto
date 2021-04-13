from populations import *
from models import *
from utils import *

import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D


n_samples = 250 # 250
act_selector = None

filename = 'dataset_mnist.pkl'
with open(filename, 'rb') as f:
        x_cal, y_cal, x_exp, y_exp, x_left, y_left = pickle.load(f)

# ---- Set parameters
n_best = 100
n_imm = 50
n_gen = 200
fk = zipf_distributed(n_best, 0.6)
distribution = set_lower_to_one(fk)
n_mut = np.sum(distribution)
n_individuals = n_mut + n_best + n_imm
n_sources = 15
img_dim = 28
n_filters = 1
count = 0
poptype = 'sources'
if poptype == 'sources':
    probabilities = {'xy': 0.5, 'curr': 0.5, 'zero': 0.05}
elif poptype == 'pixels':
    probabilities = {'mut': 0.1, 'zero': 0.01}

# ---- Set models
(x_train, y_train), (x_test, y_test) = load_mnist()
model_name = 'lindseyNet_1_3'
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
layer_to_output_model = create_layer_to_output_model(model, layer=4)
layer_to_output_model.summary()

min_error_history = np.zeros((n_samples, n_gen))
out_error_history = np.zeros((n_samples, n_gen))
guessed_cat_history = np.zeros((n_samples, n_gen))

# ---
T = 25
target_cat = np.zeros(n_samples)
guessed_cat = np.zeros(n_samples)
for ss in range(n_samples):
    count = 0
    input = np.expand_dims(x_exp[ss, :, :], axis=0)
    target_act = input_to_layer_model.predict(input)
    target_out = model.predict(input)
    target_cat[ss] = int(np.argmax(target_out[0, :]))

    acc_disp, target_list, target_out_list = generate_shifted_data(
        input_image=input,
        input_to_layer_model=input_to_layer_model,
        model=model
    )

    print('Start evolution, cat', target_cat[ss])
    if poptype == 'sources':
        pop = SourcePopulation(n_individuals, n_sources, img_dim)
    elif poptype == 'pixels':
        pop = PixelPopulation(n_individuals, img_dim, n_filters)
    pop.generate_random_population()
    pop.produce_errors_from_population(layer_to_layer_model, target_act, act_selector)
    count += n_individuals

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

    print('Sample', ss, 'generation', 0, 'error', min_error, 'class', temp_cat)

    for i in range(1, n_gen):
        if not(i % T):
            sel = np.random.randint(low=0, high=len(target_list))
            target_act = target_list[sel]
            target_out = target_out_list[sel]
            pop.produce_errors_from_population(layer_to_layer_model, target_act, act_selector)

        pop_best = pop.select_best_individuals(n_best)
        pop_mut = generate_population_for_mutation(pop_best, distribution, count)
        pop_mut = pop_mut.generate_mutated_population(probabilities)
        pop_mut.produce_errors_from_population(layer_to_layer_model, target_act, act_selector)
        count += n_mut

        pop_imm = pop.copy_template(n_imm)
        pop_imm.generate_random_population()
        pop_imm.produce_errors_from_population(layer_to_layer_model, target_act, act_selector)
        count += n_imm

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
            print('Sample', ss, 'generation', i, 'error', min_error, 'class', temp_cat)
    guessed_cat[ss] = temp_cat
    print(target_cat[ss])
    print(guessed_cat[ss])

history = {
    'target_cat': target_cat,
    'min_error': min_error_history,
    'out_error': out_error_history,
    'guessed_cat': guessed_cat_history
}

filename = 'draft01_smooth_evolution_' + str(T) + '.pkl'
with open(filename, 'wb') as f:
    pickle.dump(history, f)
