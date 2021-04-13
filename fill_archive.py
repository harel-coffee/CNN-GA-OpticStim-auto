from models import *
from utils import *
from populations import *


n_samples = 250 # 250
act_selector = None

filename = 'dataset_mnist.pkl'
with open(filename, 'rb') as f:
        x_cal, y_cal, x_exp, y_exp, x_left, y_left = pickle.load(f)

# ---- Set "global" variables
# -- Genetic algorithm parameters
n_gen = 200
poptype = 'sources'

if poptype == 'sources':
    probabilities = {'xy': 0.5, 'curr': 0.5, 'zero': 0.05}
elif poptype == 'pixels':
    probabilities = {'mut': 0.4, 'zero': 0.01} # 0.1 0.01

n_best = 50
n_imm = 100
n_cats = 10
# set n_mut according to power-law
fk = zipf_distributed(n_best, 0.6)
distribution = set_lower_to_one(fk)
n_mut = np.sum(distribution)

n_individuals = n_mut + n_best + n_imm


# -- Individual parameters
n_sources = 15
img_dim = 28
n_filters = 1


# -- Set models
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

layer_to_output_model = create_layer_to_output_model(
    model=model,
    layer=4
)
data = []
errors = []
activations = []
categories = []

cats = np.array([int(np.where(el==1)[0][0]) for el in y_left])

# ---- Simulation starts
n_samples_per_cat = 25
for ss in range(n_cats):
    n_memo = 0
    count = 0
    idx = np.arange(x_left.shape[0])
    idx = idx[np.where(cats == ss)]
    x_cat = x_left[idx, :, :]
    while (n_memo < n_samples_per_cat):
        input = np.expand_dims(x_cat[count, :, :], axis=0)
        target_act = input_to_layer_model.predict(input)

        # ---
        print('\nStart evolution')
        if poptype == 'sources':
            pop = SourcePopulation(n_individuals, n_sources, img_dim)
        elif poptype == 'pixels':
            pop = PixelPopulation(n_individuals, img_dim, n_filters)
        pop.generate_random_population()
        pop.produce_errors_from_population(layer_to_layer_model, target_act, act_selector)

        min_error = np.min(pop.errors)

        print('Sample', count, 'generation ', 0, ' error: ', min_error)

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

            if not(i % 25):
                print('Sample', count, 'generation ', i, ' error: ', min_error)

        best_ind = pop.extract_individuals(np.argmin(pop.errors))
        best_ind._data = np.expand_dims(best_ind.data, axis=0)
        act = best_ind.produce_activations_from_population(layer_to_layer_model)
        if poptype == 'sources':
            images = best_ind.convert_to_pixels(n_filters)
            output = layer_to_output_model.predict(images)
        elif poptype == 'pixels':
            output = layer_to_output_model.predict(best_ind._data)
        guessed_cat = np.argmax(output[0, :])

        if guessed_cat == ss:
            print('Class match, pushing individual to memory...')
            print('Individual ' + str(n_memo + 1) + ' of ' + str(n_samples_per_cat) + ' for class ' + str(ss))
            n_memo += 1
            data.append(best_ind._data)
            errors.append(min_error)
            act = best_ind.produce_activations_from_population(layer_to_layer_model)
            activations.append(act)
            categories.append(ss)

        count += 1

    archived_pop = SourcePopulation(n_individuals=None, n_sources=n_sources, img_dim=img_dim)
    archived_pop.set_data(np.concatenate(data, axis=0))
    archived_pop.set_errors(np.array(errors))
    archived_pop.set_activations(np.concatenate(activations, axis=0))
    archived_pop.set_categories(np.array(categories))

filename = 'archived_' + poptype + '.pkl'
with open(filename, 'wb') as f:
    pickle.dump(archived_pop, f)
