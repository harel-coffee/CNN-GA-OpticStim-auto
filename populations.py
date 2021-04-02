import numpy as np
from sklearn.metrics import pairwise_distances
import random

def produce_swap_list(p_mix, radius, shape):
    """Produce swap list for retinotopy."""
    x_mix = []
    y_mix = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            if random.random() < p_mix:
                x_new = np.clip(i + random.randint(1, radius), 0, shape[0] - 1)
                y_new = np.clip(j + random.randint(1, radius), 0, shape[1] - 1)
                x_mix.append([i, x_new])
                y_mix.append([j, y_new])
    return [x_mix, y_mix]

def mix_image(images, x_mix, y_mix):
    """Mix image given swap list."""
    new_images = images.copy()
    for j in range(images.shape[0]):
        for i in range(len(x_mix)):
            aux = new_images[j, x_mix[i][0], y_mix[i][0], 0]  # store old value
            new_images[j, x_mix[i][0], y_mix[i][0], 0] = new_images[j, x_mix[i][1], y_mix[i][1], 0]
            new_images[j, x_mix[i][1], y_mix[i][1], 0] = aux
    return new_images

def zipf_distributed(N, s):
    k = np.arange(1, N + 1)
    fk = k**(-float(s))
    F = np.sum(fk)
    fk = fk/F
    return fk

def set_lower_to_one(fk):
    alpha = 1/(fk[-1] - 1e-6)
    array = (alpha * fk).astype(int)
    return array

def join_populations(populations):
    """Join population in a sequence."""
    # TODO: here we assume that the user is careful...
    # TODO: this works only for single-parent mutations, otherwise you have a
    # list of parents and you cannot use np.concatenate
    data = [pop.data for pop in populations]
    data = np.concatenate(data, axis=0)
    new_pop = populations[0].copy()
    new_pop.set_data(data)

    if populations[0].errors is not None:
        errors = [pop.errors for pop in populations]
        errors = np.concatenate(errors, axis=0)
        new_pop.set_errors(errors)

    if populations[0].identity is not None:
        identity = [pop.identity for pop in populations]
        parent_identity = [pop.parent_identity for pop in populations]
        identity = np.concatenate(identity, axis=0)
        parent_identity = np.concatenate(parent_identity, axis=0)
        new_pop.set_identity(identity)
        new_pop.set_parent_identity(parent_identity)
    return new_pop

def generate_population_for_mutation(population, distribution, count):
    pop = population.copy_template()
    ind = []
    for i in range(distribution.shape[0]):
        shape = tuple([distribution[i]] + [1 for i in range(population.data.ndim - 1)])
        ind.append(np.tile(population.extract_individuals([i]).data, shape))
    data = np.concatenate(ind, axis=0)
    pop.set_data(data)
    identity = np.arange(count, count + pop.n_individuals)
    parent_identity = np.array([x for x in range(distribution.shape[0]) for i in range(distribution[x])])
    pop.set_identity(identity)
    pop.set_parent_identity(parent_identity)
    return pop

class Population:
    def __init__(self, n_individuals, img_dim, data=None, errors=None,
                 activations=None, identity=None, parent_identity=None, categories=None):
        self._n_individuals = n_individuals
        self._img_dim = img_dim
        self._data = data
        self._errors = errors
        self._activations = activations
        self._identity = identity
        self._parent_identity = parent_identity
        self._categories = categories

    @property
    def img_dim(self):
        return self._img_dim

    @property
    def data(self):
        return self._data

    @property
    def n_individuals(self):
        if self.data is not None:
            self._n_individuals = self.data.shape[0]
        return self._n_individuals

    @property
    def errors(self):
        return self._errors

    @property
    def activations(self):
        return self._activations

    @property
    def identity(self):
        return self._identity

    @property
    def parent_identity(self):
        return self._parent_identity

    @property
    def categories(self):
        return self._categories

    @property
    def shape(self):
        return self.data.shape

    def set_data(self, data):
        self._data = data

    def set_errors(self, errors):
        self._errors = errors

    def set_activations(self, activations):
        self._activations = activations

    def set_identity(self, identity):
        self._identity = identity

    def set_parent_identity(self, parent_identity):
        self._parent_identity = parent_identity

    def set_categories(self, categories):
        self._categories = categories

    def initialize_identity(self, last_id):
        self._identity = np.arange(last_id, last_id + self.n_individuals)
        self._parent_identity = np.empty(self.n_individuals) * np.nan

    def select_best_individuals(self, n_best):
        idx = np.argsort(self.errors)
        idx = idx[:n_best]
        pop = self.extract_individuals(idx)
        return pop

    def reorder_population(self):
        if self.errors is not None:
            idx = np.argsort(self.errors)
            pop = self.extract_individuals(idx)
        return pop

    def copy_template(self):
        pass

    def extract_individuals(self, idx):
        """Create another population with selected individuals form the current population."""
        indices = {0: idx}
        ix = tuple([indices.get(dim, slice(None)) for dim in range(self.data.ndim)])
        extracted_data = self.data[ix]
        extracted_errors = self.errors[idx] if self.errors is not None else self.errors
        extracted_identity = self.identity[idx] if self.identity is not None else self.identity
        extracted_parent_identity = self.parent_identity[idx] if self.parent_identity is not None else self.parent_identity

        pop = self.copy_template()

        pop.set_data(extracted_data)
        pop.set_errors(extracted_errors)
        pop.set_identity(extracted_identity)
        pop.set_parent_identity(extracted_parent_identity)
        return pop

    def substitute_individuals(self, pop2, idx1, idx2):
        """Substitutes individual from another population to the current population."""
        # NB. Assumes care from the user...
        indices1 = {0: idx1}
        indices2 = {0: idx2}
        ix1 = [indices1.get(dim, slice(None)) for dim in range(self.data.ndim)]
        ix2 = [indices2.get(dim, slice(None)) for dim in range(pop2.data.ndim)]
        self._data[ix1] = pop2.data[ix2]
        self._errors[idx1] = pop2.errors[idx2]
        if self.activations is not None:
            self._activations[idx1] = pop2.activations[idx2]
        if self.identity is not None:
            self._identity[idx1] = pop2.identity[idx2]
            self._parent_identity[idx1] = pop2.parent_identity[idx2]


class SourcePopulation(Population):
    def __init__(self, n_individuals, n_sources, img_dim, data=None, errors=None, activations=None,
                 identity=None, parent_identity=None, categories=None):
        super().__init__(n_individuals, img_dim, data, errors, activations, identity,
                         parent_identity, categories)
        self._n_sources = n_sources

    @property
    def n_sources(self):
        if self.data is not None:
            self._n_sources = self.data.shape[1]
        return self._n_sources

    def generate_random_population(self, init_identity=False, last_id=0):
        """Generate random data for the current population."""
        xy_locations = np.random.uniform(low=0, high=self.img_dim, size=(self.n_individuals, self.n_sources, 2))
        i_stim = np.random.uniform(low=-3, high=3, size=(self.n_individuals, self.n_sources, 1))
        self._data = np.concatenate((xy_locations, i_stim), axis=2)
        if init_identity:
            self.initialize_identity(last_id)

    def copy(self):
        """Copy current population into another population."""
        pop = SourcePopulation(self.n_individuals, self.n_sources, self.img_dim,
                               self.data, self.errors, self.activations, self.identity,
                               self.parent_identity)
        return pop

    def copy_template(self, n_individuals=None):
        pop = SourcePopulation(n_individuals, self.n_sources, self.img_dim)
        return pop

    def generate_mutated_population(self, probabilities):
        """Produce a mutated population from a given population."""
        p_xy = probabilities['xy']
        p_curr = probabilities['curr']
        p_zero = probabilities['zero']

        pop = self.copy()

        N = pop.n_individuals * pop.n_sources
        n_xy_els = int(p_xy * N)
        n_curr_els = int(p_curr * N)
        n_zero_els = int(p_zero * N)

        linear_indices = np.random.randint(low=0, high=N, size=n_xy_els).astype(int)
        coords_mut = np.unravel_index(linear_indices, (pop.n_individuals, pop.n_sources))
        coords_x = coords_mut + (0 * np.ones(n_xy_els, dtype=np.int64),)
        coords_y = coords_mut + (1 * np.ones(n_xy_els, dtype=np.int64),)
        pop._data[coords_x] += np.random.uniform(low=-0.5, high=0.5, size=n_xy_els)
        pop._data[coords_y] += np.random.uniform(low=-0.5, high=0.5, size=n_xy_els)

        linear_indices = np.random.randint(low=0, high=N, size=n_curr_els).astype(int)
        coords_mut = np.unravel_index(linear_indices, (pop.n_individuals, pop.n_sources))
        coords_mut = coords_mut + (2 * np.ones(n_curr_els, dtype=np.int64),)
        pop._data[coords_mut] += np.random.uniform(low=-0.25, high=0.25, size=n_curr_els)

        linear_indices = np.random.randint(low=0, high=N, size=n_zero_els).astype(int)
        coords_mut = np.unravel_index(linear_indices, (pop.n_individuals, pop.n_sources))
        coords_mut = coords_mut + (2 * np.ones(n_zero_els, dtype=np.int64),)
        pop._data[coords_mut] = np.zeros(n_zero_els)
        return pop

    def apply_crossover(self, probabilities):
        raise NotImplemented

    def convert_to_pixels(self, n_filters):
        pixel_locs = np.mgrid[0:self.img_dim, 0:self.img_dim]
        images = np.zeros((self.n_individuals, self.img_dim, self.img_dim, 1))
        for i in range(self.n_individuals):
            for j in range(self.n_sources):
                d = np.sqrt(
                    + (pixel_locs[0] - self.data[i, j, 0])**2 +
                    + (pixel_locs[1] - self.data[i, j, 1])**2 +
                    0.01  # avoid dividing by zero
                )
                images[i, :, :, 0] += self.data[i, j, 2] / d
        # images = custom_logit(images)
        images = np.tanh(images)
        images = np.tile(images, (1, 1, 1, n_filters))
        return images

    def produce_errors_from_population(self, layer_to_layer_model, target, sel=None, store=True, snr=None, swap_list=None):
        input_shape = layer_to_layer_model.input_shape
        n_filters = input_shape[3]
        images = self.convert_to_pixels(n_filters)
        if swap_list is not None:
            images = mix_image(images, swap_list[0], swap_list[1])
        activations = layer_to_layer_model.predict(images)
        actshape = activations.shape
        selfact = np.reshape(activations, (actshape[0], actshape[1]*actshape[2]*actshape[3]))
        if snr is not None:
            var = np.var(selfact) / snr
            selfact += np.random.randn(actshape[0], actshape[1]*actshape[2]*actshape[3]) * var
        targact = np.reshape(target, (1, actshape[1]*actshape[2]*actshape[3]))
        if sel is not None:
            selfact = selfact[:, sel]
            targact = targact[:, sel]
        errors = np.squeeze(pairwise_distances(targact, selfact, n_jobs=-1))
        if store:
            self._errors = errors
        return errors

    def produce_activations_from_population(self, layer_to_layer_model, store=False):
        input_shape = layer_to_layer_model.input_shape
        n_filters = input_shape[3]
        images = self.convert_to_pixels(n_filters)
        activations = layer_to_layer_model.predict(images)
        if store:
            self._activations = activations
        return activations


class PixelPopulation(Population):
    def __init__(self, n_individuals, img_dim, n_filters, data=None,
                 errors=None, activations=None, identity=None, parent_identity=None, categories=None):
        super().__init__(n_individuals, img_dim, data, errors, activations, identity,
                         parent_identity, categories)
        self._n_filters = n_filters

    @property
    def n_filters(self):
        if self.data is not None:
            self._n_filters = self.data.shape[3]
        return self._n_filters

    def generate_random_population(self, init_identity=False, last_id=0):
        """Generate random data for the current population."""
        self._data = np.random.uniform(low=-1, high=1, size=(self.n_individuals, self.img_dim, self.img_dim, self.n_filters))
        if init_identity:
            self.initialize_identity(last_id)

    def copy(self):
        """Copy current population into another population."""
        pop = PixelPopulation(self.n_individuals, self.img_dim, self.n_filters,
                               self.data, self.errors, self.activations, self.identity,
                               self.parent_identity)
        return pop

    def copy_template(self, n_individuals=None):
        pop = PixelPopulation(n_individuals, self.img_dim, self.n_filters)
        return pop

    def generate_mutated_population(self, probabilities):
        """Produce a mutated population from a given population."""
        p_mut = probabilities['mut']
        p_zero = probabilities['zero']

        pop = self.copy()

        N = pop.shape[0] * pop.shape[1] * pop.shape[2] * pop.shape[3]
        n_mut_els = int(p_mut * N)
        n_zero_els = int(p_zero * N)

        linear_indices = np.random.randint(low=0, high=N, size=n_mut_els).astype(int)
        coords_mut = np.unravel_index(linear_indices, pop.shape)
        pop._data[coords_mut] += np.random.uniform(low=-0.10, high=0.10, size=n_mut_els)

        linear_indices = np.random.randint(low=0, high=N, size=n_zero_els).astype(int)
        coords_zero = np.unravel_index(linear_indices, pop.shape)
        pop._data[coords_zero] = np.zeros(n_zero_els)
        return pop

    def produce_errors_from_population(self, layer_to_layer_model, target, sel=None, store=True):
        activations = layer_to_layer_model.predict(self.data)
        errors = np.zeros(self.n_individuals)
        actshape = activations.shape
        selfact = np.reshape(activations, (actshape[0], actshape[1]*actshape[2]*actshape[3]))
        targact = np.reshape(target, (1, actshape[1]*actshape[2]*actshape[3]))
        if sel is not None:
            selfact = selfact[:, sel]
            targact = targact[:, sel]
        errors = np.squeeze(pairwise_distances(targact, selfact, n_jobs=-1))
        if store:
            self._errors = errors
        return errors

    def produce_activations_from_population(self, layer_to_layer_model, store=False):
        activations = layer_to_layer_model.predict(self.data)
        if store:
            self._activations = activations
        return activations
