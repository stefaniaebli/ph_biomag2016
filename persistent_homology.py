import sys 
import numpy as np
sys.path.append("/home/stefania/local/src/Dionysus-453eedc14be0/build/bindings/python")
from dionysus import Rips,PairwiseDistances,Simplex, Filtration, StaticPersistence,\
                     vertex_cmp, data_cmp, dim_data_cmp, fill_alpha3D_complex, init_diagrams, \
                     fill_alpha_complex, fill_alpha2D_complex,points_file
from joblib import Parallel, delayed


class MyDistances:
	def __init__(self, adjacency_matrix, norm=None):
		self.adjacency_matrix = adjacency_matrix

	def __len__(self):
		return len(self.adjacency_matrix)

	def __call__(self, i1, i2):
		return self.adjacency_matrix[i1][i2]


def ph_one_matrix(matrix, order_max=0, radius_max=None, bins=10):
	myd = MyDistances(matrix)
	rips = Rips(myd)
	f = Filtration()
	rips.generate(order_max + 1, radius_max, f.append)  #  We need to compute Rips complexes till order_max+1
	f.sort(rips.cmp)
	p = StaticPersistence(f)
	p.pair_simplices()
	dgms = init_diagrams(p, f, rips.eval)
	ph_matrix = []
	features_matrix = []
	for order, dgm in enumerate(dgms):
		if order == order_max + 1:
			break

		birth = np.array([b for b, d, in dgm])
		death = np.array([d for b, d, in dgm])
		ph_matrix.append([order, birth, death])
		h_birth = np.histogram(birth, bins=bins, range=[0,radius_max])[0]
		h_death = np.histogram(death, bins=bins, range=[0,radius_max])[0]
		h_life = np.histogram(death - birth, bins=bins, range=[0,radius_max])[0]
		h_b_u_d = np.histogram(np.concatenate([death, birth]), bins=bins, range=[0,radius_max])[0]
		features_matrix.append([order, h_birth, h_death, h_life, h_b_u_d])

	return ph_matrix, features_matrix


def persistent_homology(matrices, order_max=0, radius_max=None, bins=10):
	if radius_max is None:
		radius_max = np.median(matrices)  # np.max(matrices)

	ph = []
	features = []
	for matrix in matrices:
		ph_matrix, features_matrix = ph_one_matrix(matrix, order_max=order_max, radius_max=radius_max, bins=bins)
		ph.append(ph_matrix)
		features.append(features_matrix)

	return ph, features


def persistent_homology_parallel(matrices, order_max=0, radius_max=None, bins=10):
	if radius_max is None:
		radius_max = np.median(matrices)  # np.max(matrices)

	ph = []
	features = []
	results = Parallel(n_jobs=-1)(delayed(ph_one_matrix)(matrix, order_max=order_max, radius_max=radius_max, bins=bins) for matrix in matrices)
	ph = [ph for ph, features in results]
	features = [features for ph, features in results]
	return ph, features


if __name__ == '__main__':
	np.random.seed(0)
	order_max = 1
	bins = 20
	matrices = np.random.uniform(size=(10, 204, 204))
	matrices = [(matrix + matrix.T) / 2.0 for matrix in matrices]
	ph, features = persistent_homology_parallel(matrices, order_max=order_max, bins=bins)