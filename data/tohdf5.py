# Convert pickled data in mnist.pkl to hdf5 structure
import h5py
import cPickle
import numpy as np
f = open('./mnist.pkl', 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()

outfile = h5py.File('mnist.hdf5','w')

def save_data(name, src):
	g = outfile.create_group(name)
	g['images'] = src[0].astype(np.float64)
	g['labels'] = src[1].astype(np.float64)

save_data("training_data", training_data)
save_data("validation_data", validation_data)
save_data("test_data", test_data)

# note, when reshaping image data in julia to be 8x8, need to transpose