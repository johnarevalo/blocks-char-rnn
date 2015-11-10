import numpy
import codecs
import h5py
import yaml
from fuel.datasets import H5PYDataset
from config import config

# Load config parameters
locals().update(config)
numpy.random.seed(0)

with codecs.open(text_file, 'r', 'utf-8') as f:
    data = f.read().splitlines()

chars = list(set(''.join(data)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

inputs = []
outputs = []
for line in data:
    inputs.append([char_to_ix[ch] for ch in line[:-1]])
    outputs.append([char_to_ix[ch] for ch in line[1:]])

nsamples = len(outputs)
nsamples_train = int(nsamples * train_size)
nsamples_dev = nsamples - nsamples_train

f = h5py.File(hdf5_file, mode='w')
dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
features = f.create_dataset('features', (nsamples,), dtype=dtype)
targets = f.create_dataset('targets', (nsamples,), dtype=dtype)
targets.attrs['char_to_ix'] = yaml.dump(char_to_ix)
targets.attrs['ix_to_char'] = yaml.dump(ix_to_char)
features[...] = inputs
targets[...] = outputs

split_dict = {
    'train': {'features': (0, nsamples_train), 'targets': (0, nsamples_train)},
    'dev': {'features': (nsamples_train, nsamples), 'targets': (nsamples_train, nsamples)}}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()
