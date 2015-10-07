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
    data = f.read()

if len(data) % seq_length > 0:
    data = data[:len(data) - len(data) % seq_length + 1]
else:
    data = data[:len(data) - seq_length + 1]

nsamples = len(data) // seq_length
chars = list(set(data))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
inputs = numpy.empty((nsamples, seq_length), dtype='uint8')
outputs = numpy.zeros_like(inputs)
for i, p in enumerate(range(0, len(data) - 1, seq_length)):
    inputs[i] = numpy.array([char_to_ix[ch] for ch in data[p:p + seq_length]])
    outputs[i] = numpy.array([char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]])

f = h5py.File(hdf5_file, mode='w')
features = f.create_dataset('features', inputs.shape, dtype='uint8')
targets = f.create_dataset('targets', outputs.shape, dtype='uint8')
targets.attrs['char_to_ix'] = yaml.dump(char_to_ix)
targets.attrs['ix_to_char'] = yaml.dump(ix_to_char)
features[...] = inputs
targets[...] = outputs
features.dims[0].label = 'batch'
features.dims[1].label = 'sequence'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'sequence'

nsamples_train = int(nsamples * train_size)
split_dict = {
    'train': {'features': (0, nsamples_train), 'targets': (0, nsamples_train)},
    'dev': {'features': (nsamples_train, nsamples), 'targets': (nsamples_train, nsamples)}}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()
