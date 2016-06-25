import numpy
import theano
from theano import tensor
from blocks import roles
from blocks.model import Model
from blocks.serialization import load_parameters
from blocks.filter import VariableFilter
from utils import get_metadata, MainLoop
from config import config
from model import nn_fprop
import argparse
import sys


def sample(x_curr, states_values, fprop, temperature=1.0):
    '''
    Propagate x_curr sequence and sample next element according to
    temperature sampling.
    Return: sampled element and a list of the hidden activations produced by fprop.
    '''
    activations = fprop(x_curr, *states_values)
    probs = activations.pop().astype('float64')
    if numpy.random.binomial(1, temperature) == 1:
        probs = probs / probs.sum()
        sample = numpy.random.multinomial(1, probs).nonzero()[0][0]
    else:
        sample = probs.argmax()

    return sample, activations

if __name__ == '__main__':
    # Load config parameters
    locals().update(config)

    parser = argparse.ArgumentParser(
        description='Sample from a character-level language model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', default=save_path,
                        help='model checkpoint to use for sampling')
    parser.add_argument('-primetext', default=None,
                        help='used as a prompt to "seed" the state of the RNN using a given sequence, before we sample.')
    parser.add_argument('-length', default=1000,
                        type=int, help='number of characters to sample')
    parser.add_argument('-seed', default=None,
                        type=int, help='seed for random number generator')
    parser.add_argument('-temperature', type=float,
                        default=1.0, help='temperature of sampling')
    args = parser.parse_args()

    # Define primetext
    ix_to_char, char_to_ix, vocab_size = get_metadata(hdf5_file)
    if not args.primetext or len(args.primetext) == 0:
        args.primetext = ix_to_char[numpy.random.randint(vocab_size)]
    primetext = ''.join([ch for ch in args.primetext if ch in char_to_ix.keys()])
    if len(primetext) == 0:
        raise Exception('primetext characters are not in the vocabulary')
    x_curr = numpy.expand_dims(
        numpy.array([char_to_ix[ch] for ch in primetext], dtype='uint8'), axis=1)

    print('Loading model from {0}...'.format(args.model))
    x = tensor.matrix('features', dtype='uint8')
    y = tensor.matrix('targets', dtype='uint8')
    y_hat, cost, cells = nn_fprop(x, y, vocab_size, hidden_size, num_layers, model)
    main_loop = MainLoop(algorithm=None, data_stream=None, model=Model(cost))
    with open(args.model) as f:
        main_loop.model.set_parameter_values(load_parameters(f))
    bin_model = main_loop.model
    activations = []
    initial_states = []
    for i in range(num_layers):
        brick = [b for b in bin_model.get_top_bricks() if b.name==model+str(i)][0]
        activations.extend(VariableFilter(theano_name=brick.name+'_apply_states')(bin_model.variables))
        activations.extend(VariableFilter(theano_name=brick.name+'_apply_cells')(cells))
        initial_states.extend(VariableFilter(roles=[roles.INITIAL_STATE])(brick.parameters))

    #take activations of last element
    activations = [act[-1].flatten() for act in activations]
    states_as_params = [tensor.vector(dtype=initial.dtype) for initial in initial_states]
    #Get prob. distribution of the last element in the last seq of the batch
    fprop = theano.function([x] + states_as_params, activations + [y_hat[-1, -1, :]], givens=zip(initial_states, states_as_params))

    states_values = [initial.get_value() for initial in initial_states]
    sys.stdout.write('Starting sampling\n' + primetext)
    for _ in range(args.length):
        idx, states_values = sample(x_curr, states_values, fprop, args.temperature)
        sys.stdout.write(ix_to_char[idx])
        x_curr = [[idx]]

    sys.stdout.write('\n')
