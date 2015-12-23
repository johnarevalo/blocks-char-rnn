from blocks import initialization
from blocks.bricks import Linear, NDimensionalSoftmax, Tanh
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, LSTM, SimpleRecurrent
from blocks.bricks.lookup import LookupTable


def initialize(to_init):
    for bricks in to_init:
        bricks.weights_init = initialization.Uniform(width=0.08)
        bricks.biases_init = initialization.Constant(0)
        bricks.initialize()


def softmax_layer(h, y, y_mask, vocab_size, hidden_size):
    hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_size,
                              output_dim=vocab_size)
    initialize([hidden_to_output])
    linear_output = hidden_to_output.apply(h)
    linear_output.name = 'linear_output'
    softmax = NDimensionalSoftmax()
    y_hat = softmax.apply(linear_output, extra_ndim=1)
    y_hat.name = 'y_hat'
    cost = softmax.categorical_cross_entropy(y, linear_output, extra_ndim=1)
    cost = (cost * y_mask).sum() / y_mask.sum()
    cost.name = 'cost'
    return y_hat, cost


def rnn_layer(dim, h, n, x_mask):
    linear = Linear(input_dim=dim, output_dim=dim, name='linear' + str(n))
    rnn = SimpleRecurrent(dim=dim, activation=Tanh(), name='rnn' + str(n))
    initialize([linear, rnn])
    return rnn.apply(linear.apply(h), mask=x_mask)


def gru_layer(dim, h, n, x_mask):
    fork = Fork(output_names=['linear' + str(n), 'gates' + str(n)],
                name='fork' + str(n), input_dim=dim, output_dims=[dim, dim * 2])
    gru = GatedRecurrent(dim=dim, name='gru' + str(n))
    initialize([fork, gru])
    linear, gates = fork.apply(h)
    return gru.apply(linear, gates, mask=x_mask)


def lstm_layer(dim, h, n, x_mask):
    linear = Linear(input_dim=dim, output_dim=dim * 4, name='linear' + str(n))
    lstm = LSTM(dim=dim, name='lstm' + str(n))
    initialize([linear, lstm])
    return lstm.apply(linear.apply(h), mask=x_mask)


def nn_fprop(x, y, x_mask, y_mask, vocab_size, hidden_size, num_layers, model):
    lookup = LookupTable(length=vocab_size, dim=hidden_size)
    initialize([lookup])
    h = lookup.apply(x)
    cells = []
    for i in range(num_layers):
        mask = x_mask if i == 0 else None
        if model == 'rnn':
            h = rnn_layer(hidden_size, h, i, mask)
        if model == 'gru':
            h = gru_layer(hidden_size, h, i, mask)
        if model == 'lstm':
            h, c = lstm_layer(hidden_size, h, i, mask)
            cells.append(c)
    return softmax_layer(h, y, y_mask, vocab_size, hidden_size) + (cells, )
