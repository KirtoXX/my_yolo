import numpy as np
from utils import WeightReader

wt_path = 'weight/yolo-voc.weights'

def load_weight(model):
    # load weight
    nb_conv = 23
    weight_reader = WeightReader(wt_path)
    weight_reader.reset()
    for i in range(1, nb_conv):
        conv_layer = model.get_layer('conv_' + str(i))

        if i < nb_conv:
            norm_layer = model.get_layer('norm_' + str(i))

            size = np.prod(norm_layer.get_weights()[0].shape)

            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel])
    print('weight load finish')
    return model