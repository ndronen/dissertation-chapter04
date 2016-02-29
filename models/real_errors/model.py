import sys
sys.setrecursionlimit(5000)

import numpy as np

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.normalization import BatchNormalization
from modeling.builders import (build_embedding_layer,
    build_convolutional_layer, build_pooling_layer,
    build_dense_layer, build_optimizer, load_weights,
    build_hierarchical_softmax_layer)
from modeling.dataset import HDF5FileDataset

class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)

def build_model(config, train_data):
    graph = Graph()

    input_width = train_data.data[config.target_name].shape[1]

    graph.add_input('input', input_shape=(config.input_width,), dtype='int')

    graph.add_node(build_embedding_layer(config), name='embedding', input='input')

    graph.add_node(build_convolutional_layer(config), name='conv', input='embedding')
    prev_layer = 'conv'
    if config.batch_normalization:
        graph.add_node(BatchNormalization(), name='conv_bn', input=prev_layer)
        prev_layer = 'conv_bn'
    graph.add_node(Activation('relu'), name='conv_relu', input=prev_layer)

    graph.add_node(build_pooling_layer(config), name='pool', input='conv_relu')

    graph.add_node(Flatten(), name='flatten', input='pool')
    prev_layer = 'flatten'

    # Add some number of fully-connected layers without skip connections.
    for i in range(config.n_fully_connected):
        layer_name = 'dense%02d' %i
        l = build_dense_layer(config, n_hidden=config.n_hidden)
        graph.add_node(l, name=layer_name, input=prev_layer)
        prev_layer = layer_name
        if config.batch_normalization:
            graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
            prev_layer = layer_name+'bn'
        if config.dropout_fc_p > 0.:
            graph.add_node(Dropout(config.dropout_fc_p), name=layer_name+'do', input=prev_layer)
            prev_layer = layer_name+'do'
    
    # Add sequence of residual blocks.
    for i in range(config.n_residual_blocks):
        # Add a fixed number of layers per residual block.
        block_name = '%02d' % i

        graph.add_node(Identity(), name=block_name+'input', input=prev_layer)
        prev_layer = block_input_layer = block_name+'input'

        try:
            n_layers_per_residual_block = config.n_layers_per_residual_block
        except AttributeError:
            n_layers_per_residual_block = 2

        for layer_num in range(n_layers_per_residual_block):
            layer_name = 'h%s%02d' % (block_name, layer_num)
    
            l = build_dense_layer(config, n_hidden=config.n_hidden)
            graph.add_node(l, name=layer_name, input=prev_layer)
            prev_layer = layer_name
    
            if config.batch_normalization:
                graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
                prev_layer = layer_name+'bn'
    
            if i < n_layers_per_residual_block:
                a = Activation('relu')
                graph.add_node(Activation('relu'), name=layer_name+'relu', input=prev_layer)
                prev_layer = layer_name+'relu'
                if config.dropout_fc_p > 0.:
                    graph.add_node(Dropout(config.dropout_fc_p), name=layer_name+'do', input=prev_layer)
                    prev_layer = layer_name+'do'

        graph.add_node(Identity(), name=block_name+'output', inputs=[block_input_layer, prev_layer], merge_mode='sum')
        graph.add_node(Activation('relu'), name=block_name+'relu', input=block_name+'output')
        prev_layer = block_input_layer = block_name+'relu'

    if hasattr(config, 'n_hsm_classes'):
        graph.add_node(build_hierarchical_softmax_layer(config),
                name='softmax', input=prev_layer)
    else:
        graph.add_node(build_dense_layer(config, config.n_classes,
                activation='softmax'), name='softmax', input=prev_layer)

    graph.add_output(name='output', input='softmax')

    load_weights(config, graph)

    optimizer = build_optimizer(config)

    graph.compile(loss={'output': config.loss}, optimizer=optimizer)

    return graph


def fit(config, callbacks=[]):
    train_data = HDF5FileDataset(
        config.train_path,
        config.data_name,
        [config.target_name],
        config.batch_size,
        config.seed)

    validation_data = HDF5FileDataset(
        config.validation_path,
        config.data_name,
        [config.target_name],
        config.batch_size,
        config.seed)

    graph = build_model(config, train_data)

    class_weight = train_data.class_weights(
        config.class_weight_exponent,
        config.target_name)

    graph.fit_generator(train_data.generate(),
            samples_per_epoch=int(train_data.n/100),
            nb_epoch=config.n_epochs,
            validation_data=validation_data.get_dict(),
            callbacks=callbacks,
            class_weight=train_data.class_weights(config.class_weight_exponent))
