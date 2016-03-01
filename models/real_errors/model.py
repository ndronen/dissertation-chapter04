import os
import sys
sys.setrecursionlimit(5000)

import numpy as np

from keras.utils import np_utils
from keras.models import Graph, Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer, Highway
from keras.layers.normalization import BatchNormalization
import keras.callbacks
from modeling.builders import (build_embedding_layer,
    build_convolutional_layer, build_pooling_layer,
    build_dense_layer, build_optimizer, load_weights,
    build_hierarchical_softmax_layer)
from modeling.dataset import HDF5FileDataset
from modeling.callbacks import (PredictionCallback,
        ClassificationReport, ConfusionMatrix)

class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)

def build_graph(config, train_data):
    graph = Graph()

    input_width = train_data[config.data_name[0]].shape[1]

    graph.add_input(config.data_name[0], input_shape=(input_width,), dtype='int')

    graph.add_node(build_embedding_layer(config, input_width=input_width),
            name='embedding', input=config.data_name[0])

    graph.add_node(build_convolutional_layer(config), name='conv', input='embedding')
    prev_layer = 'conv'
    if config.batch_normalization:
        graph.add_node(BatchNormalization(), name='conv_bn', input=prev_layer)
        prev_layer = 'conv_bn'
    graph.add_node(Activation('relu'), name='conv_relu', input=prev_layer)

    graph.add_node(build_pooling_layer(config, input_width=input_width),
            name='pool', input='conv_relu')

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

    graph.add_node(build_dense_layer(config, 2,
        activation='softmax'), name='softmax', input=prev_layer)

    graph.add_output(name=config.target_name, input='softmax')

    load_weights(config, graph)

    optimizer = build_optimizer(config)

    graph.compile(loss={config.target_name: config.loss}, optimizer=optimizer)

    return graph


def build_sequential(config, train_data):
    model = Sequential()

    input_width = train_data[config.data_name[0]].shape[1]

    model.add(build_embedding_layer(config, input_width=input_width))
    model.add(build_convolutional_layer(config))
    if config.batch_normalization:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(build_pooling_layer(config, input_width=input_width))
    model.add(Flatten())

    for i in range(config.n_fully_connected):
        model.add(build_dense_layer(config, n_hidden=config.n_hidden))
        if config.batch_normalization:
            model.add(BatchNormalization())
        model.add(Activation('relu'))

    for i in range(config.n_highway):
        model.add(Highway(activation='relu'))
        if config.batch_normalization:
            model.add(BatchNormalization())

    model.add(build_dense_layer(config, 2))
    #if config.batch_normalization:
    #    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    load_weights(config, model)

    optimizer = build_optimizer(config)

    model.compile(loss=config.loss, optimizer=optimizer)

    return model


def build_callbacks(model, config, X_validation=None, y_validation=None):
    callbacks = []

    callbacks.append(keras.callbacks.EarlyStopping(
            patience=config.patience, verbose=1))

    if 'persistent' in config.mode:
        callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(config.model_path, config.checkpoint_name),
                save_best_only=config.save_best_only,
                verbose=0))

    pred_cb = PredictionCallback(
            X_validation,
            logger=config.callback_logger,
            batch_size=config.batch_size)

    pred_callbacks = []

    if config.confusion_matrix:
        if isinstance(model, keras.models.Sequential):
            assert X_validation is not None
            assert y_validation is not None
            confusion_cb = ConfusionMatrix(
                    X_validation, y_validation,
                    logger=config.callback_logger)
            pred_callbacks.append(confusion_cb)

    if config.classification_report:
        if isinstance(model, keras.models.Sequential):
            assert X_validation is not None
            assert y_validation is not None
            report_cb = ClassificationReport(
                    X_validation,
                    y_validation,
                    logger=config.callback_logger)
            pred_callbacks.append(report_cb)

    for pcb in pred_callbacks:
        pred_cb.add(pcb)

    if len(pred_callbacks):
        callbacks.append(pred_cb)

    return callbacks


def fit(config):
    if 'background' in config.mode:
        sys.stdout, sys.stderr = setup_logging(
                os.path.join(config.model_path, 'model.log'))

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

    if config.n_residual_blocks > 0:
        model = build_graph(config, train_data)
    else:
        model = build_sequential(config, train_data)

    class_weight = train_data.class_weights(
        config.class_weight_exponent,
        config.target_name)

    if config.n_residual_blocks > 0:
        callbacks=build_callbacks(model, config)

        # Go through 1/10th of the training set every epoch.
        samples_per_epoch = int(train_data.n/10)

        model.fit_generator(train_data.generator(),
                samples_per_epoch=samples_per_epoch,
                nb_epoch=10*config.n_epochs,
                #nb_worker=1,
                callbacks=callbacks,
                validation_data=validation_data.generator(),
                nb_val_samples=20000,
                class_weight=class_weight)
    else:
        rng = np.random.RandomState(seed=config.seed)

        X_train = train_data.hdf5_file[config.data_name[0]].value
        y_train = np_utils.to_categorical(
                train_data.hdf5_file[config.target_name].value,
                nb_classes=config.n_classes)
        train_idx = rng.permutation(len(X_train))
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]
        X_train = X_train[1:200000]
        y_train = y_train[1:200000]

        X_validation = validation_data.hdf5_file[config.data_name[0]].value
        y_validation = train_data.hdf5_file[config.target_name].value
        y_validation_one_hot = np_utils.to_categorical(y_validation,
                nb_classes=config.n_classes)
        validation_idx = rng.permutation(len(X_validation))
        X_validation = X_validation[validation_idx]
        y_validation = y_validation[validation_idx]
        y_validation_one_hot = y_validation_one_hot[validation_idx]
        X_validation = X_validation[1:200000]
        y_validation = y_validation[1:200000]
        y_validation_one_hot = y_validation_one_hot[1:200000]

        callbacks=build_callbacks(model, config, 
                X_validation, y_validation)

        model.fit(X=X_train, y=y_train,
                nb_epoch=config.n_epochs,
                validation_data=(X_validation, y_validation_one_hot),
                callbacks=callbacks,
                class_weight=class_weight,
                show_accuracy=True,
                batch_size=config.batch_size,
                shuffle="batch")
