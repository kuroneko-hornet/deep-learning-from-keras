from keras import Model
from keras.layers import (
    Input, Dense, BatchNormalization, Dropout,
    Reshape, Conv2D, MaxPool2D, Flatten
)
from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adam
from keras.initializers import he_normal, he_uniform
from utils import *


def load_preprocess_data(params):
    '''
        args: params
        output: x_train, y_train, x_test, y_test
    '''
    use_fashipon = True
    if use_fashipon: 
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # preprocess
    x_train = x_train.astype('float') /255 
    x_test = x_test.astype('float') /255 
    y_train = to_categorical(y_train, num_classes=params['output_dim'])
    y_test = to_categorical(y_test, num_classes=params['output_dim'])

    return x_train, y_train, x_test, y_test


def define_optimizer():
    optimizers = {
        'sgd':SGD(),
        'momentum':SGD(momentum=0.9),
        'adagrad':Adagrad(),
        'adam':Adam()
        }
    return optimizers


def define_params():
    ''' no args'''
    params = {
        'activation' : 'relu',
        'batch_size' : 100,
        'epochs' : 20,
        'hidden_dim' : 100,
        'img_shape' : (28, 28),
        'kernel_initializer' : 'he_normal',
        'kernel_size' : 3, # (3, 3)
        'learning_rate' : 0.01,
        'optimizers' : define_optimizer(),
        'output_dim' : 10,
    }
    return params


def set_model(params):
    ''' Args: params'''
    _input = Input(shape=params['img_shape'])
    _hidden = Reshape((28, 28, 1), input_shape=(28,28))(_input)
    _hidden = set_conv(num=2, filters=32, params=params, _inlayer=_hidden)
    _hidden = set_pool(num=1, _inlayer=_hidden)
    # _hidden = set_conv(num=1, filters=20, params=params, _inlayer=_hidden)
    # _hidden = set_pool(num=1, _inlayer=_hidden)
    _hidden = Flatten()(_hidden)
    _hidden = set_affine(num=1, params=params, _inlayer=_hidden)
    _output = Dense(params['output_dim'], activation='softmax')(_hidden)
    model = Model(inputs=_input, outputs=_output)
    return model


def train(model, params, opt, x_train, y_train, x_test, y_test):
    ''' Args: model, params, opt, x_train, y_train, x_test, y_test'''
    model.compile(\
        optimizer = opt,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])
    result = model.fit(\
        x = x_train, y = y_train,
        batch_size = params['batch_size'],
        epochs = params['epochs'],
        verbose = 1,
        validation_data = (x_test, y_test))
    return result


def main():
    params = define_params()
    x_train, y_train, x_test, y_test = \
        load_preprocess_data(params)

    result_histories = {}
    for opt_name, opt in params['optimizers'].items():
        model = set_model(params)
        # model.summary()
        _result = train(model, params, opt, x_train, y_train, x_test, y_test)
        result_histories[opt_name] = _result.history

    build_file('loss', '03', result_histories, params['epochs'])
    build_file('acc', '03', result_histories, params['epochs'])


if __name__ == '__main__':
    main()

