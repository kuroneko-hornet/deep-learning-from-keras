from keras.layers import (
    Input, Dense, BatchNormalization, Dropout,
    Conv2D, MaxPool2D, Flatten
)
import matplotlib.pyplot as plt

### === model ===

def set_batch_drop_layer(_inlayer, droprate=0.2):
    _batched = BatchNormalization()(_inlayer)
    _outlayer = Dropout(droprate)(_batched)
    return _outlayer


def set_conv(num, filters, params, _inlayer):
    _conv = _inlayer
    for _ in range(num):
        _conv = Conv2D(\
            filters = filters,
            kernel_size = params['kernel_size'],
            activation = params['activation'],
            kernel_initializer = params['kernel_initializer']
            )(_conv)
    return _conv


def set_pool(num, _inlayer):
    _pool = _inlayer
    for _ in range(num):
        _pool = MaxPool2D(pool_size=(2,2))(_pool)
    return _pool


def set_affine(num, params, _inlayer):
    _affine = _inlayer
    for _ in range(num):
        _affine = Dense(\
            units = params['hidden_dim'],
            activation = params['activation'],
            kernel_initializer = params['kernel_initializer']
            )(_affine)
    return _affine


def build_file(figure_name, no, result_histories, epochs):
    ''' Args: history, epochs'''
    h_size = len(result_histories)
    f1 = plt.figure(figsize=(5,h_size*2.5))

    plt_num = 1
    for k, v in result_histories.items():
        d_test = v[f'{figure_name}']
        d_train = v[f'val_{figure_name}']
        plt.subplot(h_size, 1,plt_num)
        plt.plot(range(1, epochs+1), d_train, marker='.', label=f'train')
        plt.plot(range(1, epochs+1), d_test, marker='.', label=f'test')

        plt.title(k)
        plt.legend(loc='best', fontsize=10)
        plt.grid()
        plt.xlabel('epochs')
        plt.ylabel(figure_name)
        plt.tight_layout()

        plt_num += 1

    plt.savefig(f'ch07_{figure_name}/{no}.jpg')