from LayerMetaData import LayerMetaData
import tensorflow as tf

class ModelConfig(object):
    def __init__(self):
        self.Model_type = "Lenet"
        model_file = "mnist_model.h5"
        model = tf.keras.models.load_model(model_file)
        model.summary()
        self.layer_list = []
        self.input_n = 1
        self.input_h = 28
        self.input_w = 28
        self.input_c = 1
        self.input_bit = 16
        self.input_bit = 16
        self.filter_bit = 16
        
        filter_c = self.input_c
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                config = layer.get_config()
                n_filter = config['filters']
                filter_h = config['kernel_size'][0]
                filter_w = config['kernel_size'][1]
                stride = config['strides'][0]
                padding = config['padding'].upper()
                self.layer_list.append(LayerMetaData("convolution", n_filter, filter_h, filter_w, filter_c, stride, padding, 0, 0, 0, 0))
                filter_c = n_filter

            elif isinstance(layer, tf.keras.layers.Dense):
                config = layer.get_config()
                units = config['units']
                self.layer_list.append(LayerMetaData("fully", 0, 0, 0, 0, 0, 0, 0, 0, 0, units))

            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                config = layer.get_config()
                pool_h = config['pool_size'][0]
                pool_w = config['pool_size'][1]
                stride = config['strides'][0]
                self.layer_list.append(LayerMetaData("pooling", 0, 0, 0, 0, 0, 0, pool_h, pool_w, stride,   0))

            else:
                pass

        '''
        if self.Model_type == "Lenet":
            self.layer_list = [
                LayerMetaData("convolution",   6, 5, 5,  1, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",       0, 0, 0,  0, 0,       0, 2, 2, 2,   0),
                LayerMetaData("convolution",  16, 5, 5,  6, 1, 'VALID', 0, 0, 0,   0), # 16 , 5, 5, 6
                LayerMetaData("convolution", 120, 5, 5, 16, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("fully",         0, 0, 0, 0,  0,       0, 0, 0, 0, 120),
                LayerMetaData("fully",         0, 0, 0,  0, 0,       0, 0, 0, 0,  84),
                LayerMetaData("fully",         0, 0, 0,  0, 0,       0, 0, 0, 0,  10)
                ]
            self.input_n = 1
            self.input_h = 32
            self.input_w = 32
            self.input_c = 1
            self.input_bit = 16
            self.filter_bit = 16

        elif self.Model_type == "Cifar10": # Model from 子賢paper
            self.layer_list = [
                LayerMetaData("convolution", 32, 5, 5,  3, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",      0, 0, 0,  0, 0,       0, 2, 2, 2,   0),
                LayerMetaData("convolution", 32, 5, 5, 32, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",      0, 0, 0,  0, 0,       0, 2, 2, 2,   0),
                LayerMetaData("convolution", 64, 5, 5, 32, 1,  'SAME', 0, 0, 0,   0),
                LayerMetaData("pooling",      0, 0, 0,  0, 0,       0, 2, 2, 2,   0),
                LayerMetaData("fully",        0, 0, 0, 0,  0,       0, 0, 0, 0,  64),
                LayerMetaData("fully",        0, 0, 0, 0,  0,       0, 0, 0, 0,  10)
                ]
            self.input_n = 1
            self.input_h = 32
            self.input_w = 32
            self.input_c = 3
            self.input_bit = 16
            self.filter_bit = 16

        elif self.Model_type == "DeepID":
            self.layer_list = [
                LayerMetaData("convolution",   20, 4, 4,  1, 1, 'VALID', 0, 0, 0,    0),
                LayerMetaData("pooling",        0, 0, 0,  0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("convolution",   40, 3, 3, 20, 1, 'VALID', 0, 0, 0,    0),
                LayerMetaData("pooling",        0, 0, 0,  0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("convolution",   60, 3, 3, 40, 1, 'VALID', 0, 0, 0,    0),
                LayerMetaData("pooling",        0, 0, 0,  0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("convolution",   80, 2, 2, 60, 1, 'VALID', 0, 0, 0,    0),
                LayerMetaData("fully",         0,  0, 0,  0, 0,       0, 0, 0, 0,  160),
                LayerMetaData("fully",         0,  0, 0,  0, 0,       0, 0, 0, 0,  100)
            ]
            self.input_n = 1
            self.input_h = 39
            self.input_w = 31
            self.input_c = 1
            self.input_bit = 16
            self.filter_bit = 16

        elif self.Model_type == "Caffenet":
            self.layer_list = [
                LayerMetaData("convolution",  96, 11, 11,   3, 4, 'VALID', 0, 0, 0,    0),
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 3, 3, 2,    0),
                LayerMetaData("convolution", 256,  5,  5,  96, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 3, 3, 2,    0),
                LayerMetaData("convolution", 384,  3,  3, 256, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution", 384,  3,  3, 384, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution", 256,  3,  3, 384, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 3, 3, 2,    0),
                LayerMetaData("fully",         0,  0,  0,   0, 0,       0, 0, 0, 0, 4096), # 6x6x256
                LayerMetaData("fully",         0,  0,  0,   0, 0,       0, 0, 0, 0, 4096), # 4096
                LayerMetaData("fully",         0,  0,  0,   0, 0,       0, 0, 0, 0, 1000)
                ]
            self.input_n = 1
            self.input_h = 227
            self.input_w = 227
            self.input_c = 3
            self.input_bit = 16
            self.filter_bit = 16

        elif self.Model_type == "Overfeat": # Fast
            self.layer_list = [
                LayerMetaData("convolution",  96, 11, 11,   3, 4, 'VALID', 0, 0, 0,    0),
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("convolution", 256,  5,  5,  96, 1, 'VALID', 0, 0, 0,    0),
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("convolution", 512,  3,  3, 256, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution",1024,  3,  3, 512, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution",1024,  3,  3,1024, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("fully",         0,  0,  0,   0, 0,       0, 0, 0, 0, 3072),
                LayerMetaData("fully",         0,  0,  0,   0, 0,       0, 0, 0, 0, 4096),
                LayerMetaData("fully",         0,  0,  0,   0, 0,       0, 0, 0, 0, 1000)
                ]
            self.input_n = 1
            self.input_h = 221
            self.input_w = 221
            self.input_c = 3
            self.input_bit = 16
            self.filter_bit = 16

        elif self.Model_type == "VGG16":
            self.layer_list = [
                LayerMetaData("convolution",  64,  3,  3,   3, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution",  64,  3,  3,  64, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("convolution", 128,  3,  3,  64, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution", 128,  3,  3, 128, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("convolution", 256,  3,  3, 128, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution", 256,  3,  3, 256, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution", 256,  3,  3, 256, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("convolution", 512,  3,  3, 256, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution", 512,  3,  3, 512, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution", 512,  3,  3, 512, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("convolution", 512,  3,  3, 256, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution", 512,  3,  3, 512, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution", 512,  3,  3, 512, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("fully",         0,  0,  0,   0, 0,       0, 0, 0, 0, 4096),
                LayerMetaData("fully",         0,  0,  0,   0, 0,       0, 0, 0, 0, 4096),
                LayerMetaData("fully",         0,  0,  0,   0, 0,       0, 0, 0, 0, 1000)
                ]
            self.input_n = 1
            self.input_h = 224
            self.input_w = 224
            self.input_c = 3
            self.input_bit = 16
            self.filter_bit = 16

        elif self.Model_type == "Test":
            self.layer_list = [
                LayerMetaData("convolution",      1,  3,  3,  1, 1,  'SAME', 0, 0, 0,    0),
                #LayerMetaData("pooling",         0,  0,  0,   0, 0,       0, 3, 3, 2,   0),
                #LayerMetaData("convolution",      1,  3,  3,  32, 1, 'VALID', 0, 0, 0,   0),
                #LayerMetaData("pooling",         0,  0,  0,   0, 0,       0, 3, 3, 1,   0),
                #LayerMetaData("convolution",     1,  2,  2,   1, 1, 'VALID', 0, 0, 0,   0),
                #LayerMetaData("fully",           0,  0,  0,   0, 0,       0, 0, 0, 0,    2),
                #LayerMetaData("fully",           0,  0,  0,   0, 0,       0, 0, 0, 0,   2)
                ]
            self.input_n = 1
            self.input_h = 3
            self.input_w = 3
            self.input_c = 1
            self.input_bit = 1
            self.filter_bit = 16
        
        else:
            print("Wrong model type")
            exit()
        '''
    def __str__(self):
            return str(self.__dict__)

