from LayerMetaData import LayerMetaData

class ModelConfig(object):
    def __init__(self):
        model_type = 0
        if model_type == 0: # Lenet
            self.Model_type = "Lenet"
            self.layer_list = [
                LayerMetaData("convolution",   6, 5, 5,  1, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",       0, 0, 0,  0, 0,       0, 2, 2, 2,   0),
                LayerMetaData("convolution",  16, 5, 5,  6, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("convolution", 120, 5, 5, 16, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("fully",         0, 0, 0, 0,  0,       0, 0, 0, 0, 120),
                LayerMetaData("fully",         0, 0, 0,  0, 0,       0, 0, 0, 0,  84),
                LayerMetaData("fully",         0, 0, 0,  0, 0,       0, 0, 0, 0,  10)
                ]
            self.input_n = 1
            self.input_h = 32
            self.input_w = 32
            self.input_c = 1
            self.input_bit = 1
            self.filter_bit = 16

        elif model_type == 1: # Cifar10 model from 子賢paper
            self.Model_type = "Cifar10"
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
            self.input_bit = 1
            self.filter_bit = 16

        elif model_type == 2: # DeepID
            self.Model_type = "DeepID"
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
            self.input_bit = 1
            self.filter_bit = 16

        elif model_type == 3: # Caffenet
            self.Model_type = "Caffenet"
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

        elif model_type == 4: # Test0
            self.Model_type = "Test0"
            self.layer_list = [
                LayerMetaData("convolution",   1, 2, 2, 1, 1, 'VALID', 0, 0, 0,  0),
                #LayerMetaData("pooling",       0, 0, 0,  0, 0,       0, 2, 2, 1,   0),
                LayerMetaData("convolution",   1, 2, 2, 1, 1, 'VALID', 0, 0, 0,  0),
                ]
            self.input_n = 1
            self.input_h = 4
            self.input_w = 4
            self.input_c = 1
            self.input_bit = 1
            self.filter_bit = 16
        elif model_type == 5: # Test1
            self.Model_type = "Test1"
            self.layer_list = [
                LayerMetaData("convolution", 20, 5, 5,  3, 1, 'VALID', 0, 0, 0,  0),
                LayerMetaData("convolution", 10, 2, 2, 20, 1, 'VALID', 0, 0, 0,  0),
                #LayerMetaData("pooling",      0, 0, 0,  0, 0,       0, 2, 2, 2,  0),
                LayerMetaData("fully",        0, 0, 0,  0, 0,       0, 0, 0, 0, 10)
                ]
            self.input_n = 1
            self.input_h = 7
            self.input_w = 7
            self.input_c = 3
            self.input_bit = 2
            self.filter_bit = 4
        elif model_type == 6: # TestModelConfig2
            self.Model_type = "Test2"
            self.layer_list = [
                LayerMetaData("convolution",  2, 2, 2,  1, 1, 'VALID', 0, 0, 0,  0),
                #LayerMetaData("convolution", 10, 2, 2, 20, 1, 'VALID', 0, 0, 0,  0),
                LayerMetaData("pooling",      0, 0, 0,  0, 0,       0, 2, 2, 1,  0),
                #LayerMetaData("fully",        0, 0, 0,  0, 0,       0, 0, 0, 0, 10)
                ]
            self.input_n = 1
            self.input_h = 4
            self.input_w = 4
            self.input_c = 1
            self.input_bit = 2
            self.filter_bit = 2
        elif model_type == 7: # 大cifar from keras
            self.Model_type = "Cifar10"
            self.layer_list = [
                LayerMetaData("convolution", 32, 3, 3,  3, 1,  'SAME', 0, 0, 0,   0),
                LayerMetaData("convolution", 32, 3, 3, 32, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",      0, 0, 0,  0, 0,       0, 2, 2, 2,   0),
                LayerMetaData("convolution", 64, 3, 3, 32, 1,  'SAME', 0, 0, 0,   0),
                LayerMetaData("convolution", 64, 3, 3, 64, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",      0, 0, 0, 0,  0,       0, 2, 2, 2,   0),
                LayerMetaData("fully",        0, 0, 0, 0,  0,       0, 0, 0, 0, 512),
                LayerMetaData("fully",        0, 0, 0, 0,  0,       0, 0, 0, 0,  10)
                ]
            self.input_n = 1
            self.input_h = 32
            self.input_w = 32
            self.input_c = 3
            self.input_bit = 16
            self.filter_bit = 16

    def __str__(self):
            return str(self.__dict__)
