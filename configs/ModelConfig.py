from LayerMetaData import LayerMetaData

class ModelConfig(object):
    def __init__(self):
        model_type = 5
        if model_type == 0: # TestModelConfig
            self.Model_type = "Test0"
            self.layer_list = [
                LayerMetaData("convolution", 64, 3, 3, 64, 1, 'VALID', 0, 0, 0,  0),
                #LayerMetaData("convolution",  96, 11, 11,   3, 4, 'VALID', 0, 0, 0,    0),
                #LayerMetaData("convolution",  1,  3, 3, 1, 1, 'SAME', 0, 0, 0,  0),
                #LayerMetaData("convolution",  1,  2, 2, 1, 1, 'VALID', 0, 0, 0,  0),
                #LayerMetaData("pooling",       0, 0, 0,  0, 0,       0, 2, 2, 1,  0),
                #LayerMetaData("convolution",  1,  2, 2, 1, 1, 'VALID', 0, 0, 0,  0),
                #LayerMetaData("fully",        0, 0, 0,  0, 0,       0, 0, 0, 0, 20)
                ]
            self.input_n = 1
            self.input_h = 30 #50
            self.input_w = 30 #50
            self.input_c = 64
            self.input_bit = 16
            self.filter_bit = 16
        elif model_type == 1: # TestModelConfig1
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
        elif model_type == 2: # Cifar10Config
            self.Model_type = "Cifar10"
            self.layer_list = [
                LayerMetaData("convolution", 32, 3, 3,  3, 1,  'SAME', 0, 0, 0,   0),
                LayerMetaData("convolution", 32, 3, 3, 32, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",      0, 0, 0,  0, 1,       0, 2, 2, 1,   0),
                LayerMetaData("convolution", 64, 3, 3, 32, 1,  'SAME', 0, 0, 0,   0),
                LayerMetaData("convolution", 64, 3, 3, 64, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",      0, 0, 0, 0,  0,       0, 2, 2, 1,   0),
                LayerMetaData("fully",        0, 0, 0, 0,  0,       0, 0, 0, 0, 512),
                LayerMetaData("fully",        0, 0, 0, 0,  0,       0, 0, 0, 0,  10)
                ]
            self.input_n = 1
            self.input_h = 32
            self.input_w = 32
            self.input_c = 3
            self.input_bit = 16
            self.filter_bit = 16
        elif model_type == 3: # CaffenetConfig
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
        elif model_type == 4: # LenetConfig
            self.Model_type = "Lenet"
            self.layer_list = [
                LayerMetaData("convolution",   6, 5, 5,  1, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",       0, 0, 0,  0, 0,       0, 2, 2, 2,   0),
                LayerMetaData("convolution",  16, 5, 5,  6, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("convolution", 120, 5, 5, 16, 1, 'VALID', 0, 0, 0,   0),
                #LayerMetaData("fully",        0, 0, 0, 0, 0,       0, 0, 0, 0, 120),
                LayerMetaData("fully",         0, 0, 0,  0, 0,       0, 0, 0, 0,  84),
                LayerMetaData("fully",         0, 0, 0,  0, 0,       0, 0, 0, 0,  10)
                ]
            self.input_n = 1
            self.input_h = 32
            self.input_w = 32
            self.input_c = 1
            self.input_bit = 16
            self.filter_bit = 16
        elif model_type == 5: # DeepIDConfig
            self.Model_type = "DeepID"
            self.layer_information = [
                LayerMetaData("convolution",   20, 4, 4,  1, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",        0, 0, 0,  0, 0,       0, 2, 2, 2,   0),
                LayerMetaData("convolution",   40, 3, 3, 20, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",        0, 0, 0,  0, 0,       0, 2, 2, 2,   0),
                LayerMetaData("convolution",   60, 3, 3, 40, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",        0, 0, 0,  0, 0,       0, 2, 2, 2,   0),
                LayerMetaData("convolution",   80, 3, 3, 60, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("fully",         0, 0, 0,  0, 0,       0, 0, 0, 0,  160),
                LayerMetaData("fully",         0, 0, 0,  0, 0,       0, 0, 0, 0,  100)
            ]
            self.input_n = 1
            self.input_h = 39
            self.input_w = 31
            self.input_c = 1
            self.input_bit = 16
            self.filter_bit = 16
        elif model_type == 5: # TestModelConfig2
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
    def __str__(self):
            return str(self.__dict__)
