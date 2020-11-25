from LayerMetaData import LayerMetaData

class ModelConfig(object):
    def __init__(self, model_type):
        self.Model_type = model_type
        if self.Model_type   == "Lenet":
            self.layer_list = [
                LayerMetaData("convolution",   6, 5, 5,  1, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",       0, 0, 0,  0, 0,       0, 2, 2, 2,   0),
                LayerMetaData("convolution",  16, 5, 5,  6, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("pooling",       0, 0, 0,  0, 0,       0, 2, 2, 2,   0),
                LayerMetaData("convolution", 120, 5, 5, 16, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("fully",         0, 0, 0, 0,  0,       0, 0, 0, 0, 120), # 1x1x120
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
                LayerMetaData("fully",        0, 0, 0, 0,  0,       0, 0, 0, 0,  64), # 2x2x64
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
                LayerMetaData("convolution",   20, 4, 4,  1, 1, 'VALID', 0, 0, 0,    0), #36x28
                LayerMetaData("pooling",        0, 0, 0,  0, 0,       0, 2, 2, 2,    0), #18x14
                LayerMetaData("convolution",   40, 3, 3, 20, 1, 'VALID', 0, 0, 0,    0), #16x12
                LayerMetaData("pooling",        0, 0, 0,  0, 0,       0, 2, 2, 2,    0), #8x6
                LayerMetaData("convolution",   60, 3, 3, 40, 1, 'VALID', 0, 0, 0,    0), #6x4
                LayerMetaData("pooling",        0, 0, 0,  0, 0,       0, 2, 2, 2,    0), #3x2
                LayerMetaData("convolution",   80, 2, 2, 60, 1, 'VALID', 0, 0, 0,    0), #2x1
                LayerMetaData("fully",         0,  0, 0,  0, 0,       0, 0, 0, 0,  160), #160x160
                LayerMetaData("fully",         0,  0, 0,  0, 0,       0, 0, 0, 0,  100)  #160x100
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
                LayerMetaData("fully",         0,  0,  0,   0, 0,       0, 0, 0, 0, 4096//4), # 6x6x256
                LayerMetaData("fully",         0,  0,  0,   0, 0,       0, 0, 0, 0, 4096//4),
                LayerMetaData("fully",         0,  0,  0,   0, 0,       0, 0, 0, 0, 1000//4)
                ]
            self.input_n = 1
            self.input_h = 227//2
            self.input_w = 227//2
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
                LayerMetaData("fully",         0,  0,  0,   0, 0,       0, 0, 0, 0, 3072), # 5x5x1024
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
                LayerMetaData("convolution", 128,  3,  3,  64, 1,  'SAME', 0, 0, 0,    0), #3
                LayerMetaData("convolution", 128,  3,  3, 128, 1,  'SAME', 0, 0, 0,    0), #4
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("convolution", 256,  3,  3, 128, 1,  'SAME', 0, 0, 0,    0), #5
                LayerMetaData("convolution", 256,  3,  3, 256, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution", 256,  3,  3, 256, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("convolution", 512,  3,  3, 256, 1,  'SAME', 0, 0, 0,    0), #8
                LayerMetaData("convolution", 512,  3,  3, 512, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution", 512,  3,  3, 512, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("pooling",       0,  0,  0,   0, 0,       0, 2, 2, 2,    0),
                LayerMetaData("convolution", 512,  3,  3, 512, 1,  'SAME', 0, 0, 0,    0), #11
                LayerMetaData("convolution", 512,  3,  3, 512, 1,  'SAME', 0, 0, 0,    0),
                LayerMetaData("convolution", 512,  3,  3, 512, 1,  'SAME', 0, 0, 0,    0), #13
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
                LayerMetaData("convolution",      1, 2, 2 , 1, 1, 'VALID', 0, 0, 0,   0),
                LayerMetaData("convolution",      1, 2, 2 , 1, 1, 'VALID', 0, 0, 0,   0),
                #LayerMetaData("pooling",         0, 0, 0,  0, 0,       0, 2, 2, 1,   0),
                #LayerMetaData("fully",           0, 0, 0,  0, 0,       0, 0, 0, 0,   2)
                #LayerMetaData("convolution",     1, 2, 2,  4, 1, 'VALID', 0, 0, 0,   0),
                #LayerMetaData("fully",         0, 0, 0,  0, 0,       0, 0, 0, 0,  4)
                #LayerMetaData("convolution",      2,  2,  2,  1, 1, 'VALID', 0, 0, 0,   0),
                #LayerMetaData("convolution",      1,  3,  3,  1, 1, 'SAME', 0, 0, 0,   0),
                #LayerMetaData("pooling",          0,  0,  0,   0, 0,       0, 2, 2, 1,   0),
                #LayerMetaData("fully",           0,  0,  0,   0, 0,       0, 0, 0, 0,   1)
                ]
            self.input_n = 1
            self.input_h = 3
            self.input_w = 3
            self.input_c = 1
            self.input_bit = 16
            self.filter_bit = 4
        
        else:
            print("Wrong model type")
            exit()
    
    def __str__(self):
            return str(self.__dict__)
