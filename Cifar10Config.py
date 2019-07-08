from LayerMetaData import LayerMetaData
class Cifar10Config(object):
    def __init__(self):
        #layer data
        self.layer_list = [
            LayerMetaData("convolution", 16, 3, 3, 3, 0, 0, 0),
            #LayerMetaData("pooling", 0, 0, 0, 0, 2, 2, 0),
            LayerMetaData("convolution", 16, 3, 3, 16, 0, 0, 0),
            #LayerMetaData("pooling", 0, 0, 0, 0, 2, 2, 0),
            LayerMetaData("convolution", 32, 3, 3, 16, 0, 0, 0),
            #LayerMetaData("pooling", 0, 0, 0, 0, 2, 2, 0),
            LayerMetaData("output", 0, 0, 0, 0, 0, 0, 10)
            ]

        #model data
        self.input_n = 1
        self.input_h = 48
        self.input_w = 48
        self.input_c = 3

        #bit decomposition
        self.input_bit = 2
        self.filter_bit = 2

