from LayerMetaData import LayerMetaData
class Cifar10Config(object):
    def __init__(self):
        #layer data
        self.layer_information = [
            LayerMetaData("convolution", 32, 3, 3, 3, 0, 0, 0),
            LayerMetaData("pooling", 0, 0, 0, 0, 2, 2, 0),
            LayerMetaData("convolution", 32, 3, 3, 32, 0, 0, 0),
            LayerMetaData("pooling", 0, 0, 0, 0, 2, 2, 0),
            LayerMetaData("convolution", 64, 3, 3, 32, 0, 0, 0),
            LayerMetaData("pooling", 0, 0, 0, 0, 2, 2, 0),
            LayerMetaData("output", 0, 0, 0, 0, 0, 0, 10)
            ]

        #model data
        self.input_h = 48
        self.input_w = 48
        self.input_c = 3
        self.input_n = 1

        #bit decomposition
        self.input_bit = 2
        self.filter_bit = 8

