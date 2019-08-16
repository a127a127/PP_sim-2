from LayerMetaData import LayerMetaData
class Cifar10Config(object):
    def __init__(self):
        # Layer data
        self.layer_list = [
            LayerMetaData("convolution", 8, 3, 3, 3, 0, 0, 0),
            #LayerMetaData("pooling", 0, 0, 0, 0, 2, 2, 0),
            #LayerMetaData("pooling", 0, 0, 0, 0, 2, 2, 0),
            LayerMetaData("convolution", 4, 3, 3, 8, 0, 0, 0),
            LayerMetaData("convolution", 4, 3, 3, 4, 0, 0, 0),
            LayerMetaData("convolution", 4, 3, 3, 4, 0, 0, 0),
            #LayerMetaData("pooling", 0, 0, 0, 0, 2, 2, 0),
            LayerMetaData("fully", 0, 0, 0, 0, 0, 0, 10)
            ]

        # Input feature map
        self.input_n = 1
        self.input_h = 12
        self.input_w = 12
        self.input_c = 3

        # Bit width
        self.input_bit = 2
        self.filter_bit = 2

