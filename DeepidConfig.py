from LayerMetaData import LayerMetaData
class DeepidConfig(object):
    def __init__(self):
        #layer data
        self.layer_information = [
            LayerMetaData("convolution", 20, 4, 4, 1, 0, 0, 0),
            LayerMetaData("pooling", 0, 0, 0, 0, 2, 2, 0),
            LayerMetaData("convolution", 40, 3, 3, 20, 0, 0, 0),
            LayerMetaData("pooling", 0, 0, 0, 0, 2, 2, 0),
            LayerMetaData("convolution", 60, 3, 3, 40, 0, 0, 0),
            LayerMetaData("pooling", 0, 0, 0, 0, 2, 2, 0),
            LayerMetaData("convolution", 80, 2, 2, 60, 0, 0, 0),
            LayerMetaData("fully", 0, 0, 0, 0, 0, 0, 160),
            LayerMetaData("output", 0, 0, 0, 0, 0, 0, 100)
            ]

        #model data
        self.input_h = 39
        self.input_w = 36
        self.input_c = 1
        self.input_n = 1

        #bit decomposition
        self.input_bit = 2
        self.filter_bit = 8