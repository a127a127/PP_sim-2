from LayerMetaData import LayerMetaData
class LenetConfig(object):
    def __init__(self):
        #layer data
        self.layer_list = [
            LayerMetaData("convolution", 6, 5, 5, 1, 0, 0, 0),
            LayerMetaData("pooling",     0, 0, 0, 0, 2, 2, 0),
            LayerMetaData("convolution", 16, 5, 5, 6, 0, 0, 0),
            LayerMetaData("fully", 0, 0, 0, 0, 0, 0, 120),
            LayerMetaData("fully", 0, 0, 0, 0, 0, 0, 84),
            LayerMetaData("fully", 0, 0, 0, 0, 0, 0, 10)
            ]

        #model data
        self.input_n = 1
        self.input_h = 32
        self.input_w = 32
        self.input_c = 1
        
        #bit decomposition
        self.input_bit = 2
        self.filter_bit = 16
        