from LayerMetaData import LayerMetaData

class TestModelConfig(object):
    def __init__(self):
        #layer data
        self.layer_list = [
            LayerMetaData("convolution", 20, 5, 5, 3, 0, 0, 0), # (type, conv_n, conv_h, conv_w, conv_c, pool_h, pool_w, fully_n)
            LayerMetaData("convolution", 10, 2, 2, 20, 0, 0, 0),
            #LayerMetaData("pooling",     0, 0, 0, 0, 2, 2, 0),
            LayerMetaData("fully",      0, 0, 0, 0, 0, 0, 10),
            ]

        #model data
        self.input_n = 1
        self.input_h = 7
        self.input_w = 7
        self.input_c = 3
        
        #bit decomposition
        self.input_bit = 2
        self.filter_bit = 2