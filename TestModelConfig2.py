from LayerMetaData import LayerMetaData

class TestModelConfig2(object):
    def __init__(self):
        #layer data
        self.layer_list = [
            LayerMetaData("convolution", 2, 3, 2, 1, 0, 0, 0),
            #LayerMetaData("convolution", 1, 2, 2, 1, 0, 0, 0),
            #LayerMetaData("pooling",     0, 0, 0, 0, 2, 2, 0),
            #LayerMetaData("fully",       0, 0, 0, 0, 0, 0, 1)
            ]

        #model data
        self.input_n = 1
        self.input_h = 3
        self.input_w = 2
        self.input_c = 1
        
        #bit decomposition
        self.input_bit = 1
        self.filter_bit = 2