from LayerMetaData import LayerMetaData

class TestModelConfig2(object):
    def __init__(self):
        #layer data
        self.layer_list = [
            LayerMetaData("convolution", 1, 3, 3, 1, 0, 0, 0),
            LayerMetaData("pooling",     0, 0, 0, 0, 2, 2, 0),
            #LayerMetaData("fully",       0, 0, 0, 0, 0, 0, 2)
            ]

        #model data
        self.input_n = 1
        self.input_h = 5
        self.input_w = 5
        self.input_c = 1
        
        #bit decomposition
        self.input_bit = 1
        self.filter_bit = 2