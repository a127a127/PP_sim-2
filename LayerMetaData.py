class LayerMetaData(object):
    def __init__(self, layer_type, filter_n, filter_h, filter_w, filter_c, pooling_h, pooling_w, neuron_n):
        self.layer_type = layer_type # "convolution", "pooling"

        # for convolution layer information
        self.filter_n = filter_n
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.filter_c = filter_c
        self.filter_length = self.filter_h * self.filter_w * self.filter_c

        # for pooling layer information
        self.pooling_h = pooling_h
        self.pooling_w = pooling_w

        # for fully connected / output layer information
        self.neuron_n = neuron_n



    def __str__(self):
        return str(self.__dict__)