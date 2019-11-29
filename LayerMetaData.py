class LayerMetaData(object):
    def __init__(self, layer_type, filter_n, filter_h, filter_w, filter_c, strides, padding, pooling_h, pooling_w, pooling_strides, neuron_n):
        self.layer_type = layer_type # "convolution", "pooling", "fully"
        # filter's height and width should be even number
        self.filter_n   = filter_n
        self.filter_h   = filter_h
        self.filter_w   = filter_w
        self.filter_c   = filter_c
        self.strides    = strides
        self.padding    = padding
        self.pooling_h  = pooling_h
        self.pooling_w  = pooling_w
        self.pooling_strides = pooling_strides
        self.neuron_n   = neuron_n

    def __str__(self):
        return str(self.__dict__)
