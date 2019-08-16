from LayerMetaData import LayerMetaData

class CaffenetConfig(object):
    def __init__(self):
        # Layer data
        self.layer_list = [
            LayerMetaData("convolution", 96, 11, 11, 3, 0, 0, 0), # (type, conv_n, conv_h, conv_w, conv_c, pool_h, pool_w, fully_n)
            LayerMetaData("convolution", 256, 5, 5, 96, 0, 0, 0),
            LayerMetaData("convolution", 348, 3, 3, 256, 0, 0, 0),
            LayerMetaData("convolution", 348, 3, 3, 348, 0, 0, 0),
            LayerMetaData("convolution", 256, 3, 3, 348, 0, 0, 0),
            LayerMetaData("fully",      0, 0, 0, 0, 0, 0, 4096),
            LayerMetaData("fully",      0, 0, 0, 0, 0, 0, 4096),
            LayerMetaData("fully",      0, 0, 0, 0, 0, 0, 10)
            ]


        # Input feature map
        self.input_n = 1
        self.input_h = 224
        self.input_w = 224
        self.input_c = 3
        
        # Bit width
        self.input_bit = 16
        self.filter_bit = 16