class ModelMetaData(object):
    def __init__(self, layer_list,  input_h, input_w, input_c, input_n, input_bit, filter_bit):
        # for layer_info
        self.layer_list = layer_list
        # for input information
        self.input_n = input_n
        self.input_h = input_h
        self.input_w = input_w
        self.input_c = input_c

        # bit width
        self.input_bit = input_bit
        self.filter_bit = filter_bit

    def __str__(self):
        return str(self.__dict__)