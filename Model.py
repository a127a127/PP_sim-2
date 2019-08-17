class Model(object):
    def __init__(self, model_config):
        self.layer_list = model_config.layer_list 
        self.layer_length = len(self.layer_list)
        self.input_n = model_config.input_n
        self.input_h = [model_config.input_h] 
        self.input_w = [model_config.input_w] 
        self.input_c = [model_config.input_c] 
        self.input_number = [] # input windows number
        self.filter_n = [] 
        self.filter_h = []
        self.filter_w = []
        self.filter_c = []
        self.filter_length = []
        self.pooling_h = []
        self.pooling_w = []

        self.input_bit = model_config.input_bit
        self.filter_bit = model_config.input_bit

        for nlayer in range(self.layer_length):
            if self.layer_list[nlayer].layer_type == "convolution":
                self.filter_n.append(self.layer_list[nlayer].filter_n)
                self.filter_h.append(self.layer_list[nlayer].filter_h)
                self.filter_w.append(self.layer_list[nlayer].filter_w)
                self.filter_c.append(self.layer_list[nlayer].filter_c)
                self.filter_length.append(self.layer_list[nlayer].filter_h * self.layer_list[nlayer].filter_w * self.layer_list[nlayer].filter_c)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.input_h[nlayer] - self.layer_list[nlayer].filter_h + 1) # stride = 1
                self.input_w.append(self.input_w[nlayer] - self.layer_list[nlayer].filter_w + 1) # stride = 1
                self.input_c.append(self.layer_list[nlayer].filter_n)
                self.input_number.append((self.input_h[nlayer] - self.layer_list[nlayer].filter_h + 1) * (self.input_w[nlayer] - self.layer_list[nlayer].filter_w + 1))
            elif self.layer_list[nlayer].layer_type == "pooling":
                self.filter_n.append(0)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(0)
                self.pooling_h.append(self.layer_list[nlayer].pooling_h)
                self.pooling_w.append(self.layer_list[nlayer].pooling_w)
                self.input_h.append(self.input_h[nlayer] // self.layer_list[nlayer].pooling_h)
                self.input_w.append(self.input_w[nlayer] // self.layer_list[nlayer].pooling_w)
                self.input_c.append(self.input_c[nlayer])
                self.input_number.append((self.input_h[nlayer] // self.layer_list[nlayer].pooling_h) * (self.input_w[nlayer] // self.layer_list[nlayer].pooling_w) * (self.input_c[nlayer]))
            elif self.layer_list[nlayer].layer_type == "fully":
                self.filter_n.append(self.layer_list[nlayer].neuron_n)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(self.input_h[nlayer] * self.input_w[nlayer] * self.input_c[nlayer])
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.layer_list[nlayer].neuron_n)
                self.input_w.append(1)
                self.input_c.append(1)
                self.input_number.append(self.layer_list[nlayer].neuron_n)

    def __str__(self):
        return str(self.__dict__)
