class Model(object):
    def __init__(self, model_config):
        self.Model_type = model_config.Model_type
        self.layer_list = model_config.layer_list 
        self.layer_length = len(self.layer_list)
        self.filter_n = [] 
        self.filter_h = []
        self.filter_w = []
        self.filter_c = []
        self.filter_length = []
        self.strides = []
        self.pad = []
        self.input_n = model_config.input_n
        self.input_h = [model_config.input_h] 
        self.input_w = [model_config.input_w] 
        self.input_c = [model_config.input_c] 
        self.input_number = []
        self.pooling_h = []
        self.pooling_w = []
        self.pooling_strides = []
        self.input_bit = model_config.input_bit
        self.filter_bit = model_config.filter_bit

        for nlayer in range(self.layer_length):
            if self.layer_list[nlayer].layer_type == "convolution":
                self.filter_n.append(self.layer_list[nlayer].filter_n)
                self.filter_h.append(self.layer_list[nlayer].filter_h)
                self.filter_w.append(self.layer_list[nlayer].filter_w)
                self.filter_c.append(self.layer_list[nlayer].filter_c)
                self.filter_length.append(
                    self.layer_list[nlayer].filter_h *
                    self.layer_list[nlayer].filter_w *
                    self.layer_list[nlayer].filter_c
                    )
                self.strides.append(self.layer_list[nlayer].strides)
                if self.layer_list[nlayer].padding == 'VALID':
                    self.pad.append(0)
                elif self.layer_list[nlayer].padding == 'SAME':
                    self.pad.append((self.layer_list[nlayer].filter_h - 1) // 2)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.pooling_strides.append(0)
                self.input_h.append((self.input_h[nlayer] + 2 * self.pad[nlayer] - self.filter_h[nlayer])
                                     // self.strides[nlayer] + 1)
                self.input_w.append((self.input_w[nlayer] + 2 * self.pad[nlayer] - self.filter_w[nlayer])
                                     // self.strides[nlayer] + 1)
                self.input_c.append(self.layer_list[nlayer].filter_n)
                self.input_number.append(self.input_h[nlayer+1] * self.input_w[nlayer+1])
            elif self.layer_list[nlayer].layer_type == "pooling":
                self.filter_n.append(0)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(0)
                self.strides.append(0)
                self.pad.append(0)
                #self.input_h.append(self.input_h[nlayer] // self.layer_list[nlayer].pooling_h) # 沒有stride
                #self.input_w.append(self.input_w[nlayer] // self.layer_list[nlayer].pooling_w) # 沒有stride
                self.pooling_h.append(self.layer_list[nlayer].pooling_h)
                self.pooling_w.append(self.layer_list[nlayer].pooling_w)
                self.pooling_strides.append(self.layer_list[nlayer].pooling_strides)
                self.input_h.append((self.input_h[nlayer] - self.layer_list[nlayer].pooling_h)
                                    // self.pooling_strides[nlayer] + 1)
                self.input_w.append((self.input_w[nlayer] - self.layer_list[nlayer].pooling_w)
                                    // self.pooling_strides[nlayer] + 1)
                self.input_c.append(self.input_c[nlayer])
                self.input_number.append(self.input_h[nlayer+1] * self.input_w[nlayer+1] * self.input_c[nlayer+1])
            elif self.layer_list[nlayer].layer_type == "fully":
                self.filter_n.append(self.layer_list[nlayer].neuron_n)
                self.filter_h.append(1)
                self.filter_w.append(1)
                self.filter_c.append(1)
                self.filter_length.append(self.input_h[nlayer] * self.input_w[nlayer] * self.input_c[nlayer])
                self.strides.append(0)
                self.pad.append(0)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.pooling_strides.append(0)
                self.input_h.append(self.layer_list[nlayer].neuron_n)
                self.input_w.append(1)
                self.input_c.append(1)
                self.input_number.append(1)

    def __str__(self):
        return str(self.__dict__)
