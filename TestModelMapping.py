from MappingMetaData import MappingMetaData
from CrossbarGridMetaDate import CrossbarGridMetaDate
import sys

class TestModelMappingA(object):
    def __init__(self, model_info, hd_info):
        self.crossbar_array = []
        for i in range(hd_info.CU_num):
            self.crossbar_array.append([])
            for j in range(hd_info.xbar_h):
                row = [0] * hd_info.xbar_w
                self.crossbar_array[i].append(row)

        self.column_index = [0] * hd_info.CU_num

        for nlayer in range(len(model_info.layer_list)):
            if model_info.layer_list[nlayer].layer_type == "convolution":
                for nfilter in range(model_info.layer_list[nlayer].filter_n):
                    for nbit in range(hd_info.filter_bit):
                        for c in range(model_info.layer_list[nlayer].filter_c):
                            for h in range(model_info.layer_list[nlayer].filter_h):
                                for w in range(model_info.layer_list[nlayer].filter_w):
                                    x = c * model_info.layer_list[nlayer].filter_h * model_info.layer_list[nlayer].filter_w + h * model_info.layer_list[nlayer].filter_w + w
                                    self.crossbar_array[nlayer//2][x][self.column_index[nlayer//2]] = CrossbarGridMetaDate(nlayer, nfilter, x, nbit)
                        self.column_index[nlayer//2] += 1
            elif model_info.layer_list[nlayer].layer_type == "pooling":
                pass
            elif model_info.layer_list[nlayer].layer_type == "fully":
                for nfilter in range(model_info.layer_list[nlayer+1].neuron_n):
                    for nbit in range(hd_info.filter_bit):
                        for h in range(model_info.layer_list[nlayer].neuron_n):
                            if h >= hd_info.xbar_h:
                                print("not enough crossbar row")
                                sys.exit()

                            self.crossbar_array[nlayer%2][h][self.column_index[nlayer%2]] = CrossbarGridMetaDate(nlayer, nfilter, h, nbit)
                        self.column_index[nlayer%2] += 1
                    if self.column_index[nlayer%2] >= hd_info.xbar_w:
                        print("not enough crossbar column")
                        sys.exit()
                        

        self.NN_layer = len(model_info.layer_list)
        layer_list = model_info.layer_list

        self.input_h = [model_info.input_h]
        self.input_w = [model_info.input_w]
        self.input_c = [model_info.input_c]
        self.input_number = []
        self.filter_n = []
        self.filter_h = []
        self.filter_w = []
        self.filter_c = []
        self.filter_length = []
        self.pooling_h = []
        self.pooling_w = []

        for i in range(self.NN_layer):
            if layer_list[i].layer_type == "convolution":
                #self.conv_layer.append(i)
                self.filter_n.append(layer_list[i].filter_n)
                self.filter_h.append(layer_list[i].filter_h)
                self.filter_w.append(layer_list[i].filter_w)
                self.filter_c.append(layer_list[i].filter_c)
                self.filter_length.append(layer_list[i].filter_h * layer_list[i].filter_w * layer_list[i].filter_c)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.input_h[i] - layer_list[i].filter_h + 1)
                self.input_w.append(self.input_w[i] - layer_list[i].filter_w + 1)
                self.input_c.append(layer_list[i].filter_n)
                self.input_number.append((self.input_h[i] - layer_list[i].filter_h + 1) * (self.input_w[i] - layer_list[i].filter_w + 1))
                #self.weight_mat.append(np.zeros((self.input_number[i], self.filter_bit, layer_list[i].filter_h * layer_list[i].filter_w * layer_list[i].filter_c, layer_list[i].filter_h * layer_list[i].filter_w * layer_list[i].filter_c * self.filter_bit)))

            elif layer_list[i].layer_type == "pooling":
                #self.pooling_layer.append(i)
                self.filter_n.append(0)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(0)
                self.pooling_h.append(layer_list[i].pooling_h)
                self.pooling_w.append(layer_list[i].pooling_w)
                self.input_h.append(self.input_h[i] // layer_list[i].pooling_h)
                self.input_w.append(self.input_w[i] // layer_list[i].pooling_w)
                self.input_c.append(self.input_c[i])
                self.input_number.append((self.input_h[i] // layer_list[i].pooling_h) * (self.input_w[i] // layer_list[i].pooling_w) * (self.input_c[i]))
                #self.weight_mat.append([])
            elif layer_list[i].layer_type == "fully":
                #self.pooling_layer.append(i)
                self.filter_n.append(layer_list[i+1].neuron_n)
                self.filter_h.append(layer_list[i].neuron_n)
                self.filter_w.append(1)
                self.filter_c.append(1)
                self.filter_length.append(layer_list[i].neuron_n)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.filter_n[i])
                self.input_w.append(1)
                self.input_c.append(1)
                self.input_number.append(1)

            elif layer_list[i].layer_type == "output":
                self.filter_n.append(0)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(0)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.filter_n[i])
                self.input_w.append(1)
                self.input_c.append(1)
                self.input_number.append(1)


        
        self.layer_mapping_to_CU = []
        for i in range(hd_info.CU_num):
            self.layer_mapping_to_CU.append([])

        height = self.input_h[0] - self.filter_h[0] + 1
        width = self.input_w[0] - self.filter_w[0] + 1
        inputs = []
        for i in range(height):
            for j in range(width):
                nn = []
                for c in range(model_info.layer_list[0].filter_c):
                    for k in range(model_info.layer_list[0].filter_h):
                        for l in range(model_info.layer_list[0].filter_w):
                            nn.append([i+k, j+l, c])
                inputs.append(nn)
        xbar_column = []
        for column in range(hd_info.xbar_w):
            include = False
            for i in range(hd_info.xbar_h):
                if isinstance(self.crossbar_array[0][i][column], CrossbarGridMetaDate):
                    if self.crossbar_array[0][i][column].nlayer == 0 and  self.crossbar_array[0][i][column].nfilter in range(model_info.layer_list[0].filter_n):
                        include = True
                        break
            if include:
                xbar_column.append(column)
        self.layer_mapping_to_CU[0].append(MappingMetaData("convolution", 0, xbar_column, inputs))

        
        height = self.input_h[1] // self.pooling_h[1]
        width = self.input_w[1] // self.pooling_w[1]
        inputs = []
        for i in range(height):
            for j in range(width):
                for c in range(model_info.layer_list[0].filter_n):
                    nn = []
                    for k in range(model_info.layer_list[1].pooling_h):
                        for l in range(model_info.layer_list[1].pooling_w):
                            nn.append([i*2 + k, j*2 + l, c])
                    inputs.append(nn)
        self.layer_mapping_to_CU[0].append(MappingMetaData("pooling", 1, [-1], inputs))


        height = self.input_h[2] - self.filter_h[2] + 1
        width = self.input_w[2] - self.filter_w[2] + 1
        inputs = []
        for i in range(height):
            for j in range(width):
                nn = []
                for c in range(model_info.layer_list[2].filter_c):
                    for k in range(model_info.layer_list[2].filter_h):
                        for l in range(model_info.layer_list[2].filter_w):
                            nn.append([i+k, j+l, c])
                inputs.append(nn)
        xbar_column = []
        for column in range(hd_info.xbar_w):
            include = False
            for i in range(hd_info.xbar_h):
                if isinstance(self.crossbar_array[1][i][column], CrossbarGridMetaDate):
                    if self.crossbar_array[1][i][column].nlayer == 2 and  self.crossbar_array[1][i][column].nfilter in range(model_info.layer_list[2].filter_n):
                        include = True
                        break
            if include:
                xbar_column.append(column)
        self.layer_mapping_to_CU[1].append(MappingMetaData("convolution", 2, xbar_column, inputs))

        height = self.input_h[3] // self.pooling_h[3]
        width = self.input_w[3] // self.pooling_w[3]
        inputs = []
        for i in range(height):
            for j in range(width):
                for c in range(model_info.layer_list[2].filter_n):
                    nn = []
                    for k in range(model_info.layer_list[3].pooling_h):
                        for l in range(model_info.layer_list[3].pooling_w):
                            nn.append([i*2 + k, j*2 + l, c])
                    inputs.append(nn)
        self.layer_mapping_to_CU[1].append(MappingMetaData("pooling", 3, [-1], inputs))

        inputs = []
        nn = []
        for h in range(self.filter_h[4]):
            nn.append([h, 0, 0])
        inputs.append(nn)

        xbar_column = []
        for column in range(hd_info.xbar_w):
            include = False
            for i in range(hd_info.xbar_h):
                if isinstance(self.crossbar_array[0][i][column], CrossbarGridMetaDate):
                    if self.crossbar_array[0][i][column].nlayer == 4 and  self.crossbar_array[0][i][column].nfilter in range(model_info.layer_list[5].neuron_n):
                        include = True
                        break
            if include:
                xbar_column.append(column)
        self.layer_mapping_to_CU[0].append(MappingMetaData("fully", 4, xbar_column, inputs))

        
        inputs = []
        nn = []
        for h in range(self.filter_h[5]):
            nn.append([h, 0, 0])
        inputs.append(nn)

        xbar_column = []
        for column in range(hd_info.xbar_w):
            include = False
            for i in range(hd_info.xbar_h):
                if isinstance(self.crossbar_array[1][i][column], CrossbarGridMetaDate):
                    if self.crossbar_array[1][i][column].nlayer == 5 and  self.crossbar_array[1][i][column].nfilter in range(model_info.layer_list[6].neuron_n):
                        include = True
                        break
            if include:
                xbar_column.append(column)
        self.layer_mapping_to_CU[1].append(MappingMetaData("fully", 5, xbar_column, inputs))
        
        

    def __str__(self):
        return str(self.__dict__)


class TestModelMappingB(object):
    def __init__(self, model_info, hd_info):
        self.crossbar_array = []
        for i in range(hd_info.CU_num):
            self.crossbar_array.append([])
            for j in range(hd_info.xbar_h):
                row = [0] * hd_info.xbar_w
                self.crossbar_array[i].append(row)

        self.column_index = [0] * hd_info.CU_num

        for nlayer in range(len(model_info.layer_list)):
            if model_info.layer_list[nlayer].layer_type == "convolution":
                for nfilter in range(model_info.layer_list[nlayer].filter_n):
                    # CU 0 :conv0,f0, conv2,f0
                    # CU 1 :conv0,f1, conv2,f1
                    cu = nfilter % 2
                    for nbit in range(hd_info.filter_bit):
                        for c in range(model_info.layer_list[nlayer].filter_c):
                            for h in range(model_info.layer_list[nlayer].filter_h):
                                for w in range(model_info.layer_list[nlayer].filter_w):
                                    x = c * model_info.layer_list[nlayer].filter_h * model_info.layer_list[nlayer].filter_w + h * model_info.layer_list[nlayer].filter_w + w
                                    self.crossbar_array[cu][x][self.column_index[cu]] = CrossbarGridMetaDate(nlayer, nfilter, x, nbit)
                        self.column_index[cu] += 1
            elif model_info.layer_list[nlayer].layer_type == "pooling":
                pass
            elif model_info.layer_list[nlayer].layer_type == "fully":
                for nfilter in range(model_info.layer_list[nlayer+1].neuron_n):
                    for nbit in range(hd_info.filter_bit):
                        for h in range(model_info.layer_list[nlayer].neuron_n):
                            if h >= hd_info.xbar_h:
                                print("not enough crossbar row")
                                sys.exit()

                            self.crossbar_array[nlayer%2][h][self.column_index[nlayer%2]] = CrossbarGridMetaDate(nlayer, nfilter, h, nbit)
                        self.column_index[nlayer%2] += 1
                    if self.column_index[nlayer%2] >= hd_info.xbar_w:
                        print("not enough crossbar column")
                        sys.exit()

        self.NN_layer = len(model_info.layer_list)
        layer_list = model_info.layer_list

        self.input_h = [model_info.input_h]
        self.input_w = [model_info.input_w]
        self.input_c = [model_info.input_c]
        self.input_number = []
        self.filter_n = []
        self.filter_h = []
        self.filter_w = []
        self.filter_c = []
        self.filter_length = []
        self.pooling_h = []
        self.pooling_w = []

        for i in range(self.NN_layer):
            if layer_list[i].layer_type == "convolution":
                #self.conv_layer.append(i)
                self.filter_n.append(layer_list[i].filter_n)
                self.filter_h.append(layer_list[i].filter_h)
                self.filter_w.append(layer_list[i].filter_w)
                self.filter_c.append(layer_list[i].filter_c)
                self.filter_length.append(layer_list[i].filter_h * layer_list[i].filter_w * layer_list[i].filter_c)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.input_h[i] - layer_list[i].filter_h + 1)
                self.input_w.append(self.input_w[i] - layer_list[i].filter_w + 1)
                self.input_c.append(layer_list[i].filter_n)
                self.input_number.append((self.input_h[i] - layer_list[i].filter_h + 1) * (self.input_w[i] - layer_list[i].filter_w + 1))
                #self.weight_mat.append(np.zeros((self.input_number[i], self.filter_bit, layer_list[i].filter_h * layer_list[i].filter_w * layer_list[i].filter_c, layer_list[i].filter_h * layer_list[i].filter_w * layer_list[i].filter_c * self.filter_bit)))

            elif layer_list[i].layer_type == "pooling":
                #self.pooling_layer.append(i)
                self.filter_n.append(0)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(0)
                self.pooling_h.append(layer_list[i].pooling_h)
                self.pooling_w.append(layer_list[i].pooling_w)
                self.input_h.append(self.input_h[i] // layer_list[i].pooling_h)
                self.input_w.append(self.input_w[i] // layer_list[i].pooling_w)
                self.input_c.append(self.input_c[i])
                self.input_number.append((self.input_h[i] // layer_list[i].pooling_h) * (self.input_w[i] // layer_list[i].pooling_w) * (self.input_c[i]))
                #self.weight_mat.append([])
            elif layer_list[i].layer_type == "fully":
                #self.pooling_layer.append(i)
                self.filter_n.append(layer_list[i+1].neuron_n)
                self.filter_h.append(layer_list[i].neuron_n)
                self.filter_w.append(1)
                self.filter_c.append(1)
                self.filter_length.append(layer_list[i].neuron_n)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.filter_n[i])
                self.input_w.append(1)
                self.input_c.append(1)
                self.input_number.append(1)

            elif layer_list[i].layer_type == "output":
                self.filter_n.append(0)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(0)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.filter_n[i])
                self.input_w.append(1)
                self.input_c.append(1)
                self.input_number.append(1)


        
        self.layer_mapping_to_CU = []
        for i in range(hd_info.CU_num):
            self.layer_mapping_to_CU.append([])

        height = self.input_h[0] - self.filter_h[0] + 1
        width = self.input_w[0] - self.filter_w[0] + 1
        inputs = []
        for i in range(height):
            for j in range(width):
                nn = []
                for c in range(model_info.layer_list[0].filter_c):
                    for k in range(model_info.layer_list[0].filter_h):
                        for l in range(model_info.layer_list[0].filter_w):
                            nn.append([i+k, j+l, c])
                inputs.append(nn)
        xbar_column = []
        for column in range(hd_info.xbar_w):
            include = False
            for i in range(hd_info.xbar_h):
                if isinstance(self.crossbar_array[0][i][column], CrossbarGridMetaDate):
                    if self.crossbar_array[0][i][column].nlayer == 0 and  self.crossbar_array[0][i][column].nfilter in range(model_info.layer_list[0].filter_n):
                        include = True
                        break
            if include:
                xbar_column.append(column)
        self.layer_mapping_to_CU[0].append(MappingMetaData("convolution", 0, xbar_column, inputs))

        inputs = []
        for i in range(height):
            for j in range(width):
                nn = []
                for c in range(model_info.layer_list[0].filter_c):
                    for k in range(model_info.layer_list[0].filter_h):
                        for l in range(model_info.layer_list[0].filter_w):
                            nn.append([i+k, j+l, c])
                inputs.append(nn)
        xbar_column = []
        for column in range(hd_info.xbar_w):
            include = False
            for i in range(hd_info.xbar_h):
                if isinstance(self.crossbar_array[1][i][column], CrossbarGridMetaDate):
                    if self.crossbar_array[1][i][column].nlayer == 0 and  self.crossbar_array[1][i][column].nfilter in range(model_info.layer_list[0].filter_n):
                        include = True
                        break
            if include:
                xbar_column.append(column)
        self.layer_mapping_to_CU[1].append(MappingMetaData("convolution", 0, xbar_column, inputs))

        
        # CU 0 do half of pooling1 (channel 0), half of pooling3 (channel 0)
        # CU 1 do half of pooling1 (channel 1), half of pooling3 (channel 1)
        height = self.input_h[1] // self.pooling_h[1]
        width = self.input_w[1] // self.pooling_w[1]
        for c in range(model_info.layer_list[0].filter_n):
            inputs = []
            for i in range(height):
                for j in range(width):
                    nn = []
                    for k in range(model_info.layer_list[1].pooling_h):
                        for l in range(model_info.layer_list[1].pooling_w):
                            nn.append([i*2 + k, j*2 + l, c])
                    inputs.append(nn)
            self.layer_mapping_to_CU[c%2].append(MappingMetaData("pooling", 1, [-1], inputs))


        height = self.input_h[2] - self.filter_h[2] + 1
        width = self.input_w[2] - self.filter_w[2] + 1
        inputs = []
        for i in range(height):
            for j in range(width):
                nn = []
                for c in range(model_info.layer_list[2].filter_c):
                    for k in range(model_info.layer_list[2].filter_h):
                        for l in range(model_info.layer_list[2].filter_w):
                            nn.append([i+k, j+l, c])
                inputs.append(nn)
        xbar_column = []
        for column in range(hd_info.xbar_w):
            include = False
            for i in range(hd_info.xbar_h):
                if isinstance(self.crossbar_array[0][i][column], CrossbarGridMetaDate):
                    if self.crossbar_array[0][i][column].nlayer == 2 and  self.crossbar_array[0][i][column].nfilter in range(model_info.layer_list[2].filter_n):
                        include = True
                        break
            if include:
                xbar_column.append(column)
        self.layer_mapping_to_CU[0].append(MappingMetaData("convolution", 2, xbar_column, inputs))

        inputs = []
        for i in range(height):
            for j in range(width):
                nn = []
                for c in range(model_info.layer_list[2].filter_c):
                    for k in range(model_info.layer_list[2].filter_h):
                        for l in range(model_info.layer_list[2].filter_w):
                            nn.append([i+k, j+l, c])
                inputs.append(nn)
        xbar_column = []
        for column in range(hd_info.xbar_w):
            include = False
            for i in range(hd_info.xbar_h):
                if isinstance(self.crossbar_array[1][i][column], CrossbarGridMetaDate):
                    if self.crossbar_array[1][i][column].nlayer == 2 and  self.crossbar_array[1][i][column].nfilter in range(model_info.layer_list[2].filter_n):
                        include = True
                        break
            if include:
                xbar_column.append(column)
        self.layer_mapping_to_CU[1].append(MappingMetaData("convolution", 2, xbar_column, inputs))


        # CU 0 do half of pooling1 (channel 0), half of pooling3 (channel 0)
        # CU 1 do half of pooling1 (channel 1), half of pooling3 (channel 1)
        height = self.input_h[3] // self.pooling_h[3]
        width = self.input_w[3] // self.pooling_w[3]
        for c in range(model_info.layer_list[2].filter_n):
            inputs = []
            for i in range(height):
                for j in range(width):
                    nn = []
                    for k in range(model_info.layer_list[3].pooling_h):
                        for l in range(model_info.layer_list[3].pooling_w):
                            nn.append([i*2 + k, j*2 + l, c])
                    inputs.append(nn)
            self.layer_mapping_to_CU[c%2].append(MappingMetaData("pooling", 3, [-1], inputs))
        
        inputs = []
        nn = []
        for h in range(self.filter_h[4]):
            nn.append([h, 0, 0])
        inputs.append(nn)

        xbar_column = []
        for column in range(hd_info.xbar_w):
            include = False
            for i in range(hd_info.xbar_h):
                if isinstance(self.crossbar_array[0][i][column], CrossbarGridMetaDate):
                    if self.crossbar_array[0][i][column].nlayer == 4 and  self.crossbar_array[0][i][column].nfilter in range(model_info.layer_list[5].neuron_n):
                        include = True
                        break
            if include:
                xbar_column.append(column)
        self.layer_mapping_to_CU[0].append(MappingMetaData("fully", 4, xbar_column, inputs))

        
        inputs = []
        nn = []
        for h in range(self.filter_h[5]):
            nn.append([h, 0, 0])
        inputs.append(nn)

        xbar_column = []
        for column in range(hd_info.xbar_w):
            include = False
            for i in range(hd_info.xbar_h):
                if isinstance(self.crossbar_array[1][i][column], CrossbarGridMetaDate):
                    if self.crossbar_array[1][i][column].nlayer == 5 and  self.crossbar_array[1][i][column].nfilter in range(model_info.layer_list[6].neuron_n):
                        include = True
                        break
            if include:
                xbar_column.append(column)
        self.layer_mapping_to_CU[1].append(MappingMetaData("fully", 5, xbar_column, inputs))
        

    def __str__(self):
        return str(self.__dict__)