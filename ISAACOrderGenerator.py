from Model import Model
from PE import PE
from EventMetaData import EventMetaData
import numpy as np
import math
import collections

class ISAACOrderGenerator(object):
    def __init__(self, model_config, hw_config, trace):
        self.model_info = Model(model_config)

        # Hack for Controller
        self.mp_info = None

        # Setup object member
        self.model = self.model_info
        self.hw = hw_config

        self.trace = trace

        self.xbar_requirements = []
        self.calculateXbarRequirement()

        self.replicateWeight()

        self.allocateWeight()

        self.generateOrder()

        self.print_order()

    def calculateXbarRequirement(self):
        cells_per_weight = math.ceil(self.model.filter_bit / self.hw.cell_bit_width) # 16/2 = 8 cells per weight
        filters_per_xb = math.floor(self.hw.Xbar_w / cells_per_weight)
        # Initailize array to [1, 1, ..., 1], length = layer_length
        xbar_requirements = [0] * self.model.layer_length
        for i in range(self.model.layer_length):
            current_layer = self.model.layer_list[i]
            if current_layer.layer_type == "pooling":
                # TODO: Does pooling layer need Xbar?
                pass
            elif current_layer.layer_type == "convolution" or current_layer.layer_type == "fully":
                width = self.model.filter_n[i]
                height = self.model.filter_length[i]
                #print(f'{i} {current_layer.layer_type}  w:{self.model.filter_n[i]}*{cells_per_weight} h:{self.model.filter_length[i]}')
                xbar_requirements[i] = math.ceil(width / filters_per_xb) * math.ceil(height / self.hw.Xbar_h)
                #print(f'    w:{math.ceil(width / filters_per_xb)} h:{math.ceil(height / self.hw.Xbar_h)}')

        print(f"Step 1.1: Calculate Xbar requirements: {xbar_requirements}")

        self.xbar_requirements = xbar_requirements

    def to_PE_requirements(self, xbar_requirements, replicate_weight, split_input):
        replicated_xbar_requirements = np.multiply(replicate_weight, self.xbar_requirements)
        if split_input:
            replicated_xbar_requirements = np.multiply(replicated_xbar_requirements, [2])

        XBARS_PER_PE = self.hw.CU_num * self.hw.Xbar_num

        replicated_pe_requirements = list(np.ceil(np.divide(replicated_xbar_requirements, [XBARS_PER_PE])))
        return replicated_pe_requirements

    def replicateWeight(self):
        # Initailize array to [1, 1, ..., 1], length = layer_length
        replicate_weight = [1] * self.model.layer_length
        for i in reversed(range(self.model.layer_length-1)):
            replicate_weight[i] = replicate_weight[i+1]
            if self.model.layer_list[i].layer_type == "pooling":
                replicate_weight[i] *= self.model.layer_list[i].pooling_strides * self.model.layer_list[i].pooling_strides
            elif self.model.layer_list[i].layer_type == "convolution":
                replicate_weight[i] *= self.model.layer_list[i].strides * self.model.layer_list[i].strides
        print(f"Step 2.1 - balance pipeline: replicate = {replicate_weight}")

        pe_limit = self.hw.Router_num * self.hw.PE_num

        pe_requirements = self.to_PE_requirements(self.xbar_requirements, replicate_weight, False)
        pe_sum = int(np.sum(pe_requirements))
        print(f"  = PE requirements sum({pe_requirements}) = {pe_sum}   (pe_limit={pe_limit})")

        if pe_sum*2 <= pe_limit:
            # set split_input and get pe requirements
            # TODO: split input
            # pe_requirements = self.to_PE_requirements(self.xbar_requirements, replicate_weight, True)
            # pe_sum = int(np.sum(pe_requirements))
            # self.split_input = True
            self.split_input = False
        else:
            self.split_input = False
            # Fit in PEs and get pe requirements
            while(pe_sum > pe_limit):
                d = None
                for i in reversed(range(self.model.layer_length)):
                    if d == None and replicate_weight[i] > 1:
                        d = replicate_weight[i]
                    if d != None:
                        assert (replicate_weight[i] % d) == 0
                        replicate_weight[i] = replicate_weight[i] // d
                assert d != None, f'PE is not enough, require sum({pe_requirements}) = {pe_sum}, but only have {pe_limit}'
                pe_requirements = self.to_PE_requirements(self.xbar_requirements, replicate_weight, False)
                pe_sum = int(np.sum(pe_requirements))

        print(f"Step 2.2 - fit in PE limit: replicate = {replicate_weight}")
        print(f"  = PE requirements sum({pe_requirements}) = {pe_sum} (split_input={self.split_input})  (pe_limit={pe_limit})")

        self.replicate_weight = replicate_weight
        self.pe_requirements = pe_requirements

    def allocateWeight(self):
        xbar_range_used_by_weight = dict()
        xbar_idx = 0

        XBARS_PER_PE = self.hw.CU_num * self.hw.Xbar_num

        for nlayer in range(self.model.layer_length):
            current_layer = self.model.layer_list[nlayer]
            if current_layer.layer_type == "convolution" or current_layer.layer_type == "fully":
                for replicate_index in range(self.replicate_weight[nlayer]):
                    end = xbar_idx + self.xbar_requirements[nlayer]
                    if self.split_input:
                        end += self.xbar_requirements[nlayer]
                    xbar_range_used_by_weight[(nlayer, replicate_index)] = (xbar_idx, end)
                    xbar_idx = end
                # Align to PE boundary
                xbar_idx = math.ceil(xbar_idx / XBARS_PER_PE) * XBARS_PER_PE

        print(f"Step 3 - allocate xbar to weight: {{")
        for x in xbar_range_used_by_weight:
            print(f"  {x}: {xbar_range_used_by_weight[x]}")
        print('}')

        self.xbar_range_used_by_weight = xbar_range_used_by_weight

    def generateOrder(self):
        print(f"Step 4 - generate order:")
        Computation_order = []

        XBARS_PER_CU = self.hw.Xbar_num
        LOCATION_LIMIT = [self.hw.Router_num_y, self.hw.Router_num_x, self.hw.PE_num_y, self.hw.PE_num_x, self.hw.CU_num]
        def location_advance(location, addend):
            location = location.copy()
            location[4] += addend
            for i in reversed(range(5)):
                if location[i] >= LOCATION_LIMIT[i]:
                    assert i > 0
                    location[i-1] += location[i] // LOCATION_LIMIT[i]
                    location[i] = location[i] % LOCATION_LIMIT[i]
            return location

        def cu_idx_to_position(cu_idx):
            return location_advance([0, 0, 0, 0, 0], cu_idx)

        data_in_event = dict()
        for nlayer in range(self.model.layer_length):
            if self.model.layer_list[nlayer].layer_type == "convolution":
                strides = self.model.strides[nlayer]
                pad = self.model.pad[nlayer]
                o_height = self.model.input_h[nlayer+1]
                o_width = self.model.input_w[nlayer+1]

                print(f'  - {nlayer} {self.model.layer_list[nlayer].layer_type}: [{self.model.input_c[nlayer]}, {self.model.input_h[nlayer]}, {self.model.input_w[nlayer]}] x [{self.model.filter_n[nlayer]}, {self.model.filter_c[nlayer]}, {self.model.filter_h[nlayer]}, {self.model.filter_w[nlayer]}] {self.model.strides[nlayer]}, {self.model.pad[nlayer]} -> [{self.model.input_c[nlayer+1]}, {self.model.input_h[nlayer+1]}, {self.model.input_w[nlayer+1]}]')

                replicate_index = 0
                # 一個一個 window 看
                for window_h in range(o_height):
                    for window_w in range(o_width):
                        input_vector_data = []
                        for c in range(self.model.filter_c[nlayer]):
                            for h in range(self.model.filter_h[nlayer]):
                                for w in range(self.model.filter_w[nlayer]):
                                    # padding後的位置
                                    fm_h  = window_h*strides + h - pad
                                    fm_w  = window_w*strides + w - pad
                                    if fm_w >= 0 and fm_w < self.model.input_w[nlayer] and fm_h >= 0 and fm_h < self.model.input_h[nlayer]:
                                        input_vector_data.append((nlayer, fm_h, fm_w, c))
                                    else:
                                        input_vector_data.append(0) # padding的值為0
                        #print(input_vector_data)
                        num_of_inputs = len(input_vector_data)

                        # 一個 window 分配給一個 weight，因此我們就針對這個 weight 的每個 cu 去產生 event
                        # 由於在 replicate 時，我們分配的最小單位是 xbar，所以我們就要 loop 過所有 xbar 去記錄要產生的 event 的資訊
                        xbar_range = self.xbar_range_used_by_weight[(nlayer, replicate_index)]
                        # TODO: split input
                        inputs_idx_start = 0
                        inputs_count = 0
                        # TODO: 不一定可以算出全部的 output, 可能有 partial sum
                        outputs_count = self.model.filter_n[nlayer]
                        # TODO: 不一定可以算出全部的 output, 可能有 partial sum
                        outputs = [(nlayer+1, window_h, window_w, c) for c in range(self.model.filter_n[nlayer])]
                        drived_xbar = dict()
                        for xbar_idx in range(xbar_range[0], xbar_range[1]):
                            inputs_count += self.hw.Xbar_h
                            drived_xbar[xbar_idx % XBARS_PER_CU] = 1
                            # CU 的最後一個 xbar
                            if ((xbar_idx+1) % XBARS_PER_CU) == 0 or xbar_idx == (xbar_range[1] - 1):
                                inputs = None
                                if inputs_count >= num_of_inputs:
                                    inputs = input_vector_data
                                else:
                                    inputs_idx_end = (inputs_idx_start + inputs_count) % num_of_inputs
                                    if inputs_idx_end > inputs_idx_start:
                                        inputs = input_vector_data[inputs_idx_start:inputs_idx_end]
                                    else:
                                        inputs = input_vector_data[inputs_idx_start:] + input_vector_data[:inputs_idx_end]
                                    inputs_idx_start = inputs_idx_end
                                inputs = [x for x in inputs if x != 0]
                                cu_position = tuple(cu_idx_to_position(xbar_idx // XBARS_PER_CU))
                                eri_event_idx = len(Computation_order)
                                proceeding_event_count = 0
                                if nlayer != 0:
                                    # data 要從別的 PE 來，所以要看看在哪個 PE
                                    for input_position in inputs:
                                        event = data_in_event[input_position]
                                        if not eri_event_idx in event.proceeding_event:
                                            event.proceeding_event.append(eri_event_idx)
                                            event.position_idx[1] = cu_position[0:4]
                                            proceeding_event_count += 1

                                Computation_order.append(EventMetaData("edram_rd_ir", cu_position, proceeding_event_count, [eri_event_idx+1], nlayer, inputs, 0))
                                Computation_order.append(EventMetaData("cu_operation", cu_position, 1, [eri_event_idx+2], nlayer, 1, drived_xbar))
                                Computation_order.append(EventMetaData("pe_saa", cu_position[0:4], 1, [eri_event_idx+3], nlayer, outputs_count, 0))
                                # TODO: 不一定可以做 activation，可能有 partial sum
                                Computation_order.append(EventMetaData("activation", cu_position[0:4], 1, [eri_event_idx+4], nlayer, outputs_count, 0))
                                if nlayer == self.model.layer_length - 1:
                                    # 最後一層直接 edram_wr
                                    Computation_order.append(EventMetaData("edram_wr", cu_position[0:4], 1, [], nlayer, 0, outputs))
                                else:
                                    # 其他層就要傳到下一個要算的PE
                                    # TODO: 可能會有很多個要用到的 PE，要改用別的寫法
                                    event = EventMetaData("data_transfer", [cu_position[0:4], None], 1, [], nlayer, 0, outputs)
                                    Computation_order.append(event)
                                    for out in outputs:
                                        data_in_event[out] = event

                                drived_xbar = dict()
                                inputs_count = 0
                        #for cu_idx in range(math.floor(xbar_range[0] / XBARS_PER_CU), math.ceil(xbar_range[1] / XBARS_PER_CU)):

                        replicate_index = (replicate_index + 1) % self.replicate_weight[nlayer]
            elif self.model.layer_list[nlayer].layer_type == "fully":
                pass
            elif self.model.layer_list[nlayer].layer_type == "pooling":
                pass

        self.Computation_order = Computation_order

    def print_order(self):
        self.edram_rd_ir_ctr = 0
        self.cu_op_ctr  = 0
        self.pe_saa_ctr = 0
        self.activation_ctr = 0
        self.pooling_ctr = 0
        self.edram_wr_ctr = 0
        self.edram_rd_ctr = 0
        self.data_transfer_ctr = 0
        for e in self.Computation_order:
            t = e.event_type
            if t == "edram_rd_ir":
                self.edram_rd_ir_ctr += 1
            elif t == "cu_operation":
                self.cu_op_ctr += 1
            elif t == "pe_saa":
                self.pe_saa_ctr += 1
            elif t == "activation":
                self.activation_ctr += 1
            elif t == "edram_wr":
                self.edram_wr_ctr += 1
            elif t == "edram_rd":
                self.edram_rd_ctr += 1
            elif t == "pooling":
                self.pooling_ctr += 1
            elif t == "data_transfer":
                self.data_transfer_ctr += 1
            else:
                print("event type error:", t)

        print("edram_rd_ir_ctr", self.edram_rd_ir_ctr)
        print("cu_op_ctr", self.cu_op_ctr)
        print("pe_saa_ctr", self.pe_saa_ctr)
        print("activation_ctr", self.activation_ctr)
        print("edram_wr_ctr", self.edram_wr_ctr)
        print("edram_rd_ctr", self.edram_rd_ctr)
        print("data_transfer_ctr", self.data_transfer_ctr)
        print("total", len(self.Computation_order))

        if self.trace:
            layer = 0
            for e in self.Computation_order:
                if e.nlayer == layer:
                    layer += 1
                    print()
                    print("layer:", e.nlayer)
                print(self.Computation_order.index(e), e)

    def __str__(self):
        return str(self.__dict__)
