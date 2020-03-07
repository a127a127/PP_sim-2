from HardwareMetaData import HardwareMetaData as HW
from ModelConfig import ModelConfig
from Model import Model
from FreeBufferController import FreeBufferController
from PE import PE
from EventMetaData import EventMetaData
import numpy as np
import math
class OrderGenerator(object):
    def __init__(self, mapping_information, trace):
        model_config = ModelConfig()
        self.model_info = Model(model_config)
        self.mp_info = mapping_information
        self.free_buffer_controller = FreeBufferController()

        self.cu_traverse_idx = []
        for rty_idx in range(HW().Router_num_y):
            for rtx_idx in range(HW().Router_num_x):
                for pey_idx in range(HW().PE_num_y):
                    for pex_idx in range(HW().PE_num_x):
                        for cuy_idx in range(HW().CU_num_y):
                            for cux_idx in range(HW().CU_num_x):
                                cu_pos = (rty_idx, rtx_idx, pey_idx, pex_idx, cuy_idx, cux_idx)
                                self.cu_traverse_idx.append(cu_pos)

        self.pe_saa_mat   = [] # for same layer data transfer
        for i in range(self.model_info.layer_length):
            if self.model_info.layer_list[i].layer_type == "convolution":
                self.pe_saa_mat.append(np.zeros((self.model_info.input_number[i], self.model_info.filter_n[i])).tolist())
            elif self.model_info.layer_list[i].layer_type == "fully":
                self.pe_saa_mat.append(np.zeros((self.model_info.input_number[i], self.model_info.filter_n[i])).tolist())
            elif self.model_info.layer_list[i].layer_type == "pooling":
                self.pe_saa_mat.append([])


        self.transfer_mat = [] # for different layer data transfer (output feature map, 最後一層沒有)
        for i in range(self.model_info.layer_length):
            if self.model_info.layer_list[i].layer_type == "convolution" or self.model_info.layer_list[i].layer_type == "pooling":
                if i+1 < self.model_info.layer_length:
                    if self.model_info.layer_list[i+1].layer_type == "fully":
                        arr = []
                        for h in range(self.model_info.input_h[i+1] * self.model_info.input_w[i+1] * self.model_info.input_c[i+1]):
                            arr.append(set())
                    else:
                        arr = []
                        for h in range(self.model_info.input_h[i+1]):
                            arr.append([])
                            for w in range(self.model_info.input_w[i+1]):
                                arr[h].append([])
                                for c in range(self.model_info.input_c[i+1]):
                                    arr[h][w].append(set())
                    self.transfer_mat.append(arr)

            elif self.model_info.layer_list[i].layer_type == "fully":
                if i+1 < self.model_info.layer_length:
                    arr = []
                    for h in range(self.model_info.filter_n[i]):
                        arr.append(set())
                    self.transfer_mat.append(arr)

        ## 每個feature map data會被哪些PE用到
        # layer_mapping_to_xbar
        for cu_pos in self.cu_traverse_idx:
            pe_pos = cu_pos[0:4]
            rty_idx, rtx_idx = cu_pos[0], cu_pos[1]
            pey_idx, pex_idx = cu_pos[2], cu_pos[3]
            cuy_idx, cux_idx = cu_pos[4], cu_pos[5]
            for xby_idx in range(HW().Xbar_num_y):
                for xbx_idx in range(HW().Xbar_num_x):
                    for nlayer in range(self.model_info.layer_length):
                        if nlayer == 0:
                            continue
                        else:
                            xbar_inputs = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][nlayer]
                            if self.model_info.layer_list[nlayer].padding == "SAME":
                                for x_inp in xbar_inputs:
                                    for inp in x_inp.inputs:
                                        h = inp[1] - self.model_info.pad[nlayer]
                                        w = inp[2] - self.model_info.pad[nlayer]
                                        c = inp[3]
                                        if w >= 0 and w < self.model_info.input_w[nlayer] and h >= 0 and h < self.model_info.input_h[nlayer]:
                                            if self.model_info.layer_list[nlayer].layer_type != "fully":
                                                self.transfer_mat[nlayer-1][h][w][c].add(pe_pos)
                                            else:
                                                self.transfer_mat[nlayer-1][h].add(pe_pos)
                            else:
                                for x_inp in xbar_inputs:
                                    for inp in x_inp.inputs:
                                        h, w, c = inp[1], inp[2], inp[3] # num_inp不需要
                                        if self.model_info.layer_list[nlayer].layer_type != "fully":
                                            self.transfer_mat[nlayer-1][h][w][c].add(pe_pos)
                                        else:
                                            self.transfer_mat[nlayer-1][h].add(pe_pos)
        # layer_mapping_to_pe
        for rty_idx in range(HW().Router_num_y):
            for rtx_idx in range(HW().Router_num_x):
                for pey_idx in range(HW().PE_num_y):
                    for pex_idx in range(HW().PE_num_x):
                        pe_pos = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        for nlayer in range(self.model_info.layer_length):
                            if nlayer == 0:
                                continue
                            else:
                                pe_inputs = self.mp_info.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx][nlayer]
                                for pe_inp in pe_inputs:
                                    for inp in pe_inp:
                                        for data in inp:
                                            data = data[2:]
                                            if self.model_info.layer_list[nlayer].layer_type != "fully":
                                                self.transfer_mat[nlayer-1][data[0]][data[1]][data[2]].add(pe_pos)
                                            else:
                                                self.transfer_mat[nlayer-1][data[0]].add(pe_pos)

        self.Computation_order = []
        self.generate_order()
        if trace:
            self.trace_order()

    def generate_order(self):
        for nlayer in range(self.model_info.layer_length):
            print("Generate layer", nlayer, self.model_info.layer_list[nlayer].layer_type)
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                ### Event: edram_rd_ir
                for cu_pos in self.cu_traverse_idx:
                    pe_pos = cu_pos[:-2]
                    rty_idx, rtx_idx= cu_pos[0], cu_pos[1]
                    pey_idx, pex_idx = cu_pos[2], cu_pos[3]
                    cuy_idx, cux_idx = cu_pos[4], cu_pos[5]

                    # 算CU內的所有XB，最多需要多少組input
                    max_xb_input_len = 0
                    for xby_idx in range(HW().Xbar_num_y):
                        for xbx_idx in range(HW().Xbar_num_x):
                            xbar_inputs = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][nlayer]
                            num_inp = len(xbar_inputs)
                            max_xb_input_len = max(max_xb_input_len, num_inp)

                    # 每一組input產生一個edram read event
                    for nInp in range(max_xb_input_len):
                        data_feed_to_cu = [] # 一次edram read的資料
                        for xby_idx in range(HW().Xbar_num_y):
                            for xbx_idx in range(HW().Xbar_num_x):
                                xbar_inputs = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][nlayer]
                                if nInp < len(xbar_inputs):
                                    inp = xbar_inputs[nInp].inputs
                                else:
                                    inp = []

                                # inp紀錄的位置為padding後的位置
                                if self.model_info.layer_list[nlayer].padding == "SAME":
                                    for d in inp:
                                        h = d[1] - self.model_info.pad[nlayer]
                                        w = d[2] - self.model_info.pad[nlayer]
                                        c = d[3]
                                        # 不紀錄pading的值
                                        if w >= 0 and w < self.model_info.input_w[nlayer] and h >= 0 and h < self.model_info.input_h[nlayer]:
                                            data = (nlayer, h, w, c)
                                            if data not in data_feed_to_cu:
                                                data_feed_to_cu.append(data)
                                else: # padding == "VALID"
                                    for d in inp:
                                        data = [nlayer] + d[1:]
                                        data = tuple(data)
                                        if data not in data_feed_to_cu:
                                            data_feed_to_cu.append(data)

                        eri_event_idx = len(self.Computation_order)
                        if nlayer == 0:
                            eri_preceding_count = 0
                        else:
                            eri_preceding_count = len(data_feed_to_cu)
                            # add dependency
                            for data in data_feed_to_cu:
                                pre_event_idx = self.transfer_mat[nlayer-1][data[1]][data[2]][data[3]][pe_pos]
                                self.Computation_order[pre_event_idx].proceeding_event.append(eri_event_idx)
                        eri_position_idx = cu_pos
                        eri_input_sequence  = data_feed_to_cu
                        eri_output_sequence = 0
                        event = EventMetaData("edram_rd_ir", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_input_sequence, eri_output_sequence)
                        self.Computation_order.append(event)

                        # input requirement
                        pe_id = cu_pos[3] + cu_pos[2]*HW().PE_num_x + \
                                cu_pos[1]*HW().PE_num + cu_pos[0]*HW().PE_num*HW().Router_num_x
                        for d in data_feed_to_cu:
                            pos = d[2] + d[1]*self.model_info.input_w[nlayer] + d[3]*self.model_info.input_w[nlayer]*self.model_info.input_h[nlayer] # w + h*width + c*height*width
                            self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1

                ### Event: cu_operation
                        cu_operation_idx = len(self.Computation_order)
                        num_ou_in_xb = dict()
                        max_ou = 0
                        for xby_idx in range(HW().Xbar_num_y):
                            for xbx_idx in range(HW().Xbar_num_x):
                                xbar_inputs  = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][nlayer]
                                if nInp < len(xbar_inputs):
                                    inp = xbar_inputs[nInp]
                                else:
                                    continue

                                # Total ou in the event
                                num_ou_h = math.ceil(len(inp.inputs)/ HW().OU_h)
                                num_ou_w = math.ceil(inp.Cols  / HW().OU_w)
                                num_ou = num_ou_h * num_ou_w * self.model_info.input_bit

                                xb_idx = xbx_idx + xby_idx * HW().Xbar_num_x
                                num_ou_in_xb[xb_idx] = num_ou
                                max_ou = max(num_ou, max_ou)

                                num_input = inp.inputs[0][0]

                                filter_list = inp.Filters # 此xbar 會計算到的 filter
                                for nfilter in filter_list:
                                    grid = self.pe_saa_mat[nlayer][num_input][nfilter]
                                    if grid == 0.0:
                                        self.pe_saa_mat[nlayer][num_input][nfilter] = []
                                    if cu_operation_idx not in self.pe_saa_mat[nlayer][num_input][nfilter]:
                                        self.pe_saa_mat[nlayer][num_input][nfilter].append(cu_operation_idx)
                        
                        position_idx = cu_pos
                        preceding_count = 1
                        cu_op_inputs  = max_ou # 最多要幾次ou做完
                        cu_op_outputs = num_ou_in_xb
                        event = EventMetaData("cu_operation", position_idx, preceding_count, [], nlayer, cu_op_inputs, cu_op_outputs)
                        self.Computation_order.append(event)

                ### Event: edram_wr, data_transfer (for pe_saa), pe_saa
                windowlen_w = self.model_info.input_w[nlayer+1]
                windowlen_h = self.model_info.input_h[nlayer+1]
                for nfilter in range(self.model_info.filter_n[nlayer]):
                    for window_h in range(windowlen_h):
                        for window_w in range(windowlen_w):
                            num_input = window_h * windowlen_w + window_w
                            preceding_list = self.pe_saa_mat[nlayer][num_input][nfilter] 
                            first_pre_event_idx = preceding_list[0]  # do pe_saa in first pe of preceding cu_saa event
                            do_pe_saa_pos = self.Computation_order[first_pre_event_idx].position_idx[:-2]

                ### Event: edram_wr, data_transfer (for pe_saa)
                            pe_saa_precding_event_idx = []
                            preceding_pe = dict()
                            for pre_event_idx in preceding_list:
                                if self.Computation_order[pre_event_idx].position_idx[:-2] != do_pe_saa_pos: # in other pe
                                    if self.Computation_order[pre_event_idx].position_idx not in preceding_pe:
                                        preceding_pe[self.Computation_order[pre_event_idx].position_idx[:-2]] = [pre_event_idx]
                                    else:
                                        preceding_pe[self.Computation_order[pre_event_idx].position_idx[:-2]].append(pre_event_idx)
                                else: # in same PE
                                    pe_saa_precding_event_idx.append(pre_event_idx)
                            # other PEs
                            for pe_idx in preceding_pe:
                                # Event: pe_saa
                                pe_saa_event_idx = len(self.Computation_order)
                                pe_saa_pos = pe_idx
                                pe_saa_preceding_count = len(preceding_pe[pe_idx])
                                saa_amount = pe_saa_preceding_count
                                preceding_tmp_data = ()
                                pe_saa_inputs  = saa_amount
                                pe_saa_outputs = preceding_tmp_data
                                event = EventMetaData("pe_saa", pe_saa_pos, pe_saa_preceding_count, [pe_saa_event_idx+1], nlayer, pe_saa_inputs, pe_saa_outputs)
                                self.Computation_order.append(event)
                                # add dependency
                                for cu_op_event_idx in preceding_pe[pe_idx]:
                                    self.Computation_order[cu_op_event_idx].proceeding_event.append(pe_saa_event_idx)

                                # Event: edram_wr
                                edram_wr_event_idx = len(self.Computation_order)
                                do_edram_wr_pos = pe_idx
                                edram_wr_preceding_count = 1 # pe_saa
                                edram_wr_inputs  = 0
                                edram_wr_outputs = (nlayer, window_h, window_w, nfilter, "u")
                                event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [edram_wr_event_idx+1], nlayer, edram_wr_inputs, edram_wr_outputs)
                                self.Computation_order.append(event)

                                # Event: data_transfer
                                data_transfer_idx = len(self.Computation_order)
                                source_pe_idx      = pe_idx
                                destination_pe_idx = do_pe_saa_pos
                                data_transfer_preceding_count = 1
                                transfer_inputs  = 0
                                transfer_outputs = (nlayer, window_h, window_w, nfilter, "u")
                                event = EventMetaData("data_transfer", [source_pe_idx, destination_pe_idx], data_transfer_preceding_count, [], nlayer, transfer_inputs, transfer_outputs)
                                self.Computation_order.append(event)
                                pe_saa_precding_event_idx.append(data_transfer_idx)

                ### Event: pe_saa
                            pe_saa_event_idx = len(self.Computation_order)
                            pe_saa_pos = do_pe_saa_pos
                            pe_saa_preceding_count = len(pe_saa_precding_event_idx)
                            saa_amount = pe_saa_preceding_count
                            preceding_tmp_data = (nlayer, nfilter, 0, 0, "u")
                            pe_saa_inputs  = saa_amount
                            pe_saa_outputs = preceding_tmp_data
                            event = EventMetaData("pe_saa", pe_saa_pos, pe_saa_preceding_count, [pe_saa_event_idx+1], nlayer, pe_saa_inputs, pe_saa_outputs)
                            self.Computation_order.append(event)
                            # add dependency
                            for event_idx in pe_saa_precding_event_idx:
                                self.Computation_order[event_idx].proceeding_event.append(pe_saa_event_idx)
                
                ### Event: activation
                            act_event_idx = len(self.Computation_order)
                            do_act_pos = do_pe_saa_pos 
                            act_preceding_count = 1
                            act_inputs  = 0
                            act_outputs = 0
                            event = EventMetaData("activation", do_act_pos, act_preceding_count, [act_event_idx+1], nlayer, act_inputs, act_outputs)
                            self.Computation_order.append(event)

                ### Event: edram_wr, data_transfer(between layer)
                            edram_wr_event_idx = len(self.Computation_order)
                            do_edram_wr_pos = do_pe_saa_pos
                            edram_wr_preceding_count = 1
                            edram_wr_inputs  = 0
                            edram_wr_outputs = (nlayer+1, window_h, window_w, nfilter)
                            if nlayer+1 < self.model_info.layer_length:
                                if self.model_info.layer_list[nlayer+1].layer_type == "fully":
                                    seq = window_w + window_h * windowlen_w + nfilter * windowlen_w * windowlen_h
                                    edram_wr_outputs  = (nlayer+1, seq, 0, 0)
                            event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [], nlayer, edram_wr_inputs, edram_wr_outputs)
                            self.Computation_order.append(event)

                ### Event: data_transfer
                            if nlayer < self.model_info.layer_length - 1: # 最後一層不用生transfer
                                edram_wr_event = event
                                data = edram_wr_outputs
                                if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                    dependency_pe = self.transfer_mat[nlayer][data[1]][data[2]][data[3]]
                                else:
                                    dependency_pe = self.transfer_mat[nlayer][data[1]]
                                dependency_index = dict()
                                for pe_pos in dependency_pe:
                                    if pe_pos == do_edram_wr_pos:
                                        dependency_index[pe_pos] = edram_wr_event_idx
                                    else:
                                        data_transfer_event_idx = len(self.Computation_order)
                                        dependency_index[pe_pos] = data_transfer_event_idx

                                        data_transfer_source = do_edram_wr_pos
                                        data_transfer_destination = pe_pos
                                        data_transfer_preceding_count = 1
                                        data_transfer_inputs  = 0
                                        data_transfer_outputs = edram_wr_event.outputs
                                        event = EventMetaData("data_transfer", [data_transfer_source, data_transfer_destination], data_transfer_preceding_count, [], nlayer, data_transfer_inputs, data_transfer_outputs)
                                        self.Computation_order.append(event)

                                        # add dependency
                                        edram_wr_event.proceeding_event.append(data_transfer_event_idx)
                                
                                        # input requirement
                                        pe_id = data_transfer_source[3] + data_transfer_source[2]*HW().PE_num_x + \
                                                data_transfer_source[1]*HW().PE_num + data_transfer_source[0]*HW().PE_num*HW().Router_num_x
                                        if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                            pos = data[2] + data[1]*self.model_info.input_w[nlayer+1] + data[3]*self.model_info.input_w[nlayer+1]*self.model_info.input_h[nlayer+1] # w + h*width + c*height*width
                                            self.free_buffer_controller.input_require[pe_id][nlayer+1][pos] += 1
                                        else:
                                            self.free_buffer_controller.input_require[pe_id][nlayer+1][data[1]] += 1
                                
                                if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                    self.transfer_mat[nlayer][data[1]][data[2]][data[3]] = dependency_index
                                else:
                                    self.transfer_mat[nlayer][data[1]] = dependency_index

            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                ### Event: edram_rd_ir
                for cu_pos in self.cu_traverse_idx:
                    pe_pos = cu_pos[:-2]
                    rty_idx, rtx_idx = cu_pos[0], cu_pos[1]
                    pey_idx, pex_idx = cu_pos[2], cu_pos[3]
                    cuy_idx, cux_idx = cu_pos[4], cu_pos[5]

                    max_xb_input_len = 0
                    for xby_idx in range(HW().Xbar_num_y):
                        for xbx_idx in range(HW().Xbar_num_x):
                            xbar_inputs = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][nlayer]
                            num_inp = len(xbar_inputs)
                            max_xb_input_len = max(max_xb_input_len, num_inp)

                    for nInp in range(max_xb_input_len):
                        data_feed_to_cu = []
                        for xby_idx in range(HW().Xbar_num_y):
                            for xbx_idx in range(HW().Xbar_num_x):
                                xbar_inputs = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][nlayer]
                                if nInp < len(xbar_inputs):
                                    inp = xbar_inputs[nInp].inputs
                                else:
                                    inp = []
                                for d in inp:
                                    data = [nlayer] + d[1:]
                                    data = tuple(data)
                                    if data not in data_feed_to_cu:
                                        data_feed_to_cu.append(data)

                        eri_event_idx = len(self.Computation_order)
                        if nlayer == 0:
                            eri_preceding_count = 0
                        else:
                            eri_preceding_count = len(data_feed_to_cu)
                            # add dependency
                            for data in data_feed_to_cu:
                                pre_event_idx = self.transfer_mat[nlayer-1][data[1]][pe_pos]
                                self.Computation_order[pre_event_idx].proceeding_event.append(eri_event_idx)
                        eri_position_idx = cu_pos
                        eri_input_sequence  = data_feed_to_cu
                        eri_output_sequence = 0
                        event = EventMetaData("edram_rd_ir", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_input_sequence, eri_output_sequence)
                        self.Computation_order.append(event)

                        # input requirement
                        pe_id = cu_pos[3] + cu_pos[2]*HW().PE_num_x + \
                                cu_pos[1]*HW().PE_num + cu_pos[0]*HW().PE_num*HW().Router_num_x
                        for d in data_feed_to_cu:
                            pos = d[1]
                            self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1

                ### Event: cu_operation
                        cu_operation_idx = len(self.Computation_order)
                        num_ou_in_xb = dict()
                        max_ou = 0
                        for xby_idx in range(HW().Xbar_num_y):
                            for xbx_idx in range(HW().Xbar_num_x):
                                xbar_inputs  = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][nlayer]
                                if nInp < len(xbar_inputs):
                                    inp = xbar_inputs[nInp]
                                else:
                                    continue
                                    
                                # Total ou in the event
                                num_ou_h = math.ceil(len(inp.inputs)/ HW().OU_h)
                                num_ou_w = math.ceil(inp.Cols  / HW().OU_w)
                                num_ou = num_ou_h * num_ou_w * self.model_info.input_bit

                                xb_idx = xbx_idx + xby_idx * HW().Xbar_num_x
                                num_ou_in_xb[xb_idx] = num_ou # Q1: xb_idx -> (0,0)
                                max_ou = max(num_ou, max_ou)

                                num_input = 0 #num_input = inp.inputs[0][0]

                                filter_list = inp.Filters # 此xbar 會計算到的 filter
                                for nfilter in filter_list:
                                    grid = self.pe_saa_mat[nlayer][num_input][nfilter]
                                    if grid == 0.0: # 優: init直接設定成list
                                        self.pe_saa_mat[nlayer][num_input][nfilter] = []
                                    if cu_operation_idx not in self.pe_saa_mat[nlayer][num_input][nfilter]:
                                        self.pe_saa_mat[nlayer][num_input][nfilter].append(cu_operation_idx)
                        
                        position_idx = cu_pos
                        preceding_count = 1
                        cu_op_inputs  = max_ou # 最多要幾次ou做完
                        cu_op_outputs = num_ou_in_xb
                        event = EventMetaData("cu_operation", position_idx, preceding_count, [], nlayer, cu_op_inputs, cu_op_outputs)
                        self.Computation_order.append(event)

                ### Event: pe_saa, edram_wr, data_transfer
                for nfilter in range(self.model_info.filter_n[nlayer]):
                    num_input = 0 # fully只有一組input vector
                    preceding_list = self.pe_saa_mat[nlayer][num_input][nfilter]
                    first_pre_event_idx = preceding_list[0]  # do pe_saa in first pe of preceding cu_saa event
                    do_pe_saa_pos = self.Computation_order[first_pre_event_idx].position_idx[:-2]

                ### Event: pe_saa, edram_wr, data_transfer
                    pe_saa_precding_event_idx = []
                    preceding_pe = dict()
                    for pre_event_idx in preceding_list:
                        if self.Computation_order[pre_event_idx].position_idx[:-2] != do_pe_saa_pos: # in other PE
                            if self.Computation_order[pre_event_idx].position_idx[:-2] not in preceding_pe:
                                preceding_pe[self.Computation_order[pre_event_idx].position_idx[:-2]] = [pre_event_idx]
                            else:
                                preceding_pe[self.Computation_order[pre_event_idx].position_idx[:-2]].append(pre_event_idx)
                        else: # in same PE
                            pe_saa_precding_event_idx.append(pre_event_idx)
                    # other PEs
                    for pe_idx in preceding_pe:
                        # Event: pe_saa
                        pe_saa_event_idx = len(self.Computation_order)
                        pe_saa_pos = pe_idx
                        pe_saa_preceding_count = len(preceding_pe[pe_idx])
                        saa_amount = pe_saa_preceding_count
                        preceding_tmp_data = ()
                        pe_saa_inputs  = saa_amount
                        pe_saa_outputs = preceding_tmp_data
                        event = EventMetaData("pe_saa", pe_saa_pos, pe_saa_preceding_count, [pe_saa_event_idx+1], nlayer, pe_saa_inputs, pe_saa_outputs)
                        self.Computation_order.append(event)
                        # add dependency
                        for cu_op_event_idx in preceding_pe[pe_idx]:
                            self.Computation_order[cu_op_event_idx].proceeding_event.append(pe_saa_event_idx)

                        # Event: edram_wr
                        edram_wr_event_idx = len(self.Computation_order)
                        do_edram_wr_pos = pe_idx
                        edram_wr_preceding_count = 1 # pe_saa
                        edram_wr_inputs  = 0
                        edram_wr_outputs = (nlayer, nfilter, 0, 0, "u") # Q2: 不同pe應該寫不同的質
                        event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [edram_wr_event_idx+1], nlayer, edram_wr_inputs, edram_wr_outputs)
                        self.Computation_order.append(event)

                        # Event: data_transfer
                        data_transfer_idx = len(self.Computation_order)
                        source_pe_idx      = pe_idx
                        destination_pe_idx = do_pe_saa_pos
                        data_transfer_preceding_count = 1
                        transfer_inputs  = 0
                        transfer_outputs = (nlayer, nfilter, 0, 0, "u") # Q3: 不同pe應該寫不同的質
                        event = EventMetaData("data_transfer", [source_pe_idx, destination_pe_idx], data_transfer_preceding_count, [], nlayer, transfer_inputs, transfer_outputs)
                        self.Computation_order.append(event)
                        pe_saa_precding_event_idx.append(data_transfer_idx)

                ### Event: pe_saa
                    pe_saa_event_idx = len(self.Computation_order)
                    pe_saa_pos = do_pe_saa_pos
                    pe_saa_preceding_count = len(pe_saa_precding_event_idx)
                    saa_amount = pe_saa_preceding_count
                    preceding_tmp_data = (nlayer, nfilter, 0, 0, "u") # Q4: 做完pe_saa要free掉的資料, 不同pe應該寫不同的質
                    pe_saa_inputs  = saa_amount
                    pe_saa_outputs = preceding_tmp_data
                    event = EventMetaData("pe_saa", pe_saa_pos, pe_saa_preceding_count, [pe_saa_event_idx+1], nlayer, pe_saa_inputs, pe_saa_outputs)
                    self.Computation_order.append(event)
                    # add dependency
                    for event_idx in pe_saa_precding_event_idx:
                        self.Computation_order[event_idx].proceeding_event.append(pe_saa_event_idx)

                ### Event: activation
                    act_event_idx = len(self.Computation_order)
                    do_act_pos = do_pe_saa_pos
                    act_preceding_count = 1
                    act_inputs  = 0
                    act_outputs = 0
                    event = EventMetaData("activation", do_act_pos, act_preceding_count, [act_event_idx+1], nlayer, act_inputs, act_outputs)
                    self.Computation_order.append(event)

                ### Event: edram_wr, data_transfer(between layer)
                    edram_wr_event_idx = len(self.Computation_order)
                    do_edram_wr_pos = do_pe_saa_pos
                    edram_wr_preceding_count = 1
                    edram_wr_inputs  = 0
                    edram_wr_outputs = (nlayer+1, nfilter, 0, 0)
                    event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [], nlayer, edram_wr_inputs, edram_wr_outputs)
                    self.Computation_order.append(event) 
               
                ### Event: data_transfer
                    if nlayer < self.model_info.layer_length - 1: # 最後一層不用生transfer
                        edram_wr_event = event
                        dependency_pe = self.transfer_mat[nlayer][nfilter] # set
                        dependency_index = dict()
                        for pe_pos in dependency_pe: # pe destination
                            if pe_pos == do_edram_wr_pos:
                                dependency_index[pe_pos] = edram_wr_event_idx
                            else:
                                data_transfer_event_idx = len(self.Computation_order)
                                dependency_index[pe_pos] = data_transfer_event_idx

                                data_transfer_source = do_edram_wr_pos
                                data_transfer_destination = pe_pos
                                data_transfer_preceding_count = 1
                                data_transfer_inputs  = 0
                                data_transfer_outputs = (nlayer+1, nfilter, 0, 0)
                                event = EventMetaData("data_transfer", [data_transfer_source, data_transfer_destination], data_transfer_preceding_count, [], nlayer, data_transfer_inputs, data_transfer_outputs)
                                self.Computation_order.append(event)

                                # add dependency
                                edram_wr_event.proceeding_event.append(data_transfer_event_idx)
                                
                                # input requirement
                                pe_id = data_transfer_source[3] + data_transfer_source[2]*HW().PE_num_x + \
                                        data_transfer_source[1]*HW().PE_num + data_transfer_source[0]*HW().PE_num*HW().Router_num_x
                                self.free_buffer_controller.input_require[pe_id][nlayer+1][nfilter] += 1

                        self.transfer_mat[nlayer][nfilter] = dependency_index # set換成dict

            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                for rty_idx in range(HW().Router_num_y):
                    for rtx_idx in range(HW().Router_num_x):
                        for pey_idx in range(HW().PE_num_y):
                            for pex_idx in range(HW().PE_num_x):
                                pe_pos = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                for Inputs in self.mp_info.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx][nlayer]:
                ### Event: edram_rd_pool
                                    for inputs in Inputs:
                                        pool_inputs = []
                                        pool_event_idx = len(self.Computation_order)
                                        pool_preceding_count = len(inputs)
                                        # add dependency
                                        for data in inputs:
                                            pool_inputs.append(tuple(data[1:]))
                                            data = data[2:] # [h, w, c]
                                            pre_event_idx = self.transfer_mat[nlayer-1][data[0]][data[1]][data[2]][pe_pos]
                                            self.Computation_order[pre_event_idx].proceeding_event.append(pool_event_idx)

                                        pool_position_idx = pe_pos
                                        pool_inputs  = pool_inputs
                                        pool_outputs = 0
                                        event = EventMetaData("edram_rd_pool", pool_position_idx, pool_preceding_count, [pool_event_idx+1], nlayer, pool_inputs, pool_outputs)
                                        self.Computation_order.append(event)

                                        # input require
                                        pe_id = pe_pos[3] + pe_pos[2]*HW().PE_num_x + \
                                                pe_pos[1]*HW().PE_num + pe_pos[0]*HW().PE_num*HW().Router_num_x
                                        for data in pool_inputs:
                                            d = data[1:]
                                            pos = d[1] + d[0]*self.model_info.input_w[nlayer] + d[2]*self.model_info.input_w[nlayer]*self.model_info.input_h[nlayer] # w + h*width + c*height*width
                                            self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1

                ### Event: edram_wr
                                        edram_wr_event_idx = len(self.Computation_order)
                                        do_edram_wr_pos = pe_pos
                                        edram_wr_preceding_count = 1
                                        edram_wr_inputs  = 0
                                        edram_wr_outputs = (nlayer+1,
                                                            inputs[0][2] // self.model_info.pooling_strides[nlayer], 
                                                            inputs[0][3] // self.model_info.pooling_strides[nlayer], 
                                                            inputs[0][4])

                                        if nlayer+1 < self.model_info.layer_length:
                                            if self.model_info.layer_list[nlayer+1].layer_type == "fully":
                                                seq = edram_wr_outputs[1] * self.model_info.input_w[nlayer+1] + edram_wr_outputs[2] + edram_wr_outputs[3] * self.model_info.input_h[nlayer+1] * self.model_info.input_w[nlayer+1]
                                                edram_wr_outputs  = (nlayer+1, seq, 0, 0)
                                        event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [], nlayer, edram_wr_inputs, edram_wr_outputs)
                                        self.Computation_order.append(event)

                ### Event: data_transfer
                                        if nlayer < self.model_info.layer_length - 1: # 最後一層不用生transfer
                                            edram_wr_event = event
                                            data = edram_wr_outputs
                                            if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                                dependency_pe = self.transfer_mat[nlayer][data[1]][data[2]][data[3]]
                                            else:
                                                dependency_pe = self.transfer_mat[nlayer][data[1]]
                                            dependency_index = dict()
                                            for des_pe_pos in dependency_pe: # pe destination
                                                if des_pe_pos == do_edram_wr_pos:
                                                    dependency_index[des_pe_pos] = edram_wr_event_idx
                                                else:
                                                    data_transfer_event_idx = len(self.Computation_order)
                                                    dependency_index[des_pe_pos] = data_transfer_event_idx

                                                    data_transfer_source = do_edram_wr_pos
                                                    data_transfer_destination = des_pe_pos
                                                    data_transfer_preceding_count = 1
                                                    data_transfer_inputs  = 0
                                                    data_transfer_outputs = edram_wr_event.outputs
                                                    event = EventMetaData("data_transfer", [data_transfer_source, data_transfer_destination], data_transfer_preceding_count, [], nlayer, data_transfer_inputs, data_transfer_outputs)
                                                    self.Computation_order.append(event)
                                                    
                                                    # add dependency
                                                    edram_wr_event.proceeding_event.append(data_transfer_event_idx)
                                
                                                    # input requirement
                                                    pe_id = data_transfer_source[3] + data_transfer_source[2]*HW().PE_num_x + \
                                                            data_transfer_source[1]*HW().PE_num + data_transfer_source[0]*HW().PE_num*HW().Router_num_x
                                                    if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                                        pos = data[2] + data[1]*self.model_info.input_w[nlayer+1] + data[3]*self.model_info.input_w[nlayer+1]*self.model_info.input_h[nlayer+1] # w + h*width + c*height*width
                                                        self.free_buffer_controller.input_require[pe_id][nlayer+1][pos] += 1
                                                    else:
                                                        self.free_buffer_controller.input_require[pe_id][nlayer+1][data[1]] += 1
                                            if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                                self.transfer_mat[nlayer][data[1]][data[2]][data[3]] = dependency_index
                                            else:
                                                self.transfer_mat[nlayer][data[1]] = dependency_index
        print('Order generated!')

    def trace_order(self):
        edram_rd_ir_ctr = 0
        cu_op_ctr  = 0
        pe_saa_ctr = 0
        activation_ctr = 0
        pooling_ctr = 0
        edram_wr_ctr = 0
        edram_rd_pool_ctr = 0
        data_transfer_ctr = 0
        for e in self.Computation_order:
            t = e.event_type
            if t == "edram_rd_ir":
                edram_rd_ir_ctr += 1
            elif t == "cu_operation":
                cu_op_ctr += 1
            elif t == "pe_saa":
                pe_saa_ctr += 1
            elif t == "activation":
                activation_ctr += 1
            elif t == "edram_wr":
                edram_wr_ctr += 1
            elif t == "edram_rd_pool":
                edram_rd_pool_ctr += 1
            elif t == "pooling":
                pooling_ctr += 1
            elif t == "data_transfer":
                data_transfer_ctr += 1
            else:
                print("event type error:", t)

        print("edram_rd_ir_ctr", edram_rd_ir_ctr)
        print("cu_op_ctr", cu_op_ctr)
        print("pe_saa_ctr", pe_saa_ctr)
        print("activation_ctr", activation_ctr)
        print("edram_wr_ctr", edram_wr_ctr)
        print("edram_rd_pool_ctr", edram_rd_pool_ctr)
        print("data_transfer_ctr", data_transfer_ctr)
        print("total", len(self.Computation_order))

        if False:
            for e in self.Computation_order:
                print(self.Computation_order.index(e), e)

    def __str__(self):
        return str(self.__dict__)
