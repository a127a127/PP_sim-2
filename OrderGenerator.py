from HardwareMetaData import HardwareMetaData
from configs.ModelConfig import ModelConfig
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
        self.hd_info = HardwareMetaData()
        self.mp_info = mapping_information
        self.free_buffer_controller = FreeBufferController()

        # mapping
        self.layer_mapping_to_pe = self.mp_info.layer_mapping_to_pe

        self.cu_traverse_idx = []
        for rty_idx in range(self.hd_info.Router_num_y):
            for rtx_idx in range(self.hd_info.Router_num_x):
                for pey_idx in range(self.hd_info.PE_num_y):
                    for pex_idx in range(self.hd_info.PE_num_x):
                        for cuy_idx in range(self.hd_info.CU_num_y):
                            for cux_idx in range(self.hd_info.CU_num_x):
                                cu_pos = (rty_idx, rtx_idx, pey_idx, pex_idx, cuy_idx, cux_idx)
                                self.cu_traverse_idx.append(cu_pos)

        self.pe_saa_mat = [] # for dependency
        self.feature_mat = [] # for dependency
        for i in range(self.model_info.layer_length):
            if self.model_info.layer_list[i].layer_type == "convolution":
                self.pe_saa_mat.append(np.zeros((self.model_info.input_number[i], self.model_info.filter_n[i])).tolist())
                if i+1 < self.model_info.layer_length:
                    if self.model_info.layer_list[i+1].layer_type == "fully":
                        self.feature_mat.append(np.zeros((self.model_info.input_h[i+1] * self.model_info.input_w[i+1] * self.model_info.input_c[i+1], 1, 1)).tolist())
                    else:
                        self.feature_mat.append(np.zeros((self.model_info.input_h[i+1], self.model_info.input_w[i+1], self.model_info.input_c[i+1])).tolist())
            elif self.model_info.layer_list[i].layer_type == "pooling":
                self.pe_saa_mat.append([])
                if i+1 < self.model_info.layer_length:
                    if self.model_info.layer_list[i+1].layer_type == "fully":
                        self.feature_mat.append(np.zeros((self.model_info.input_h[i+1] * self.model_info.input_w[i+1] * self.model_info.input_c[i+1], 1, 1)).tolist())
                    else:
                        self.feature_mat.append(np.zeros((self.model_info.input_h[i+1], self.model_info.input_w[i+1], self.model_info.input_c[i+1])).tolist())
            elif self.model_info.layer_list[i].layer_type == "fully":
                self.pe_saa_mat.append(np.zeros((self.model_info.input_number[i], self.model_info.filter_n[i])).tolist())
                if i+1 < self.model_info.layer_length:
                    self.feature_mat.append(np.zeros((self.model_info.filter_n[i], 1, 1)).tolist())
            else:
                print("Wrong layer type")
                exit()
        self.Computation_order = []
        self.generate_order()
        if trace:
            self.trace_order()

    def generate_order(self):
        for nlayer in range(self.model_info.layer_length):
            print("Generate layer", nlayer, self.model_info.layer_list[nlayer].layer_type)
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                ### Event: data_transfer, edram_rd_ir
                for cu_pos in self.cu_traverse_idx:
                    pe_pos = cu_pos[:-2]
                    rty_idx, rtx_idx= cu_pos[0], cu_pos[1]
                    pey_idx, pex_idx = cu_pos[2], cu_pos[3]
                    cuy_idx, cux_idx = cu_pos[4], cu_pos[5]

                    # 算CU內的所有XB，最多需要多少組input
                    max_xb_input_len = 0
                    for xby_idx in range(self.hd_info.Xbar_num_y):
                        for xbx_idx in range(self.hd_info.Xbar_num_x):
                            num_inp = 0
                            xbar_inputs = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx]
                            for inp in xbar_inputs:
                                if inp.nlayer == nlayer:
                                    num_inp += 1
                            max_xb_input_len = max(max_xb_input_len, num_inp)

                    # 每一組input產生一個edram read event
                    for nInp in range(max_xb_input_len):
                        data_feed_to_cu = [] # 一次edram read的資料
                        for xby_idx in range(self.hd_info.Xbar_num_y):
                            for xbx_idx in range(self.hd_info.Xbar_num_x):
                                xbar_inputs = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx]
                                if nInp < len(xbar_inputs):
                                    inp = xbar_inputs[nInp].inputs
                                else:
                                    inp = []

                                # inp紀錄的位置為padding後的位置
                                if self.model_info.layer_list[nlayer].padding == "SAME":
                                    for d in inp:
                                        input_num = d[0]
                                        h = d[1] - self.model_info.pad[nlayer]
                                        w = d[2] - self.model_info.pad[nlayer]
                                        c = d[3]
                                        # 不紀錄pading的值
                                        if w >= 0 and w < self.model_info.input_w[nlayer] and h >= 0 and h < self.model_info.input_h[nlayer]:
                                            data = [input_num, h, w, c]
                                            if data not in data_feed_to_cu: # 這裡會search
                                                data_feed_to_cu.append(data)
                                else: # padding == "VALID"
                                    for d in inp:
                                        if d not in data_feed_to_cu: # 這裡會search
                                            data_feed_to_cu.append(d)

                        if nlayer == 0:
                            eri_preceding_count = 0
                        else: # 若非第一層要算dependency
                            eri_preceding_count = 0
                            start_append_idx = len(self.Computation_order)
                            for input_data in data_feed_to_cu: # 檢查每一筆資料的有dependency的event
                                preceding_list = self.feature_mat[nlayer-1][input_data[1]][input_data[2]][input_data[3]] # [h][w][c]
                                for pre_event in preceding_list: 
                                    edram_wr_event            = self.Computation_order[pre_event]
                                    data_transfer_source      = edram_wr_event.position_idx
                                    data_transfer_destination = pe_pos
                                    if data_transfer_source != data_transfer_destination: # edram write event發生在別的pe, 則多生data transfer event
                                        ### Event: data_transfer
                                        data_transfer_event_idx = len(self.Computation_order)
                                        edram_wr_event.proceeding_event.append(data_transfer_event_idx)
                                        data_transfer_preceding_count = 1 
                                        data_transfer_inputs = 0
                                        data_transfer_outputs = edram_wr_event.outputs # [[h, w, c]]
                                        event = EventMetaData("data_transfer", [data_transfer_source, data_transfer_destination], data_transfer_preceding_count, [], nlayer-1, data_transfer_inputs, data_transfer_outputs)
                                        self.Computation_order.append(event)

                                        # input requirement
                                        pe_id = data_transfer_source[3] + data_transfer_source[2]*self.hd_info.PE_num_x + data_transfer_source[1]*self.hd_info.PE_num + data_transfer_source[0]*self.hd_info.PE_num*self.hd_info.Router_num_x
                                        for d in data_transfer_outputs:
                                            pos = d[1] + d[0]*self.model_info.input_w[nlayer] + d[2]*self.model_info.input_w[nlayer]*self.model_info.input_h[nlayer] # w + h*width + c*height*width
                                            self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1

                            ## dependency
                            eri_event_idx = len(self.Computation_order) 
                            # transfer event的dependency
                            for idx in range(start_append_idx, eri_event_idx):
                                self.Computation_order[idx].proceeding_event.append(eri_event_idx)
                            # edram write event的dependency
                            for input_data in data_feed_to_cu:
                                preceding_list = self.feature_mat[nlayer-1][input_data[1]][input_data[2]][input_data[3]]
                                for pre_event in preceding_list:
                                    edram_wr_event            = self.Computation_order[pre_event]
                                    data_transfer_source      = edram_wr_event.position_idx
                                    data_transfer_destination = pe_pos
                                    if data_transfer_source == data_transfer_destination: 
                                        self.Computation_order[pre_event].proceeding_event.append(eri_event_idx)
                                    eri_preceding_count += 1

                        eri_event_idx = len(self.Computation_order)
                        eri_position_idx = cu_pos
                        eri_input_sequence = data_feed_to_cu # [[num_input, h, w, c]]
                        eri_output_sequence = 0
                        event = EventMetaData("edram_rd_ir", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_input_sequence, eri_output_sequence)
                        self.Computation_order.append(event)

                        # input requirement
                        pe_id = cu_pos[3] + cu_pos[2]*self.hd_info.PE_num_x + \
                                cu_pos[1]*self.hd_info.PE_num + cu_pos[0]*self.hd_info.PE_num*self.hd_info.Router_num_x
                        for d in eri_input_sequence:
                            pos = d[2] + d[1]*self.model_info.input_w[nlayer] + d[3]*self.model_info.input_w[nlayer]*self.model_info.input_h[nlayer] # w + h*width + c*height*width
                            self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1

                ### Event: cu_operation
                        cu_operation_idx = len(self.Computation_order)
                        num_ou_in_xb = dict()
                        max_ou = 0
                        for xby_idx in range(self.hd_info.Xbar_num_y):
                            for xbx_idx in range(self.hd_info.Xbar_num_x):
                                xbar_inputs  = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx]
                                xbar_weights = self.mp_info.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx]
                                if nInp < len(xbar_inputs):
                                    inp = xbar_inputs[nInp]
                                else:
                                    inp = []
                                if not inp:
                                    continue
                                
                                # Total ou in the event
                                num_ou_h = math.ceil(len(inp.xbar_row)    / self.hd_info.OU_h)
                                num_ou_w = math.ceil(len(inp.xbar_column) / self.hd_info.OU_w)
                                num_ou = num_ou_h * num_ou_w * self.model_info.input_bit

                                #num_ou_in_xb[(xby_idx, xbx_idx)] = num_ou
                                xb_idx = xbx_idx + xby_idx * HardwareMetaData().Xbar_num_x
                                num_ou_in_xb[xb_idx] = num_ou
                                max_ou = max(num_ou, max_ou)

                                num_input = inp.inputs[0][0]

                                filter_list = list() # 此xbar 會計算到的 filter
                                row_idx = inp.xbar_row[0] # 只需取一第一個row
                                for col_idx in inp.xbar_column:
                                    nfilter = xbar_weights[row_idx][col_idx][2]
                                    if nfilter not in filter_list:
                                        filter_list.append(nfilter)
                                for nfilter in filter_list:
                                    # dependecy matrix
                                    grid = self.pe_saa_mat[nlayer][num_input][nfilter]
                                    if grid == 0.0:
                                        self.pe_saa_mat[nlayer][num_input][nfilter] = []
                                    if cu_operation_idx not in self.pe_saa_mat[nlayer][num_input][nfilter]:
                                        self.pe_saa_mat[nlayer][num_input][nfilter].append(cu_operation_idx)
                        
                        position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx, cuy_idx, cux_idx)
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
                            pe_saa_preceding_count = len(preceding_list)#0 # append pe_saa前有幾個event先append了
                            first_pre_event_idx = preceding_list[0]  # do pe_saa in first pe of preceding cu_saa event
                            do_pe_saa_pos = self.Computation_order[first_pre_event_idx].position_idx[:-2]
                            start_append_idx = len(self.Computation_order)

                ### Event: edram_wr, data_transfer (for pe_saa)
                            preceding_cu = dict()
                            for pre_event_idx in preceding_list:
                                if self.Computation_order[pre_event_idx].position_idx[:-2] != do_pe_saa_pos: # data in other pe
                                    if self.Computation_order[pre_event_idx].position_idx not in preceding_cu:
                                        preceding_cu[self.Computation_order[pre_event_idx].position_idx] = [pre_event_idx]
                                    else:
                                        preceding_cu[self.Computation_order[pre_event_idx].position_idx].append(pre_event_idx)
                            for cu_idx in preceding_cu:
                                edram_wr_event_idx = len(self.Computation_order)
                                for pre_event_idx in preceding_cu[cu_idx]:
                                    self.Computation_order[pre_event_idx].proceeding_event.append(edram_wr_event_idx)
                                pe_idx = cu_idx[0:4]

                                edram_wr_pe_pos  = pe_idx
                                edram_wr_inputs  = 0
                                edram_wr_outputs = [[window_h, window_w, nfilter, "u"]]
                                edram_wr_preceding_count = len(preceding_cu[cu_idx])
                                event = EventMetaData("edram_wr", edram_wr_pe_pos, edram_wr_preceding_count, [edram_wr_event_idx+1], nlayer, edram_wr_inputs, edram_wr_outputs)
                                self.Computation_order.append(event)

                                source_pe_idx = pe_idx
                                transfer_inputs  = 0 #[[window_h, window_w, nfilter, "u"]]
                                transfer_outputs = [[window_h, window_w, nfilter, "u"]]
                                event = EventMetaData("data_transfer", [source_pe_idx, do_pe_saa_pos], 1, [], nlayer, transfer_inputs, transfer_outputs)
                                self.Computation_order.append(event)

                ### Event: pe_saa
                            pe_saa_event_idx = len(self.Computation_order)
                            preceding_cu_idx = list()
                            preceding_tmp_data = list()
                            for pre_event_idx in preceding_list:
                                if self.Computation_order[pre_event_idx].position_idx[:-2] == do_pe_saa_pos: # in same PE
                                    self.Computation_order[pre_event_idx].proceeding_event.append(pe_saa_event_idx)
                                else:
                                    data_transfer_id = self.Computation_order[pre_event_idx].proceeding_event[0] + 1
                                    preceding_tmp_data.append(self.Computation_order[data_transfer_id].outputs[0]) # data transfer output
                                if self.Computation_order[pre_event_idx].position_idx not in preceding_cu_idx:
                                    preceding_cu_idx.append(self.Computation_order[pre_event_idx].position_idx)
                            # add dependency
                            for idx in range(start_append_idx+1, pe_saa_event_idx, 2):
                                self.Computation_order[idx].proceeding_event.append(pe_saa_event_idx)
                            pe_saa_inputs  = [preceding_cu_idx, preceding_tmp_data]
                            pe_saa_outputs = [[window_h, window_w, nfilter]]
                            event = EventMetaData("pe_saa", do_pe_saa_pos, pe_saa_preceding_count, [], nlayer, pe_saa_inputs, pe_saa_outputs)
                            self.Computation_order.append(event)

                ### Event: activation
                            do_act_pos = do_pe_saa_pos 
                            act_preceding_count = 1
                            act_inputs  = 0 #[[window_h, window_w, nfilter]]
                            act_outputs = 0 #[[window_h, window_w, nfilter]]
                            # add dependency
                            act_event_idx = len(self.Computation_order)
                            self.Computation_order[pe_saa_event_idx].proceeding_event.append(act_event_idx)

                            event = EventMetaData("activation", do_act_pos, act_preceding_count, [], nlayer, act_inputs, act_outputs)
                            self.Computation_order.append(event)

                ### Event: edram_wr
                            edram_wr_event_idx = len(self.Computation_order)
                            self.Computation_order[act_event_idx].proceeding_event.append(edram_wr_event_idx)

                            do_edram_wr_pos = do_act_pos
                            edram_wr_preceding_count = 1
                            edram_wr_inputs  = 0
                            edram_wr_outputs = [[window_h, window_w, nfilter]]

                            # dependecy matrix
                            if nlayer+1 < self.model_info.layer_length:
                                if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                    if self.feature_mat[nlayer][window_h][window_w][nfilter] == 0.0:
                                        self.feature_mat[nlayer][window_h][window_w][nfilter] = []
                                    self.feature_mat[nlayer][window_h][window_w][nfilter].append(edram_wr_event_idx)
                                else:
                                    seq = window_w + window_h * windowlen_w + nfilter * windowlen_w * windowlen_h
                                    edram_wr_outputs  = [[seq, 0, 0]]
                                    if self.feature_mat[nlayer][seq][0][0] == 0.0:
                                        self.feature_mat[nlayer][seq][0][0] = []
                                    self.feature_mat[nlayer][seq][0][0].append(edram_wr_event_idx)
                            event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [], nlayer, edram_wr_inputs, edram_wr_outputs)
                            self.Computation_order.append(event)

            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                ### Event: data_transfer, edram_rd_ir
                for cu_pos in self.cu_traverse_idx:
                    pe_pos = cu_pos[:-2]
                    rty_idx, rtx_idx= cu_pos[0], cu_pos[1]
                    pey_idx, pex_idx = cu_pos[2], cu_pos[3]
                    cuy_idx, cux_idx = cu_pos[4], cu_pos[5]

                    max_xb_input_len = 0
                    for xby_idx in range(self.hd_info.Xbar_num_y):
                        for xbx_idx in range(self.hd_info.Xbar_num_x):
                            num_inp = 0
                            xbar_inputs = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx]
                            for inp in xbar_inputs:
                                if inp.nlayer == nlayer:
                                    num_inp += 1
                            max_xb_input_len = max(max_xb_input_len, num_inp)
                    for nInp in range(max_xb_input_len):
                        data_feed_to_cu = []
                        for xby_idx in range(self.hd_info.Xbar_num_y):
                            for xbx_idx in range(self.hd_info.Xbar_num_x):
                                xbar_inputs = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx]
                                if nInp < len(xbar_inputs):
                                    inp = xbar_inputs[nInp].inputs
                                else:
                                    inp = []

                                for d in inp:
                                    if d not in data_feed_to_cu:
                                        data_feed_to_cu.append(d)
                        
                        if nlayer == 0:
                            eri_preceding_count = 0
                        if nlayer != 0:
                            eri_preceding_count = 0
                            start_append_idx = len(self.Computation_order)
                            for input_data in data_feed_to_cu:
                                preceding_list = self.feature_mat[nlayer-1][input_data[1]][input_data[2]][input_data[3]] # [h][w][c]
                                for pre_event in preceding_list: 
                                    edram_wr_event            = self.Computation_order[pre_event]
                                    data_transfer_source      = edram_wr_event.position_idx
                                    data_transfer_destination = pe_pos
                                    if data_transfer_source != data_transfer_destination:
                                        ### Event: data_transfer
                                        data_transfer_event_idx = len(self.Computation_order)
                                        edram_wr_event.proceeding_event.append(data_transfer_event_idx)
                                        data_transfer_preceding_count = 1
                                        data_transfer_inputs  = 0
                                        data_transfer_outputs = edram_wr_event.outputs
                                        event = EventMetaData("data_transfer", [data_transfer_source, data_transfer_destination], data_transfer_preceding_count, [], nlayer-1, data_transfer_inputs, data_transfer_outputs)
                                        self.Computation_order.append(event)

                                        # input requirement
                                        pe_id = data_transfer_source[3] + data_transfer_source[2]*self.hd_info.PE_num_x + data_transfer_source[1]*self.hd_info.PE_num + data_transfer_source[0]*self.hd_info.PE_num*self.hd_info.Router_num_x
                                        for d in data_transfer_outputs:
                                            pos = d[0]
                                            self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1
                            
                            # dependency
                            eri_event_idx = len(self.Computation_order)
                            for idx in range(start_append_idx, eri_event_idx):
                                self.Computation_order[idx].proceeding_event.append(eri_event_idx)
                            for input_data in data_feed_to_cu:
                                preceding_list = self.feature_mat[nlayer-1][input_data[1]][input_data[2]][input_data[3]] # [h][w][c]
                                for pre_event in preceding_list:
                                    edram_wr_event            = self.Computation_order[pre_event]
                                    data_transfer_source      = edram_wr_event.position_idx
                                    data_transfer_destination = pe_pos
                                    if data_transfer_source == data_transfer_destination:
                                        self.Computation_order[pre_event].proceeding_event.append(eri_event_idx)
                                    eri_preceding_count += 1

                        eri_event_idx = len(self.Computation_order)
                        eri_position_idx = cu_pos
                        eri_input_sequence = data_feed_to_cu
                        eri_output_sequence = 0
                        event = EventMetaData("edram_rd_ir", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_input_sequence, eri_output_sequence)
                        self.Computation_order.append(event)

                        # input requirement
                        pe_id = cu_pos[3] + cu_pos[2]*self.hd_info.PE_num_x + \
                                cu_pos[1]*self.hd_info.PE_num + cu_pos[0]*self.hd_info.PE_num*self.hd_info.Router_num_x
                        for d in eri_input_sequence:
                            pos = d[1]
                            self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1

                ### Event: cu_operation
                        cu_operation_idx = len(self.Computation_order)
                        num_ou_in_xb = dict()
                        max_ou = 0
                        for xby_idx in range(self.hd_info.Xbar_num_y):
                            for xbx_idx in range(self.hd_info.Xbar_num_x):
                                xbar_inputs  = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx]
                                xbar_weights = self.mp_info.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx]
                                if nInp < len(xbar_inputs):
                                    inp = xbar_inputs[nInp]
                                else:
                                    inp = []
                                if not inp:
                                    continue
                                # Total ou in the event
                                num_ou_h = math.ceil(len(inp.xbar_row)    / self.hd_info.OU_h)
                                num_ou_w = math.ceil(len(inp.xbar_column) / self.hd_info.OU_w)
                                num_ou = num_ou_h * num_ou_w * self.model_info.input_bit

                                #num_ou_in_xb[(xby_idx, xbx_idx)] = num_ou
                                xb_idx = xbx_idx + xby_idx * HardwareMetaData().Xbar_num_x
                                num_ou_in_xb[xb_idx] = num_ou
                                max_ou = max(num_ou, max_ou)

                                num_input = inp.inputs[0][0]

                                filter_list = list() # 此xbar 會計算到的 filter
                                row_idx = inp.xbar_row[0] # 只需取一第一個row
                                for col_idx in inp.xbar_column:
                                    nfilter = xbar_weights[row_idx][col_idx][2]
                                    if nfilter not in filter_list:
                                        filter_list.append(nfilter)
                                for nfilter in filter_list:
                                    # dependecy matrix
                                    grid = self.pe_saa_mat[nlayer][num_input][nfilter]
                                    if grid == 0.0:
                                        self.pe_saa_mat[nlayer][num_input][nfilter] = []
                                    if cu_operation_idx not in self.pe_saa_mat[nlayer][num_input][nfilter]:
                                        self.pe_saa_mat[nlayer][num_input][nfilter].append(cu_operation_idx)
                        
                        position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx, cuy_idx, cux_idx)
                        preceding_count = 1
                        cu_op_inputs  = max_ou # 最多要幾次ou做完
                        cu_op_outputs = num_ou_in_xb
                        event = EventMetaData("cu_operation", position_idx, preceding_count, [], nlayer, cu_op_inputs, cu_op_outputs)
                        self.Computation_order.append(event)

                ### Event: edram_wr, data_transfer (for pe_saa), pe_saa
                for nfilter in range(self.model_info.filter_n[nlayer]):
                    num_input = 0
                    preceding_list = self.pe_saa_mat[nlayer][num_input][nfilter]
                    pe_saa_preceding_count = len(preceding_list) # append pe_saa前有幾個event先append了
                    first_pre_event_idx = preceding_list[0]  # do pe_saa in first pe of preceding cu_saa event
                    do_pe_saa_pos = self.Computation_order[first_pre_event_idx].position_idx[:-2]
                    start_append_idx = len(self.Computation_order)

                ### Event: edram_wr, data_transfer (for pe_saa)
                    preceding_cu = dict()
                    for pre_event_idx in preceding_list:
                        if self.Computation_order[pre_event_idx].position_idx[:-2] != do_pe_saa_pos: # data in other pe
                            if self.Computation_order[pre_event_idx].position_idx not in preceding_cu:
                                preceding_cu[self.Computation_order[pre_event_idx].position_idx] = [pre_event_idx]
                            else:
                                preceding_cu[self.Computation_order[pre_event_idx].position_idx].append(pre_event_idx)
                    for cu_idx in preceding_cu:
                        edram_wr_event_idx = len(self.Computation_order)
                        for pre_event_idx in preceding_cu[cu_idx]:
                            self.Computation_order[pre_event_idx].proceeding_event.append(edram_wr_event_idx)
                        pe_idx = cu_idx[0:4]

                        edram_wr_pe_pos  = pe_idx
                        edram_wr_inputs  = 0
                        edram_wr_outputs = [[nfilter, 0, 0, "u"]]
                        edram_wr_preceding_count = len(preceding_cu[cu_idx])
                        event = EventMetaData("edram_wr", edram_wr_pe_pos, edram_wr_preceding_count, [edram_wr_event_idx+1], nlayer, edram_wr_inputs, edram_wr_outputs)
                        self.Computation_order.append(event)

                        source_pe_idx = pe_idx
                        transfer_inputs =  0 #[[nfilter, 0, 0, "u"]]
                        transfer_outputs = [[nfilter, 0, 0, "u"]]
                        event = EventMetaData("data_transfer", [source_pe_idx, do_pe_saa_pos], 1, [], nlayer, transfer_inputs, transfer_outputs)
                        self.Computation_order.append(event)

                ### Event: pe_saa
                    pe_saa_event_idx = len(self.Computation_order)
                    preceding_cu_idx = list()
                    preceding_tmp_data = list()
                    for pre_event_idx in preceding_list:
                        if self.Computation_order[pre_event_idx].position_idx[:-2] == do_pe_saa_pos: # in same PE
                            self.Computation_order[pre_event_idx].proceeding_event.append(pe_saa_event_idx)
                        else:
                            data_transfer_id = self.Computation_order[pre_event_idx].proceeding_event[0] + 1
                            preceding_tmp_data.append(self.Computation_order[data_transfer_id].outputs[0]) # data transfer output
                        if self.Computation_order[pre_event_idx].position_idx not in preceding_cu_idx:
                            preceding_cu_idx.append(self.Computation_order[pre_event_idx].position_idx)
                    # add dependency
                    for idx in range(start_append_idx+1, pe_saa_event_idx, 2):
                        self.Computation_order[idx].proceeding_event.append(pe_saa_event_idx)
                    pe_saa_inputs  = [preceding_cu_idx, preceding_tmp_data]
                    pe_saa_outputs = [[nfilter, 0, 0]]
                    event = EventMetaData("pe_saa", do_pe_saa_pos, pe_saa_preceding_count, [], nlayer, pe_saa_inputs, pe_saa_outputs)
                    self.Computation_order.append(event)

                ### Event: activation
                    do_act_pos = do_pe_saa_pos 
                    act_preceding_count = 1
                    act_inputs  = 0 #[[0, 0, nfilter]]
                    act_outputs = 0 #[[0, 0, nfilter]]
                    # add dependency
                    act_event_idx = len(self.Computation_order)
                    self.Computation_order[pe_saa_event_idx].proceeding_event.append(act_event_idx)

                    event = EventMetaData("activation", do_act_pos, act_preceding_count, [], nlayer, act_inputs, act_outputs)
                    self.Computation_order.append(event)

                ### Event: edram_wr
                    edram_wr_event_idx = len(self.Computation_order)
                    self.Computation_order[act_event_idx].proceeding_event.append(edram_wr_event_idx)

                    do_edram_wr_pos = do_act_pos
                    edram_wr_preceding_count = 1
                    edram_wr_inputs  = 0
                    edram_wr_outputs = [[nfilter, 0, 0]]
                    event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [], nlayer, edram_wr_inputs, edram_wr_outputs)
                    self.Computation_order.append(event) 

                    # dependecy matrix
                    if nlayer+1 < self.model_info.layer_length:
                        if self.feature_mat[nlayer][nfilter][0][0] == 0.0:
                            self.feature_mat[nlayer][nfilter][0][0] = []
                        self.feature_mat[nlayer][nfilter][0][0].append(edram_wr_event_idx)

            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                for rty_idx in range(self.hd_info.Router_num_y):
                    for rtx_idx in range(self.hd_info.Router_num_x):
                        for pey_idx in range(self.hd_info.PE_num_y):
                            for pex_idx in range(self.hd_info.PE_num_x):
                                pe_pos = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                for mapping_data in self.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx]:
                                    if mapping_data.nlayer == nlayer:
                ### Event: edram_rd_pool, data_transfer
                                        for inputs in mapping_data.inputs:
                                            erp_preceding_count = 0
                                            start_append_idx = len(self.Computation_order)
                                            for input_data in inputs:
                                                d = input_data[1:]
                                                preceding_list = self.feature_mat[nlayer-1][d[0]][d[1]][d[2]]
                                                for pre_event in preceding_list:
                                                    edram_wr_event = self.Computation_order[pre_event]
                                                    data_transfer_source      = edram_wr_event.position_idx
                                                    data_transfer_destination = pe_pos
                                                    if data_transfer_source != data_transfer_destination:
                                                        ## Event: data_transfer
                                                        data_transfer_event_idx = len(self.Computation_order)
                                                        edram_wr_event.proceeding_event.append(data_transfer_event_idx)
                                                        data_transfer_preceding_count = 1
                                                        #data_transfer_inputs  = edram_wr_event.outputs
                                                        data_transfer_inputs  = 0
                                                        data_transfer_outputs = edram_wr_event.outputs
                                                        event = EventMetaData("data_transfer", [data_transfer_source, data_transfer_destination], data_transfer_preceding_count, [], nlayer-1, data_transfer_inputs, data_transfer_outputs)
                                                        self.Computation_order.append(event)

                                            # dependency
                                            erp_event_idx = len(self.Computation_order)
                                            for input_data in inputs:
                                                d = input_data[1:]
                                                preceding_list = self.feature_mat[nlayer-1][d[0]][d[1]][d[2]]
                                                
                                                for pre_event in preceding_list:
                                                    edram_wr_event = self.Computation_order[pre_event]
                                                    data_transfer_source      = edram_wr_event.position_idx
                                                    data_transfer_destination = pe_pos
                                                    if data_transfer_source == data_transfer_destination:
                                                        self.Computation_order[pre_event].proceeding_event.append(erp_event_idx)
                                                    erp_preceding_count += 1

                                            for idx in range(start_append_idx, erp_event_idx):
                                                self.Computation_order[idx].proceeding_event.append(erp_event_idx)

                                            erp_event_idx = len(self.Computation_order)
                                            erp_position_idx = pe_pos
                                            erp_input_sequence = inputs
                                            erp_output_sequence = 0
                                            event = EventMetaData("edram_rd_pool", erp_position_idx, erp_preceding_count, [], nlayer, erp_input_sequence, erp_output_sequence)
                                            self.Computation_order.append(event)

                                            #if self.isFreeBuffer:
                                            # input require
                                            pe_id = pe_pos[3] + pe_pos[2]*self.hd_info.PE_num_x + \
                                                    pe_pos[1]*self.hd_info.PE_num + pe_pos[0]*self.hd_info.PE_num*self.hd_info.Router_num_x
                                            for data in erp_input_sequence:
                                                d = data[1:]
                                                pos = d[1] + d[0]*self.model_info.input_w[nlayer] + d[2]*self.model_info.input_w[nlayer]*self.model_info.input_h[nlayer] # w + h*width + c*height*width
                                                self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1
                ### Event: pooling
                                            # pool_event_index = len(self.Computation_order)
                                            # self.Computation_order[erp_event_idx].proceeding_event.append(pool_event_index)

                                            # pool_position_idx = pe_pos
                                            # pool_preceding_count = 1
                                            # pool_input_sequence = 0
                                            # pool_output_sequence = 0
                                            # event = EventMetaData("pooling", pool_position_idx, pool_preceding_count, [], nlayer, pool_input_sequence, pool_output_sequence)
                                            # self.Computation_order.append(event)
                ### Event: edram_wr
                                            edram_wr_event_idx = len(self.Computation_order)
                                            #self.Computation_order[pool_event_index].proceeding_event.append(edram_wr_event_idx)
                                            self.Computation_order[erp_event_idx].proceeding_event.append(edram_wr_event_idx)

                                            edram_wr_position_idx = pe_pos
                                            edram_wr_preceding_count = 1
                                            edram_wr_input_sequence  = 0
                                            edram_wr_output_sequence = [[inputs[0][1] // self.model_info.pooling_strides[nlayer], 
                                                                        inputs[0][2] // self.model_info.pooling_strides[nlayer], 
                                                                        inputs[0][3]]]

                                            if nlayer+1 < self.model_info.layer_length:
                                                if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                                    if self.feature_mat[nlayer][edram_wr_output_sequence[0][0]][edram_wr_output_sequence[0][1]][edram_wr_output_sequence[0][2]] == 0.0:
                                                        self.feature_mat[nlayer][edram_wr_output_sequence[0][0]][edram_wr_output_sequence[0][1]][edram_wr_output_sequence[0][2]] = []
                                                    self.feature_mat[nlayer][edram_wr_output_sequence[0][0]][edram_wr_output_sequence[0][1]][edram_wr_output_sequence[0][2]].append(edram_wr_event_idx)
                                                else:
                                                    seq = edram_wr_output_sequence[0][0] * self.model_info.input_w[nlayer+1] + edram_wr_output_sequence[0][1] + edram_wr_output_sequence[0][2] * self.model_info.input_h[nlayer+1] * self.model_info.input_w[nlayer+1]
                                                    edram_wr_output_sequence  = [[seq, 0, 0]]
                                                    if self.feature_mat[nlayer][seq][0][0] == 0.0:
                                                        self.feature_mat[nlayer][seq][0][0] = []
                                                    self.feature_mat[nlayer][seq][0][0].append(edram_wr_event_idx)
                                            event = EventMetaData("edram_wr", edram_wr_position_idx, edram_wr_preceding_count, [], nlayer, edram_wr_input_sequence, edram_wr_output_sequence)
                                            self.Computation_order.append(event)

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

        if True:
            for e in self.Computation_order:
                print(self.Computation_order.index(e), e)
                # if self.Computation_order.index(e) == 10:
                #     break

    def __str__(self):
        return str(self.__dict__)
