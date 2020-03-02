from HardwareMetaData import HardwareMetaData
from ModelConfig import ModelConfig
from PE import PE

from EventMetaData import EventMetaData
from FetchEvent import FetchEvent
from Interconnect import Interconnect
from Packet import Packet

import numpy as np
from math import ceil, floor
import os, csv, copy, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from multiprocessing import Process
import time

class Controller(object):
    def __init__(self, ordergenerator, trace, mapping_str, scheduling_str):
        self.ordergenerator = ordergenerator
        self.trace = trace
        self.mapping_str = mapping_str
        self.scheduling_str = scheduling_str
        if self.scheduling_str == "Pipeline":
            self.isPipeLine = True
        else:
            self.isPipeLine = False
        self.Computation_order = self.ordergenerator.Computation_order
        print("Computation order length:", len(self.Computation_order))
        self.hd_info = HardwareMetaData()
        self.input_bit = self.ordergenerator.model_info.input_bit
        self.cycle_ctr = 0
        # self.edram_read_cycles = ceil(
        #     self.hd_info.Xbar_num * self.hd_info.Xbar_h *
        #     self.input_bit * self.hd_info.eDRAM_read_latency /
        #     self.hd_info.cycle_time) # edram read to ir需要幾個cycle
        self.edram_read_data  = floor(self.hd_info.eDRAM_read_bits / self.input_bit) # 1個cycle可以讀多少data
        self.edram_write_data = floor(self.hd_info.eDRAM_write_bits / self.input_bit) # 1個cycle可以寫多少data
        self.done_event = 0

        self.PE_array = []
        for rty_idx in range(self.hd_info.Router_num_y):
            for rtx_idx in range(self.hd_info.Router_num_x):
                for pey_idx in range(self.hd_info.PE_num_y):
                    for pex_idx in range(self.hd_info.PE_num_x):
                        pe_pos = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        pe = PE(pe_pos, ModelConfig().input_bit)
                        self.PE_array.append(pe)

        self.fetch_array = []
        # Interconnect
        self.Total_energy_interconnect = 0
        self.interconnect = Interconnect(self.hd_info.Router_num_y, self.hd_info.Router_num_x, self.input_bit)
        self.interconnect_step = self.hd_info.Router_flit_size / self.input_bit * self.hd_info.cycle_time * self.hd_info.Frequency # scaling from ISAAC
        self.interconnect_step = floor(self.interconnect_step)
        self.data_transfer_trigger = []
        self.data_transfer_erp = []
        # Pipeline control
        if not self.isPipeLine:
            self.pipeline_layer_stage = 0
            self.events_each_layer = []
            for layer in range(self.ordergenerator.model_info.layer_length):
                self.events_each_layer.append(0)
            for e in self.Computation_order:
                self.events_each_layer[e.nlayer] += 1
            self.this_layer_event_ctr = 0
            self.this_layer_cycle_ctr = 0
            self.cycles_each_layer = []
            print("events_each_layer:", self.events_each_layer)

        # Utilization
        self.pe_state_for_plot = [[], []]
        # self.cu_state_for_plot = [[], []]
        self.xb_state_for_plot = [[], []]
        self.buffer_size = []
        self.buffer_size_i = []
        for i in range(len(self.PE_array)):
            self.buffer_size.append([])
            self.buffer_size_i.append([])
        self.max_buffer_size = 0 # num of data
        self.max_buffer_size_i = 0 # num of data
        self.check_buffer_pe_set = set()

        # execute event
        self.erp_rd      = set()
        self.erp_cu_op   = set()
        self.erp_pe_saa  = set()
        self.erp_act     = set()
        self.erp_wr      = set()
        self.erp_pool_rd = set()

        self.check_state_pe = set()

        # Trigger pe
        self.trigger_edram_rd = set()
        self.trigger_cu_op    = set()
        self.trigger_pe_saa   = set()
        self.trigger_edram_wr = set()
        self.trigger_act      = set()
        self.trigger_pool_rd  = set()

        self.trigger_next_layer = False
        
        self.mp_info = ordergenerator.mp_info
        
        # 把input feature map放到buffer中
        for pe in self.PE_array:
            rty_idx, rtx_idx = pe.position[0], pe.position[1]
            pey_idx, pex_idx = pe.position[2], pe.position[3]
            feature_m = np.zeros((ModelConfig().input_h, ModelConfig().input_w, ModelConfig().input_c))
            for cuy_idx in range(self.hd_info.CU_num_y):
                for cux_idx in range(self.hd_info.CU_num_x):
                    for xby_idx in range(self.hd_info.Xbar_num_y):
                        for xbx_idx in range(self.hd_info.Xbar_num_x):
                            mapping = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][0] # layer0
                            for mp in mapping:
                                for inp in mp.inputs:
                                    feature_m[inp[1]][inp[2]][inp[3]] = 1

            for h in range(feature_m.shape[0]):
                for w in range(feature_m.shape[1]):
                    for c in range(feature_m.shape[2]):
                        if feature_m[h][w][c] == 1:
                            data = [0, [h, w, c]]
                            pe.edram_buffer.buffer.append(data)
                            pe.edram_buffer_i.buffer.append(data)

    def run(self):
        for e in self.Computation_order:
            if e.event_type == 'edram_rd_ir':
                if e.preceding_event_count == 0:
                    pos = e.position_idx
                    rty, rtx, pey, pex, cuy, cux = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
                    pe_idx = pex + pey * self.hd_info.PE_num_x + rtx * self.hd_info.PE_num + rty * self.hd_info.Router_num_x * self.hd_info.PE_num
                    cu_idx = cux + cuy * self.hd_info.CU_num_x
                    pe = self.PE_array[pe_idx]
                    pe.CU_array[cu_idx].edram_rd_ir_erp.append(e)

                    if cu_idx not in pe.idle_eventQueuing_CU:
                        pe.idle_eventQueuing_CU.append(cu_idx)
                    
                    self.erp_rd.add(pe)

                    # for performance analysis
                    if cu_idx not in pe.eventQueuing_CU: 
                        pe.eventQueuing_CU.append(cu_idx)
            if e.nlayer != 0:
                break
        
        # self.cycle_ctr += self.hd_info.Fetch_cycle
        # for pe_idx in range(len(pe_fetch_dict)):
        #     des = self.PE_array[pe_idx].position
        #     src = (0, des[1], -1, -1)

        #     for d in pe_fetch_dict[pe_idx]:
        #         data = [0, d[0]]
        #         pro_event_idx = d[1]
        #         packet = Packet(src, des, data, pro_event_idx)
        #         self.interconnect.input_packet(packet)
        
        t_edram = 0
        t_cuop = 0
        t_pesaa = 0
        t_act = 0
        t_wr = 0
        t_it = 0
        t_fe = 0
        t_tr = 0
        t_st = 0
        t_buf = 0
        t_poo = 0
        t_oth = 0
        t_ = time.time()
        start_time = time.time()
        layer = 0
        while True:
            if self.cycle_ctr % 100000 == 0:
                if self.done_event == 0:
                    pass
                else:
                    print("layer:", layer)
                    print("Cycle",self.cycle_ctr, "Done event:", self.done_event, "time per event", (time.time()-start_time)/self.done_event, "time per cycle", (time.time()-start_time)/self.cycle_ctr)
                    print("edram:", t_edram, "t_cuop", t_cuop, "pesaa", t_pesaa, "act", t_act, "wr", t_wr)
                    print("iterconeect", t_it, "fetch", t_fe, "trigger", t_tr, "state", t_st)
                    print("buffer", t_buf, "pool", t_poo, "other", t_oth)
                    print("t:", time.time()-t_)
                    t_edram, t_cuop, t_pesaa, t_act, t_wr = 0, 0, 0, 0, 0
                    t_it, t_fe, t_tr, t_st, t_buf, t_poo = 0, 0, 0, 0, 0, 0
                    t_oth = 0
                    t_ = time.time()

            self.cycle_ctr += 1
            if not self.isPipeLine:
                self.this_layer_cycle_ctr += 1
            if self.trace:
                print("cycle:", self.cycle_ctr)

            staa = time.time()
            self.event_edram_rd()
            t_edram += time.time() - staa

            staa = time.time()
            self.event_cu_op()
            self.performance_analysis()
            t_cuop += time.time() - staa

            staa = time.time()
            self.event_pe_saa()
            t_pesaa += time.time() - staa

            staa = time.time()
            self.event_act()
            t_act += time.time() - staa

            staa = time.time()
            self.event_edram_wr()
            t_wr += time.time() - staa

            staa = time.time()
            self.event_edram_rd_pool()
            t_poo += time.time() - staa
            
            staa = time.time()
            self.process_interconnect()
            t_it += time.time() - staa

            staa = time.time()
            self.event_transfer()
            t_tr += time.time() - staa

            staa = time.time()
            #self.fetch()
            t_fe += time.time() - staa

            ### Pipeline stage control ###
            if not self.isPipeLine:
                if self.this_layer_event_ctr == self.events_each_layer[self.pipeline_layer_stage]:
                    layer += 1
                    self.pipeline_layer_stage += 1
                    self.cycles_each_layer.append(self.this_layer_cycle_ctr)
                    self.this_layer_event_ctr = 0
                    self.this_layer_cycle_ctr = 0
                    self.trigger_next_layer = True

            staa = time.time()
            self.trigger()
            t_tr += time.time() - staa

            ### Record PE State ###
            staa = time.time()
            for pe in self.check_state_pe:
                self.pe_state_for_plot[0].append(self.cycle_ctr)
                self.pe_state_for_plot[1].append(self.PE_array.index(pe))
            self.check_state_pe = set()
            t_st += time.time() - staa

            ### Reset PE ###
            staa = time.time()
            for pe in self.PE_array:
                pe.this_cycle_read_data = 0
            t_oth += time.time() - staa

            ### Buffer utilization
            staa = time.time()

            for pe in self.check_buffer_pe_set:
                pe.edram_buffer_i.buffer_size_util[0].append(self.cycle_ctr)
                pe.edram_buffer_i.buffer_size_util[1].append(len(pe.edram_buffer_i.buffer))
                pe.edram_buffer.buffer_size_util[0].append(self.cycle_ctr)
                pe.edram_buffer.buffer_size_util[1].append(len(pe.edram_buffer.buffer))
            self.check_buffer_pe_set = set()
            t_buf += time.time() - staa

            ### Finish
            if self.done_event == len(self.Computation_order):
                break

    def buffer_util(self):
        pe_idx = 0
        for pe in self.PE_array:
            bs   = len(pe.edram_buffer.buffer)
            bs_i = len(pe.edram_buffer_i.buffer)
            self.buffer_size  [pe_idx].append(bs)
            self.buffer_size_i[pe_idx].append(bs_i)
            self.max_buffer_size   = max(bs,   self.max_buffer_size)
            self.max_buffer_size_i = max(bs_i, self.max_buffer_size_i)
            pe_idx += 1

    def event_edram_rd(self):
        erp = set()
        for pe in self.erp_rd:
            # 正在處理哪一個event
            if not pe.edram_rd_event:
                cu_idx = pe.idle_eventQueuing_CU.popleft()
                pe.edram_rd_cu_idx = cu_idx
                cu = pe.CU_array[cu_idx]
                event = cu.edram_rd_ir_erp.popleft()
                pe.edram_rd_event = event
                # if event.nlayer == 0:
                #     fetch_data = []
                #     for data in event.inputs:
                #         if not pe.edram_buffer.check([event.nlayer, data]): # Q1: 這很慢
                #             fetch_data.append(data)
                #     if fetch_data: # 有資料不在buffer要fetch
                #         event.data_is_transfer += len(fetch_data)
                #         self.fetch_array.append([FetchEvent(event), fetch_data])
            else:
                cu_idx = pe.edram_rd_cu_idx
                event = pe.edram_rd_event
                cu = pe.CU_array[cu_idx]
            
            num_data = len(event.inputs)
            if not pe.data_to_ir_ing: # 還沒開始從CU傳資料到IR
                if event.data_is_transfer != 0: # 資料是否還在transfer？
                    pass
                else: # do edram read
                    if self.trace:
                        print("\tdo edram_rd_ir, nlayer:", event.nlayer,", pos:", event.position_idx)
                    pe.data_to_ir_ing = True
                    self.check_state_pe.add(pe)

                    # Energy
                    pe.Edram_buffer_energy += self.hd_info.Energy_edram_buffer * self.input_bit * num_data
                    pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit * num_data
                    pe.CU_IR_energy += self.hd_info.Energy_ir_in_cu * self.input_bit * num_data # write

                    # 要幾個cycle讀完
                    pe.edram_read_cycles = ceil(num_data / self.edram_read_data)

                    # 判斷是否完成(只有read一個cycle內完成才會進入這邊)
                    pe.edram_rd_cycle_ctr += 1
                    if pe.edram_rd_cycle_ctr == pe.edram_read_cycles: # 完成edram read
                        if self.trace:
                            print("\t\tfinish edram read")
                        pe.this_cycle_read_data = num_data % self.edram_read_data
                        self.done_event += 1
                        if not self.isPipeLine:
                            self.this_layer_event_ctr += 1

                        pe.edram_rd_cycle_ctr = 0
                        pe.edram_rd_event = None #可處理下一個
                        pe.edram_rd_cu_idx = None
                        pe.data_to_ir_ing = False
                        # cu.state = True

                        # trigger cu operation
                        proceeding_index = event.proceeding_event[0] # 只會trigger一個cu operation
                        pro_event = self.Computation_order[proceeding_index]
                        pro_event.current_number_of_preceding_event += 1
                        pe.cu_op_trigger = pro_event
                        #if pe not in self.trigger_cu_op:
                        self.trigger_cu_op.add(pe)
                        # Free buffer (ideal)
                        pe_id = self.PE_array.index(pe)
                        nlayer = event.nlayer
                        if self.ordergenerator.model_info.layer_list[nlayer].layer_type == "convolution":
                            for d in event.inputs:
                                pos = d[1] + d[0]*self.ordergenerator.model_info.input_w[nlayer] + d[2]*self.ordergenerator.model_info.input_w[nlayer]*self.ordergenerator.model_info.input_h[nlayer] # w + h*width + c*height*width
                                self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                                if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                    self.PE_array[pe_id].edram_buffer_i.buffer.remove([nlayer, d])
                                    self.check_buffer_pe_set.add(self.PE_array[pe_id])
                        elif self.ordergenerator.model_info.layer_list[nlayer].layer_type == "fully":
                            for d in event.inputs:
                                pos = d[0]
                                self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                                if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                    self.PE_array[pe_id].edram_buffer_i.buffer.remove([nlayer, d])
                                    self.check_buffer_pe_set.add(self.PE_array[pe_id])
                    else:
                        pe.this_cycle_read_data = self.edram_read_data # 這個cycle read多少data量
            else: # CU傳資料到IR
                pe.edram_rd_cycle_ctr += 1
                self.check_state_pe.add(pe)
                if pe.edram_rd_cycle_ctr == pe.edram_read_cycles: # 完成edram read
                    if self.trace:
                        print("\tfinish edram_rd_ir, nlayer:", event.nlayer,", pos:", event.position_idx)
                    pe.this_cycle_read_data = num_data % self.edram_read_data
                    self.done_event += 1
                    if not self.isPipeLine:
                        self.this_layer_event_ctr += 1

                    pe.edram_rd_cycle_ctr = 0
                    pe.edram_rd_event = None
                    pe.edram_rd_cu_idx = None
                    pe.data_to_ir_ing = False
                    #cu.state = True

                    # trigger cu operation
                    proceeding_index = event.proceeding_event[0] # 只會trigger一個cu operation
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1
                    pe.cu_op_trigger = pro_event
                    #if pe not in self.trigger_cu_op:
                    self.trigger_cu_op.add(pe)

                    # Free buffer (ideal)
                    pe_id = self.PE_array.index(pe)
                    nlayer = event.nlayer
                    if self.ordergenerator.model_info.layer_list[nlayer].layer_type == "convolution":
                        for d in event.inputs:
                            pos = d[1] + d[0]*self.ordergenerator.model_info.input_w[nlayer] + d[2]*self.ordergenerator.model_info.input_w[nlayer]*self.ordergenerator.model_info.input_h[nlayer] # w + h*width + c*height*width
                            self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                            if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                self.PE_array[pe_id].edram_buffer_i.buffer.remove([nlayer, d])
                                self.check_buffer_pe_set.add(self.PE_array[pe_id])
                    elif self.ordergenerator.model_info.layer_list[nlayer].layer_type == "fully":
                        for d in event.inputs:
                            pos = d[0]
                            self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                            if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                self.PE_array[pe_id].edram_buffer_i.buffer.remove([nlayer, d])
                                self.check_buffer_pe_set.add(self.PE_array[pe_id])
                else:
                    pe.this_cycle_read_data = self.edram_read_data # 這個cycle read多少data量
            # 下個cycle還要不要檢查此pe
            if pe.edram_rd_event: # 有event正在處理
                erp.add(pe)
            elif pe.idle_eventQueuing_CU:
                erp.add(pe)
        self.erp_rd = erp

    def event_cu_op(self):
        erp = set()
        for pe in self.erp_cu_op:
            #pe.state = True
            self.check_state_pe.add(pe)
            cu_op_list = []
            for cu in pe.cu_op_list:
                if cu.finish_cycle == 0 and cu.cu_op_event != 0: # cu operation start
                    if self.trace:
                        pass
                        print("\tcu operation start", "pos:", cu.cu_op_event.position_idx)
                    cu.finish_cycle = self.cycle_ctr - 1 + cu.cu_op_event.inputs + 2 # +2: pipeline 最後兩個 stage
                    # pe.data_to_ir_ing = False
                    ## Energy
                    ou_num_dict = cu.cu_op_event.outputs
                    for xb_idx in ou_num_dict:
                        ou_num = ou_num_dict[xb_idx]
                        pe.CU_dac_energy += self.hd_info.Energy_ou_dac * ou_num
                        pe.CU_crossbar_energy += self.hd_info.Energy_ou_crossbar * ou_num
                        pe.CU_adc_energy += self.hd_info.Energy_ou_adc * ou_num
                        pe.CU_shift_and_add_energy += self.hd_info.Energy_ou_ssa * ou_num
                        
                        pe.CU_IR_energy += self.hd_info.Energy_ir_in_cu * ou_num * self.hd_info.OU_h 
                        pe.CU_OR_energy += self.hd_info.Energy_or_in_cu * ou_num * self.hd_info.OU_w * self.hd_info.ADC_resolution

                    cu_id = self.PE_array.index(pe) * self.hd_info.CU_num + pe.CU_array.index(cu)
                    # self.cu_state_for_plot[0].append(self.cycle_ctr)
                    # self.cu_state_for_plot[1].append(cu_id)
                    for xb_idx in ou_num_dict:
                        xb_id = cu_id * self.hd_info.Xbar_num + xb_idx
                        ou_num = ou_num_dict[xb_idx]
                        for c in range(ou_num):
                            self.xb_state_for_plot[0].append(self.cycle_ctr+c)
                            self.xb_state_for_plot[1].append(xb_id)
                    cu_op_list.append(cu)

                elif cu.finish_cycle != 0 and cu.cu_op_event != 0: ## doing cu operation
                    if cu.finish_cycle == self.cycle_ctr: # finish
                        if self.trace:
                            pass
                            print("\tcu operation finish", "pos:", cu.cu_op_event.position_idx)
                        self.done_event += 1
                        if not self.isPipeLine:
                            self.this_layer_event_ctr += 1
                        cu_idx = pe.CU_array.index(cu)
                        if cu.edram_rd_ir_erp:
                            pe.idle_eventQueuing_CU.append(cu_idx)
                            #if pe not in self.erp_rd:
                            self.erp_rd.add(pe)

                        ### add next event counter: pe_saa
                        for proceeding_index in cu.cu_op_event.proceeding_event:
                            pro_event = self.Computation_order[proceeding_index]
                            pro_event.current_number_of_preceding_event += 1
                            if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                if self.trace:
                                    pass
                                    #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                if pro_event.event_type == "pe_saa":
                                    pe.pe_saa_trigger.append([pro_event, []])
                                    #if pe not in self.trigger_pe_saa:
                                    self.trigger_pe_saa.add(pe)
                                elif pro_event.event_type == "edram_wr":
                                    pe.edram_wr_trigger.append([pro_event, []])
                                    #if pe not in self.trigger_edram_wr:
                                    self.trigger_edram_wr.add(pe)

                        cu.finish_cycle = 0
                        cu.cu_op_event = 0
                        cu.state = False
                        # pe.cu_op_list.remove(cu)
                    else:
                        cu_op_list.append(cu)
                        if self.trace:
                            print("\tcu operation")
                
                #cu_id = self.PE_array.index(pe) * self.hd_info.CU_num + pe.CU_array.index(cu)
                # self.cu_state_for_plot[0].append(self.cycle_ctr)
                # self.cu_state_for_plot[1].append(cu_id)

                # performance analysis
                isEventWaiting = False
                for event in cu.edram_rd_ir_erp:
                    if event.data_is_transfer == 0: # 有event資料已經ready但還不能做
                        isEventWaiting = True
                        break
                if isEventWaiting:
                    cu.wait_resource_time += 1
                else:
                    cu.pure_computation_time += 1
            pe.cu_op_list = cu_op_list
            if pe.cu_op_list:
                erp.add(pe)
        self.erp_cu_op = erp

    def performance_analysis(self):
        for pe in self.erp_rd:
            for cu_idx in pe.idle_eventQueuing_CU:
                cu = pe.CU_array[cu_idx]
                cu.wait_transfer_time += 1 # CU在等資料傳到
            if pe.edram_rd_event:
                event = pe.edram_rd_event
                cuy, cux = event.position_idx[4], event.position_idx[5]
                cu_idx = cux + cuy * self.hd_info.CU_num_x
                cu = pe.CU_array[cu_idx]
                cu.wait_transfer_time += 1

    def event_pe_saa(self):
        erp = set()
        for pe in self.erp_pe_saa:
            pe_saa_erp = []
            #pe.state = True
            self.check_state_pe.add(pe)
            for event in pe.pe_saa_erp: # 1個cycle全部做完
                if event.data_is_transfer != 0: # 此event的資料正在傳輸
                    pe_saa_erp.append(event)
                    continue
                if self.trace:
                    pass
                    print("\tdo pe_saa, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event)) 

                self.done_event += 1
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1
                
                
                #pe.pe_saa_erp.remove(event)
                saa_amount = event.inputs[0] # inputs[0]: 前面的cu index
                #rm_data_list = event.inputs[1]    # 要移除的資料

                # Energy
                pe.Or_energy += self.hd_info.Energy_or * self.input_bit * saa_amount
                pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit * saa_amount
                pe.PE_shift_and_add_energy += self.hd_info.Energy_shift_and_add * saa_amount
                pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit * saa_amount
                pe.Or_energy += self.hd_info.Energy_or * self.input_bit * saa_amount

                ### add next event counter: activation
                for proceeding_index in event.proceeding_event:
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1
                    
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            pass
                            #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                        if pro_event.event_type == "activation":
                            pe.activation_trigger.append([pro_event, []])
                            #if pe not in self.trigger_act:
                            self.trigger_act.add(pe)

                        elif pro_event.event_type == "edram_wr":
                            pe.edram_wr_trigger.append([pro_event, []])
                            #if pe not in self.trigger_edram_wr:
                            self.trigger_edram_wr.add(pe)

                # Free buffer (ideal)
                # for d in rm_data_list:
                #     try:
                #         pe.edram_buffer_i.buffer.remove([event.nlayer, d])
                #         self.check_buffer_pe_set.add(pe)
                #     except ValueError:
                #         pass
                    # if [event.nlayer, d] in pe.edram_buffer_i.buffer:
                    #     pe.edram_buffer_i.buffer.remove([event.nlayer, d])
                    #     self.check_buffer_pe_set.add(pe)

            pe.pe_saa_erp = pe_saa_erp
            #if pe.pe_saa_erp:
            erp.add(pe)
        self.erp_pe_saa = erp
 
    def event_act(self):
        erp = set()
        for pe in self.erp_act:
            if not pe.activation_erp:
                continue
            #pe.state = True
            self.check_state_pe.add(pe)
            if pe.activation_epc <= len(pe.activation_erp):
                do_act_num = pe.activation_epc
            else:
                do_act_num = len(pe.activation_erp)
            #act_erp_copy = pe.activation_erp.copy()
            for idx in range(do_act_num):
                #event = act_erp_copy[idx]
                event = pe.activation_erp.popleft()
                if self.trace:
                    pass
                    print("\tdo activation, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                self.done_event += 1
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1

                # Energy
                pe.Or_energy += self.hd_info.Energy_or * self.input_bit
                pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit
                pe.Activation_energy += self.hd_info.Energy_activation

                ### add next event counter: edram_wr
                for proceeding_index in event.proceeding_event:
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1
                    
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            pass
                            #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx, self.Computation_order.index(pro_event))
                        pe.edram_wr_trigger.append([pro_event, []])
                        #if pe not in self.trigger_edram_wr:
                        self.trigger_edram_wr.add(pe)

            #if pe.activation_erp:
            erp.add(pe)
        self.erp_act = erp

    def event_edram_wr(self):
        erp = set()
        for pe in self.erp_wr:
            #pe.state = True
            self.check_state_pe.add(pe)
            if self.edram_write_data <= len(pe.edram_wr_erp):
                do_wr_num = self.edram_write_data
            else:
                do_wr_num = len(pe.edram_wr_erp)
            for idx in range(do_wr_num):
                event = pe.edram_wr_erp.popleft()
                self.done_event += 1
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1

                if self.trace:
                    pass
                    print("\tdo edram_wr, pe_pos:", pe.position, "layer:", event.nlayer, \
                    ",order index:", self.Computation_order.index(event), "data:", event.outputs)
                
                # Energy
                pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit
                pe.Edram_buffer_energy += self.hd_info.Energy_edram_buffer * self.input_bit

                if len(event.outputs[0]) == 4: #and event.outputs[0][3] == "u": # same layer transfer
                    pe.edram_buffer.put([event.nlayer, event.outputs[0]])
                    pe.edram_buffer_i.put([event.nlayer, event.outputs[0]])
                else:
                    pe.edram_buffer.put([event.nlayer+1, event.outputs[0]])
                    pe.edram_buffer_i.put([event.nlayer+1, event.outputs[0]])

                ### add next event counter: edram_rd_ir, edram_rd_pool, data_transfer
                for proceeding_index in event.proceeding_event:
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1
                    
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            pass
                            #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                        pos = pro_event.position_idx
                        if pro_event.event_type == "edram_rd_ir":
                            cu_y, cu_x = pos[4], pos[5]
                            cu_idx = cu_x + cu_y * self.hd_info.CU_num_x
                            pe.edram_rd_ir_trigger.append([pro_event, [cu_idx]])
                            #if pe not in self.trigger_edram_rd:
                            self.trigger_edram_rd.add(pe)
                        elif pro_event.event_type == "edram_rd_pool":
                            pe.edram_rd_pool_trigger.append([pro_event, []])
                            #if pe not in self.trigger_pool_rd:
                            self.trigger_pool_rd.append(pe)
                        elif pro_event.event_type == "data_transfer":
                            self.data_transfer_trigger.append([pro_event, []])
            
            self.check_buffer_pe_set.add(pe)

            #if pe.edram_wr_erp:
            erp.add(pe)
        self.erp_wr = erp

    def event_edram_rd_pool(self):
        erp = set()
        for pe in self.erp_pool_rd:
            pool_read_data = self.edram_read_data - pe.this_cycle_read_data # 這個cycle pooling還能夠read多少資料
            self.check_state_pe.add(pe)
            for event in pe.edram_rd_pool_erp.copy():
                if event.data_is_transfer != 0: # 此event的資料正在傳輸
                    continue
                # fetch_data = []
                # for inp in event.inputs:
                #     data = inp[1:]
                #     if not pe.edram_buffer.check([event.nlayer, data]):
                #         # Data not in buffer
                #         if self.trace:
                #             print("\tData not ready for edram_rd_pool. Data: layer", event.nlayer, event.event_type, [event.nlayer, data])
                #         fetch_data.append(data)
                # if fetch_data:
                #     event.data_is_transfer += len(fetch_data)
                #     self.fetch_array.append([FetchEvent(event), fetch_data])
                #     continue
                
                num_data = len(event.inputs)
                pool_read_data = pool_read_data - num_data
                if pool_read_data < 0: #沒辦法再read了
                    break
                self.done_event += 1
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1
                if self.trace:
                    print("\tdo edram_rd_pool, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                

                # Energy
                pe.Edram_buffer_energy += self.hd_info.Energy_edram_buffer * self.input_bit * num_data
                pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit * num_data
                pe.Pooling_energy += self.hd_info.Energy_pooling

                #pe.state = True
                self.check_state_pe.add(pe)
                pe.edram_rd_pool_erp.remove(event)

                ### add next event counter: pooling
                for proceeding_index in event.proceeding_event:
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1

                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            pass
                            #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                        pe.edram_wr_trigger.append([pro_event, []])
                        #if pe not in self.trigger_edram_wr:
                        self.trigger_edram_wr.add(pe)
                
                # Free buffer (ideal)
                pe_id = self.PE_array.index(pe)
                nlayer = event.nlayer
                for d in event.inputs:
                    pos = d[2] + d[1]*self.ordergenerator.model_info.input_w[nlayer] + d[3]*self.ordergenerator.model_info.input_w[nlayer]*self.ordergenerator.model_info.input_h[nlayer] # w + h*width + c*height*width
                    self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                    if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                        data = d[1:]
                        self.PE_array[pe_id].edram_buffer_i.buffer.remove([nlayer, data])
                        self.check_buffer_pe_set.add(self.PE_array[pe_id])

            #if pe.edram_rd_pool_erp:
            erp.add(pe)

        self.erp_pool_rd = erp

    def process_interconnect(self):
        for s in range(self.interconnect_step):
            self.interconnect.step()
            self.Total_energy_interconnect += self.interconnect.step_energy_consumption
        # Packets arrive: Store data, trigger event
        for pk in self.interconnect.arrived_list:
            if self.trace:
                pass
                print("\tArrived packet:", pk)
            rty, rtx = pk.destination[0], pk.destination[1]
            pey, pex = pk.destination[2], pk.destination[3]
            pe_idx = pex + pey * self.hd_info.PE_num_x + rtx * self.hd_info.PE_num + rty * self.hd_info.PE_num * self.hd_info.Router_num_x
            pe = self.PE_array[pe_idx]

            # Energy
            pe.Edram_buffer_energy += self.hd_info.Energy_edram_buffer * self.input_bit # write

            if len(pk.data[1]) == 3:
                pe.edram_buffer.put(pk.data)
                pe.edram_buffer_i.put(pk.data)
                self.check_buffer_pe_set.add(pe)

            if self.trace:
                pass
                #print("put packet data:", pk)

            for pro_event_idx in pk.pro_event_list:
                pro_event = self.Computation_order[pro_event_idx]
                pro_event.data_is_transfer -= 1

        self.interconnect.packet_in_module_ctr -= len(self.interconnect.arrived_list)
        self.interconnect.arrived_list = []

    def event_transfer(self):
        for event in self.data_transfer_erp:
            self.done_event += 1
            if not self.isPipeLine:
                self.this_layer_event_ctr += 1

            if self.trace:
                pass
                print("\tdo data_transfer, layer:", event.nlayer, ",order index:", self.Computation_order.index(event), \
                       "pos:", event.position_idx, "data:", event.outputs)
            
            # src, des
            src = event.position_idx[0]
            src_pe_id = src[3] + src[2] * self.hd_info.PE_num_x + \
                        src[1] * self.hd_info.PE_num + \
                        src[0] * self.hd_info.PE_num * self.hd_info.Router_num_x
            src_pe = self.PE_array[src_pe_id]
            des = event.position_idx[1]
            des_pe_id = des[3] + des[2] * self.hd_info.PE_num_x + \
                        des[1] * self.hd_info.PE_num + \
                        des[0] * self.hd_info.PE_num * self.hd_info.Router_num_x
            des_pe = self.PE_array[des_pe_id]

            # data
            pro_event_list = event.proceeding_event
            event_type = self.Computation_order[pro_event_list[0]].event_type
            if event_type == "edram_rd_ir" or event_type == "edram_rd_pool":
                data = [event.nlayer+1, event.outputs[0]]
            elif event_type == "pe_saa":
                data = [event.nlayer, event.outputs[0]]

            packet = Packet(src, des, data, pro_event_list)
            self.interconnect.input_packet(packet)

            # Energy
            src_pe.Edram_buffer_energy += self.hd_info.Energy_edram_buffer * self.input_bit # read

            ### add next event counter
            for proceeding_index in pro_event_list:
                pro_event = self.Computation_order[proceeding_index]
                pro_event.current_number_of_preceding_event += 1
                pro_event.data_is_transfer += 1
                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                    if pro_event.event_type == "edram_rd_ir":
                        cuy, cux = pro_event.position_idx[4], pro_event.position_idx[5]
                        cu_idx = cux + cuy * self.hd_info.CU_num_x
                        des_pe.edram_rd_ir_trigger.append([pro_event, [cu_idx]])
                        #if des_pe not in self.trigger_edram_rd:
                        self.trigger_edram_rd.add(des_pe)
                    elif pro_event.event_type == "edram_rd_pool":
                        des_pe.edram_rd_pool_trigger.append([pro_event, []])
                        #if des_pe not in self.trigger_pool_rd:
                        self.trigger_pool_rd.add(des_pe)
                    elif pro_event.event_type == "pe_saa":
                        des_pe.pe_saa_trigger.append([pro_event, []])
                        #if des_pe not in self.trigger_pe_saa:
                        self.trigger_pe_saa.add(des_pe)

            # Free buffer (ideal)
            if event_type == "pe_saa": # same layer data transfer
                if data in self.PE_array[src_pe_id].edram_buffer_i.buffer:
                    self.PE_array[src_pe_id].edram_buffer_i.buffer.remove(data)
                    self.check_buffer_pe_set.add(self.PE_array[src_pe_id])
            else:
                if self.ordergenerator.model_info.layer_list[event.nlayer+1].layer_type != "fully":
                    d = event.outputs[0]
                    pos = d[1] + d[0]*self.ordergenerator.model_info.input_w[event.nlayer+1] + d[2]*self.ordergenerator.model_info.input_w[event.nlayer+1]*self.ordergenerator.model_info.input_h[event.nlayer+1] # w + h*width + c*height*width
                    self.ordergenerator.free_buffer_controller.input_require[src_pe_id][event.nlayer+1][pos] -= 1
                    if self.ordergenerator.free_buffer_controller.input_require[src_pe_id][event.nlayer+1][pos] == 0:
                        self.PE_array[src_pe_id].edram_buffer_i.buffer.remove([event.nlayer+1, d])
                        self.check_buffer_pe_set.add(self.PE_array[src_pe_id])
                else:
                    d = event.outputs[0]
                    pos = d[0]
                    self.ordergenerator.free_buffer_controller.input_require[src_pe_id][event.nlayer+1][pos] -= 1
                    if self.ordergenerator.free_buffer_controller.input_require[src_pe_id][event.nlayer+1][pos] == 0:
                        self.PE_array[src_pe_id].edram_buffer_i.buffer.remove([event.nlayer+1, d])
                        self.check_buffer_pe_set.add(self.PE_array[src_pe_id])

        self.data_transfer_erp = []

    def fetch(self):
        pass
        # des_dict = dict()
        # for FE_d in self.fetch_array.copy():
        #     FE, fetch_data = FE_d[0], FE_d[1]
        #     FE.cycles_counter += 1
        #     if FE.cycles_counter == FE.fetch_cycle:
        #         src  = (0, FE.event.position_idx[1], -1, -1)
        #         des  = FE.event.position_idx[0:4]
        #         # if not self.isPipeLine:
        #         #     self.this_layer_event_ctr -= len(fetch_data)
        #         if des not in des_dict:
        #             des_dict[des] = []
        #         for data in fetch_data:
        #             #data = inp[1:]
        #             data = [FE.event.nlayer, data]
        #             pro_event_idx = self.Computation_order.index(FE.event)

        #             isDataInDict = False
        #             for data_pro_event in des_dict[des]:
        #                 if data == data_pro_event[0]:
        #                     data_pro_event[1].append(pro_event_idx)
        #                     isDataInDict = True
        #                     break
        #             if not isDataInDict:
        #                 des_dict[des].append([data, [pro_event_idx]])
        #         self.fetch_array.remove(FE_d)
        # for des in des_dict:
        #     src = (0, des[1], -1, -1)
        #     for data_pro_event in des_dict[des]:
        #         data, pro_event_idx = data_pro_event[0], data_pro_event[1]
        #         packet = Packet(src, des, data, pro_event_idx)
        #         self.interconnect.input_packet(packet)

    def trigger(self):
        ## Trigger interconnect
        for trigger in self.data_transfer_trigger:
            pro_event = trigger[0]
            self.data_transfer_erp.append(pro_event)
        self.data_transfer_trigger = []

        # Trigger edram_rd_ir
        if not self.isPipeLine:
            if self.trigger_next_layer:
                if self.pipeline_layer_stage < self.ordergenerator.model_info.layer_length:
                    if self.ordergenerator.model_info.layer_list[self.pipeline_layer_stage].layer_type != "pooling":
                        for pe in self.trigger_edram_rd:
                            for trigger in pe.edram_rd_ir_trigger:
                                pro_event = trigger[0]
                                cu_idx    = trigger[1][0]
                                pe.CU_array[cu_idx].edram_rd_ir_erp.append(pro_event)
                                if cu_idx != pe.edram_rd_cu_idx and not pe.CU_array[cu_idx].state:
                                    if cu_idx not in pe.idle_eventQueuing_CU:
                                        pe.idle_eventQueuing_CU.append(cu_idx)
                            pe.edram_rd_ir_trigger = []
                            #if pe not in self.erp_rd and pe.idle_eventQueuing_CU:
                            if pe.idle_eventQueuing_CU:
                                self.erp_rd.add(pe)
                        self.trigger_edram_rd = set()
                        self.trigger_next_layer = False
        else: # pipeline
            for pe in self.trigger_edram_rd:
                for trigger in pe.edram_rd_ir_trigger:
                    pro_event = trigger[0]
                    cu_idx    = trigger[1][0]
                    pe.CU_array[cu_idx].edram_rd_ir_erp.append(pro_event)
                    if cu_idx != pe.edram_rd_cu_idx and not pe.CU_array[cu_idx].state:
                        if cu_idx not in pe.idle_eventQueuing_CU:
                            pe.idle_eventQueuing_CU.append(cu_idx)
                pe.edram_rd_ir_trigger = []
                #if pe not in self.erp_rd and pe.idle_eventQueuing_CU:
                if pe.idle_eventQueuing_CU:
                    self.erp_rd.add(pe)
            self.trigger_edram_rd = set()

        for pe in self.trigger_cu_op:
            event = pe.cu_op_trigger
            cuy, cux = event.position_idx[4], event.position_idx[5]
            cu_idx = cux + cuy * self.hd_info.CU_num_x
            cu = pe.CU_array[cu_idx]
            cu.cu_op_event = event
            pe.cu_op_list.append(cu)
            pe.cu_op_trigger = 0
            #if pe not in self.erp_cu_op:
            self.erp_cu_op.add(pe)
        self.trigger_cu_op = set()

        for pe in self.trigger_pe_saa:
            for trigger in pe.pe_saa_trigger:
                pro_event = trigger[0]
                pe.pe_saa_erp.append(pro_event) 
            pe.pe_saa_trigger = []
            #if pe not in self.erp_pe_saa:
            self.erp_pe_saa.add(pe)
        self.trigger_pe_saa = set()

        for pe in self.trigger_act:
            for trigger in pe.activation_trigger:
                pro_event = trigger[0]
                pe.activation_erp.append(pro_event)
            pe.activation_trigger = []
            #if pe not in self.erp_act:
            self.erp_act.add(pe)
        self.trigger_act = set()

        for pe in self.trigger_edram_wr:
            for trigger in pe.edram_wr_trigger:
                pro_event = trigger[0]
                pe.edram_wr_erp.append(pro_event)
            pe.edram_wr_trigger = []
            #if pe not in self.erp_wr:
            self.erp_wr.add(pe)
        self.trigger_edram_wr = set()

        # Trigger edram_read_pool
        if not self.isPipeLine:
            if self.trigger_next_layer:
                if self.pipeline_layer_stage < self.ordergenerator.model_info.layer_length:
                    if self.ordergenerator.model_info.layer_list[self.pipeline_layer_stage].layer_type == "pooling":
                        # 這邊只有要trigger conv 或 fully
                        for pe in self.trigger_pool_rd:
                            for trigger in pe.edram_rd_pool_trigger:
                                pro_event = trigger[0]
                                pe.edram_rd_pool_erp.append(pro_event)
                            pe.edram_rd_pool_trigger = []
                            #if pe not in self.erp_pool_rd:
                            self.erp_pool_rd.add(pe)
                        self.trigger_pool_rd = set()
                        self.trigger_next_layer = False
        else: # pipeline
            for pe in self.trigger_pool_rd:
                for trigger in pe.edram_rd_pool_trigger:
                    pro_event = trigger[0]
                    pe.edram_rd_pool_erp.append(pro_event)
                pe.edram_rd_pool_trigger = []
                #if pe not in self.erp_pool_rd:
                self.erp_pool_rd.add(pe)
            self.trigger_pool_rd = set()

    def print_statistics_result(self):
        self.color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.Edram_buffer_energy     = 0.
        self.Bus_energy              = 0.
        self.PE_shift_and_add_energy = 0.
        self.Or_energy               = 0.
        self.Activation_energy       = 0.
        self.Pooling_energy          = 0.

        self.CU_shift_and_add_energy = 0.
        self.CU_dac_energy           = 0.
        self.CU_adc_energy           = 0.
        self.CU_crossbar_energy      = 0.
        self.CU_IR_energy            = 0.
        self.CU_OR_energy            = 0.

        for pe in self.PE_array:
            self.Edram_buffer_energy     += pe.Edram_buffer_energy
            self.Bus_energy              += pe.Bus_energy
            self.PE_shift_and_add_energy += pe.PE_shift_and_add_energy
            self.Or_energy               += pe.Or_energy
            self.Activation_energy       += pe.Activation_energy
            self.Pooling_energy          += pe.Pooling_energy
            self.CU_shift_and_add_energy += pe.CU_shift_and_add_energy
            self.CU_dac_energy           += pe.CU_dac_energy
            self.CU_adc_energy           += pe.CU_adc_energy
            self.CU_crossbar_energy      += pe.CU_crossbar_energy
            self.CU_IR_energy            += pe.CU_IR_energy
            self.CU_OR_energy            += pe.CU_OR_energy
        self.Total_energy = self.Edram_buffer_energy + self.Bus_energy + self.PE_shift_and_add_energy + \
                            self.Or_energy + self.Activation_energy + self.Pooling_energy + \
                            self.CU_shift_and_add_energy + self.CU_dac_energy + self.CU_adc_energy + \
                            self.CU_crossbar_energy + self.CU_IR_energy + self.CU_OR_energy + \
                            self.Total_energy_interconnect

        print("Total Cycles:", self.cycle_ctr)
        if not self.isPipeLine:
            print("Cycles each layer:", self.cycles_each_layer)
        print("Cycles time:", self.hd_info.cycle_time, "ns\n")
        print()

        print("Energy breakdown:")
        self.PE_energy_breakdown()
        print("output buffer utilization...")
        self.buffer_analysis()
        print("output pe utilization...")
        self.pe_utilization()
        # print("output cu utilization...")
        # self.cu_utilization()
        print("output performance anaylsis...")
        self.performance_statistics()
        self.output_result()

    def output_result(self):
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Result.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Cycles", self.cycle_ctr])
            writer.writerow(["Total energy consumption(nJ)", self.Total_energy])
            writer.writerow([])
            writer.writerow(["", "Energy(nJ)"])
            writer.writerow(["Interconnect", self.Total_energy_interconnect])

            writer.writerow(["Edram Buffer", self.Edram_buffer_energy])
            writer.writerow(["Bus", self.Bus_energy])
            writer.writerow(["PE Shift and Add", self.PE_shift_and_add_energy])
            writer.writerow(["OR", self.Or_energy])
            writer.writerow(["Activation", self.Activation_energy])
            writer.writerow(["Pooling", self.Pooling_energy])

            writer.writerow(["CU Shift and Add", self.CU_shift_and_add_energy])
            writer.writerow(["DAC", self.CU_dac_energy])
            writer.writerow(["ADC", self.CU_adc_energy])
            writer.writerow(["Crossbar Array", self.CU_crossbar_energy])
            writer.writerow(["IR", self.CU_IR_energy])
            writer.writerow(["OR", self.CU_OR_energy])

    def PE_energy_breakdown(self):
        # PE breakdown
        print("PE energy breakdown")
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Energy_breakdown_PE.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "Buffer", "Bus", "PE Shift and add", "OR", "Activation", "Pooling",
                             "CU Shift and add", "DAC", "ADC", "Crossbar", "IR", "OR"
                            ])
            for pe in self.PE_array:
                arr = ["PE"+str(self.PE_array.index(pe))]
                arr.append(pe.Edram_buffer_energy)
                arr.append(pe.Bus_energy)
                arr.append(pe.PE_shift_and_add_energy)
                arr.append(pe.Or_energy)
                arr.append(pe.Activation_energy)
                arr.append(pe.Pooling_energy)

                arr.append(pe.CU_shift_and_add_energy)
                arr.append(pe.CU_dac_energy)
                arr.append(pe.CU_adc_energy)
                arr.append(pe.CU_crossbar_energy)
                arr.append(pe.CU_IR_energy)
                arr.append(pe.CU_OR_energy)
                writer.writerow(arr)

    def buffer_analysis(self):
        # num轉KB
        for pe in self.PE_array:
            for i in range(len(pe.edram_buffer_i.buffer_size_util[1])):
                pe.edram_buffer_i.buffer_size_util[1][i] *= self.input_bit/8/1000
            for i in range(len(pe.edram_buffer.buffer_size_util[1])):
                pe.edram_buffer.buffer_size_util[1][i] *= self.input_bit/8/1000
            pe.edram_buffer_i.maximal_usage *= self.input_bit/8/1000
            pe.edram_buffer.maximal_usage *= self.input_bit/8/1000

        ### time history
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Buffer_time_history_ideal.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for pe in self.PE_array:
                util = pe.edram_buffer_i.buffer_size_util
                if len(util[0]) == 0:
                    continue
                writer.writerow(["PE"+str(self.PE_array.index(pe))])
                writer.writerow(util[0])
                writer.writerow(util[1])
        
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Buffer_time_history_nonideal.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for pe in self.PE_array:
                util = pe.edram_buffer.buffer_size_util
                if len(util[0]) == 0:
                    continue
                writer.writerow(["PE"+str(self.PE_array.index(pe))])
                writer.writerow(util[0])
                writer.writerow(util[1])

        ### utilization
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Buffer_utilization_ideal.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "Size(KB)"])
            for pe in self.PE_array:
                if pe.edram_buffer_i.maximal_usage == 0:
                    continue
                pe_id = self.PE_array.index(pe)
                writer.writerow(["PE"+str(pe_id), pe.edram_buffer_i.maximal_usage])
        
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Buffer_utilization_nonideal.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "Size(KB)"])
            for pe in self.PE_array:
                if pe.edram_buffer.maximal_usage == 0:
                    continue
                pe_id = self.PE_array.index(pe)
                writer.writerow(["PE"+str(pe_id), pe.edram_buffer.maximal_usage])

    def pe_utilization(self):
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/PE_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(len(self.pe_state_for_plot[0])):
                writer.writerow([self.pe_state_for_plot[0][row], self.pe_state_for_plot[1][row]])

    def cu_utilization(self):
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/CU_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(len(self.cu_state_for_plot[0])):
                writer.writerow([self.cu_state_for_plot[0][row], self.cu_state_for_plot[1][row]])

    def performance_statistics(self):
        # PE breakdown
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Performance_CU.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "Pure computation", "Pure idle", "Wait resource", "Wait transfer"])
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    arr = ["PE"+str(self.PE_array.index(pe))+", CU"+str(pe.CU_array.index(cu))]
                    arr.append(cu.pure_computation_time)
                    pure_idle_time = self.cycle_ctr - cu.pure_computation_time - cu.wait_resource_time - cu.wait_transfer_time
                    arr.append(pure_idle_time)
                    arr.append(cu.wait_resource_time)
                    arr.append(cu.wait_transfer_time)
                    writer.writerow(arr)
