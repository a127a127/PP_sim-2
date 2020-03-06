from HardwareMetaData import HardwareMetaData as HW
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
    def __init__(self, ordergenerator, trace, mapping_str, scheduling_str, replacement):
        self.ordergenerator = ordergenerator
        self.mp_info = ordergenerator.mp_info
        self.Computation_order = self.ordergenerator.Computation_order
        print("Computation order length:", len(self.Computation_order))
        self.input_bit = self.ordergenerator.model_info.input_bit

        self.trace = trace
        self.mapping_str = mapping_str
        self.scheduling_str = scheduling_str
        if self.scheduling_str == "Pipeline":
            self.isPipeLine = True
        elif self.scheduling_str == "Non_pipeline":
            self.isPipeLine = False
        if replacement == "Ideal":
            self.ideal_replacement = True
        elif replacement == "LRU":
            self.ideal_replacement = False

        self.edram_read_data  = floor(HW().eDRAM_read_bits / self.input_bit) # 1個cycle可以讀多少data
        self.edram_write_data = floor(HW().eDRAM_write_bits / self.input_bit) # 1個cycle可以寫多少data

        self.cycle_ctr = 0
        self.done_event = 0

        self.PE_array = []
        for rty_idx in range(HW().Router_num_y):
            for rtx_idx in range(HW().Router_num_x):
                for pey_idx in range(HW().PE_num_y):
                    for pex_idx in range(HW().PE_num_x):
                        pe_pos = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        pe = PE(pe_pos, ModelConfig().input_bit)
                        self.PE_array.append(pe)

        self.fetch_array = []
        # Interconnect
        self.Total_energy_interconnect = 0
        self.interconnect = Interconnect(HW().Router_num_y, HW().Router_num_x, self.input_bit)
        self.interconnect_step = HW().Router_flit_size / self.input_bit * HW().cycle_time * HW().Frequency # scaling from ISAAC
        self.interconnect_step = floor(self.interconnect_step)
        
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

            self.trigger_next_layer = False

        # Event pool
        self.data_transfer_erp = []
        self.data_transfer_trigger = []

        # 要檢查Event pool的PE index
        self.pe_id_rd      = set()
        self.pe_id_cu_op   = set()
        self.pe_id_pe_saa  = set()
        self.pe_id_act     = set()
        self.pe_id_wr      = set()
        self.pe_id_pool_rd = set()

        # Trigger pe_idx
        self.trigger_edram_rd = set()
        self.trigger_cu_op    = set()
        self.trigger_pe_saa   = set()
        self.trigger_edram_wr = set()
        self.trigger_act      = set()
        self.trigger_pool_rd  = set()

        self.busy_pe = set()

        # Utilization
        self.pe_state_for_plot = [[], []]
        self.cu_state_for_plot = [[], []]
        self.xb_state_for_plot = [[], []]
        self.check_buffer_pe_set = set()

    def run(self):
        # 把input feature map放到buffer中
        for pe in self.PE_array:
            rty_idx, rtx_idx = pe.position[0], pe.position[1]
            pey_idx, pex_idx = pe.position[2], pe.position[3]
            feature_m = np.zeros((ModelConfig().input_h, ModelConfig().input_w, ModelConfig().input_c))
            for cuy_idx in range(HW().CU_num_y):
                for cux_idx in range(HW().CU_num_x):
                    for xby_idx in range(HW().Xbar_num_y):
                        for xbx_idx in range(HW().Xbar_num_x):
                            mapping = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][0] # layer0
                            for mp in mapping:
                                for inp in mp.inputs:
                                    feature_m[inp[1]][inp[2]][inp[3]] = 1

            for h in range(feature_m.shape[0]):
                for w in range(feature_m.shape[1]):
                    for c in range(feature_m.shape[2]):
                        if feature_m[h][w][c] == 1:
                            data = [0, h, w, c]
                            pe.edram_buffer.buffer.append(data)

        for e in self.Computation_order:
            if e.event_type == 'edram_rd_ir':
                if e.preceding_event_count == 0:
                    pos = e.position_idx
                    rty, rtx, pey, pex, cuy, cux = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
                    pe_idx = pex + pey * HW().PE_num_x + rtx * HW().PE_num + rty * HW().Router_num_x * HW().PE_num
                    cu_idx = cux + cuy * HW().CU_num_x
                    pe = self.PE_array[pe_idx]
                    cu = pe.CU_array[cu_idx]
                    cu.edram_rd_ir_erp.append(e)

                    if cu not in pe.idle_eventQueuing_CU:
                        pe.idle_eventQueuing_CU.append(cu)
                    
                    self.pe_id_rd.add(pe)

                    # for performance analysis
                    if cu_idx not in pe.eventQueuing_CU: 
                        pe.eventQueuing_CU.append(cu_idx)
            if e.nlayer != 0:
                break
        
        t_edram = 0
        t_cuop = 0
        t_pesaa = 0
        t_act = 0
        t_wr = 0
        t_it = 0
        t_trans = 0
        t_fe = 0
        t_tr = 0
        t_st = 0
        t_buf = 0
        t_poo = 0
        t_ = time.time()
        start_time = time.time()
        layer = 0
        while True:
            if self.cycle_ctr % 100000 == 0:
                if self.done_event == 0:
                    pass
                else:
                    print("完成比例:", self.done_event/len(self.Computation_order))
                    print("layer:", layer)
                    print("Cycle",self.cycle_ctr, "Done event:", self.done_event, "time per event", (time.time()-start_time)/self.done_event, "time per cycle", (time.time()-start_time)/self.cycle_ctr)
                    print("edram:", t_edram, "t_cuop", t_cuop, "pesaa", t_pesaa, "act", t_act, "wr", t_wr)
                    print("iterconeect", t_it, "fetch", t_fe, "transfer", t_trans, "trigger", t_tr, "state", t_st)
                    print("buffer", t_buf, "pool", t_poo)
                    print("t:", time.time()-t_)
                    t_edram, t_cuop, t_pesaa, t_act, t_wr = 0, 0, 0, 0, 0
                    t_it, t_fe, t_tr, t_st, t_buf, t_poo = 0, 0, 0, 0, 0, 0
                    t_trans = 0
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
            t_trans += time.time() - staa

            staa = time.time()
            #self.fetch()
            t_fe += time.time() - staa

            ### Pipeline stage control ###
            if not self.isPipeLine:
                if self.this_layer_event_ctr == self.events_each_layer[self.pipeline_layer_stage]:
                    layer += 1
                    self.pipeline_layer_stage += 1
                    self.cycles_each_layer.append(self.this_layer_cycle_ctr)
                    self.this_layer_cycle_ctr = 0
                    self.this_layer_event_ctr = 0
                    self.trigger_next_layer = True

            staa = time.time()
            self.trigger()
            t_tr += time.time() - staa

            ### Record PE State ###
            staa = time.time()
            for pe in self.busy_pe:
                self.pe_state_for_plot[0].append(self.cycle_ctr)
                self.pe_state_for_plot[1].append(self.PE_array.index(pe))
            self.busy_pe = set()
            t_st += time.time() - staa


            ### Buffer utilization
            staa = time.time()
            if self.cycle_ctr % 200 == 0:
                for pe in self.check_buffer_pe_set:
                    pe.edram_buffer.buffer_size_util[0].append(self.cycle_ctr)
                    pe.edram_buffer.buffer_size_util[1].append(len(pe.edram_buffer.buffer))
                self.check_buffer_pe_set = set()
            t_buf += time.time() - staa

            ### Finish
            if self.done_event == len(self.Computation_order):
                break

    def event_edram_rd(self):
        erp = set()
        for pe in self.pe_id_rd:
            if not pe.edram_rd_event:
                # 從CU event pool拿一個event來處理
                cu = pe.idle_eventQueuing_CU.popleft()
                pe.edram_rd_cu_idx = cu
                event = cu.edram_rd_ir_erp.popleft()
                pe.edram_rd_event = event
            else:
                cu = pe.edram_rd_cu_idx
                event = pe.edram_rd_event
        
            num_data = len(event.inputs)
            if not pe.data_to_ir_ing: # 還沒開始從CU傳資料到IR
                if event.data_is_transfer != 0: # 資料是否還在transfer？
                    pass
                else: # do edram read
                    if self.trace:
                        print("\tdo edram_rd_ir, nlayer:", event.nlayer,", pos:", event.position_idx)
                    pe.data_to_ir_ing = True
                    self.busy_pe.add(pe)

                    # Energy
                    pe.Edram_buffer_energy += HW().Energy_edram_buffer * self.input_bit * num_data
                    pe.Bus_energy += HW().Energy_bus * self.input_bit * num_data
                    pe.CU_IR_energy += HW().Energy_ir_in_cu * self.input_bit * num_data # write

                    # 要幾個cycle讀完
                    pe.edram_read_cycles = ceil(num_data / self.edram_read_data)

                    # 判斷是否完成(只有read一個cycle內完成才會進入這邊)
                    pe.edram_rd_cycle_ctr += 1
                    if pe.edram_rd_cycle_ctr == pe.edram_read_cycles: # 完成edram read
                        if self.trace:
                            print("\t\tfinish edram read")
                        self.done_event += 1
                        if not self.isPipeLine:
                            self.this_layer_event_ctr += 1

                        pe.edram_rd_cycle_ctr = 0
                        pe.edram_rd_event  = None
                        pe.edram_rd_cu_idx = None
                        pe.data_to_ir_ing = False
                        cu.state = True

                        # trigger cu operation
                        proceeding_index = event.proceeding_event[0] # 只會trigger一個cu operation
                        pro_event = self.Computation_order[proceeding_index]
                        pro_event.current_number_of_preceding_event += 1
                        pe.cu_op_trigger = pro_event
                        self.trigger_cu_op.add(pe)

                        if self.ideal_replacement:
                            # Free buffer (ideal)
                            pe_id = self.PE_array.index(pe)
                            nlayer = event.nlayer
                            if self.ordergenerator.model_info.layer_list[nlayer].layer_type == "convolution":
                                for d in event.inputs:
                                    pos = d[2] + d[1]*self.ordergenerator.model_info.input_w[nlayer] + d[3]*self.ordergenerator.model_info.input_w[nlayer]*self.ordergenerator.model_info.input_h[nlayer] # w + h*width + c*height*width
                                    self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                                    if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                        self.PE_array[pe_id].edram_buffer.buffer.remove(d)
                                        self.check_buffer_pe_set.add(self.PE_array[pe_id])
                            elif self.ordergenerator.model_info.layer_list[nlayer].layer_type == "fully":
                                for d in event.inputs:
                                    pos = d[1]
                                    self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                                    if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                        self.PE_array[pe_id].edram_buffer.buffer.remove(d)
                                        self.check_buffer_pe_set.add(self.PE_array[pe_id])
            
            else: # CU傳資料到IR
                pe.edram_rd_cycle_ctr += 1
                self.busy_pe.add(pe)
                if pe.edram_rd_cycle_ctr == pe.edram_read_cycles: # 完成edram read
                    if self.trace:
                        print("\tfinish edram_rd_ir, nlayer:", event.nlayer,", pos:", event.position_idx)
                    self.done_event += 1
                    if not self.isPipeLine:
                        self.this_layer_event_ctr += 1

                    pe.edram_rd_cycle_ctr = 0
                    pe.edram_rd_event  = None
                    pe.edram_rd_cu_idx = None
                    pe.data_to_ir_ing = False
                    cu.state = True

                    # trigger cu operation
                    proceeding_index = event.proceeding_event[0] # 只會trigger一個cu operation
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1
                    pe.cu_op_trigger = pro_event
                    self.trigger_cu_op.add(pe)

                    if self.ideal_replacement:
                        # Free buffer (ideal)
                        pe_id = self.PE_array.index(pe)
                        nlayer = event.nlayer
                        if self.ordergenerator.model_info.layer_list[nlayer].layer_type == "convolution":
                            for d in event.inputs:
                                pos = d[2] + d[1]*self.ordergenerator.model_info.input_w[nlayer] + d[3]*self.ordergenerator.model_info.input_w[nlayer]*self.ordergenerator.model_info.input_h[nlayer] # w + h*width + c*height*width
                                self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                                if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                    self.PE_array[pe_id].edram_buffer.buffer.remove(d)
                                    self.check_buffer_pe_set.add(self.PE_array[pe_id])
                        elif self.ordergenerator.model_info.layer_list[nlayer].layer_type == "fully":
                            for d in event.inputs:
                                pos = d[1]
                                self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                                if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                    self.PE_array[pe_id].edram_buffer.buffer.remove(d)
                                    self.check_buffer_pe_set.add(self.PE_array[pe_id])
            
            # 下個cycle還要不要檢查此pe
            if pe.edram_rd_event: # 有event正在處理
                erp.add(pe)
            elif pe.idle_eventQueuing_CU:
                erp.add(pe)
        self.pe_id_rd = erp

    def event_cu_op(self):
        erp = set()
        for pe in self.pe_id_cu_op:
            self.busy_pe.add(pe)
            cu_op_list = []
            for cu in pe.cu_op_list:
                if cu.finish_cycle == 0: # cu operation start
                    if self.trace:
                        pass
                        print("\tcu operation start", "pos:", cu.cu_op_event.position_idx)
                    cu.finish_cycle = self.cycle_ctr - 1 + cu.cu_op_event.inputs + 2 # +2: pipeline 最後兩個 stage

                    ## Energy
                    ou_num_dict = cu.cu_op_event.outputs
                    for xb_idx in ou_num_dict:
                        ou_num = ou_num_dict[xb_idx]
                        pe.CU_dac_energy += HW().Energy_ou_dac * ou_num
                        pe.CU_crossbar_energy += HW().Energy_ou_crossbar * ou_num
                        pe.CU_adc_energy += HW().Energy_ou_adc * ou_num
                        pe.CU_shift_and_add_energy += HW().Energy_ou_ssa * ou_num
                        
                        pe.CU_IR_energy += HW().Energy_ir_in_cu * ou_num * HW().OU_h 
                        pe.CU_OR_energy += HW().Energy_or_in_cu * ou_num * HW().OU_w * HW().ADC_resolution

                    ## xb state ##
                    # for xb_idx in ou_num_dict:
                    #     xb_id = cu_id * HW().Xbar_num + xb_idx
                    #     ou_num = ou_num_dict[xb_idx]
                    #     for c in range(ou_num):
                    #         self.xb_state_for_plot[0].append(self.cycle_ctr+c)
                    #         self.xb_state_for_plot[1].append(xb_id)
                    cu_op_list.append(cu)

                elif cu.finish_cycle != 0: ## doing cu operation
                    if cu.finish_cycle == self.cycle_ctr: # finish
                        if self.trace:
                            pass
                            print("\tcu operation finish", "pos:", cu.cu_op_event.position_idx)
                        self.done_event += 1
                        if not self.isPipeLine:
                            self.this_layer_event_ctr += 1
                        
                        # 如果還有edram read event則
                        if cu.edram_rd_ir_erp:
                            pe.idle_eventQueuing_CU.append(cu)
                            self.pe_id_rd.add(pe)

                        ### add next event counter: pe_saa
                        for proceeding_index in cu.cu_op_event.proceeding_event:
                            pro_event = self.Computation_order[proceeding_index]
                            pro_event.current_number_of_preceding_event += 1
                            if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                if self.trace:
                                    pass
                                    #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                pe.pe_saa_trigger.append([pro_event, []])
                                self.trigger_pe_saa.add(pe)

                        cu.finish_cycle = 0
                        cu.cu_op_event = 0
                        cu.state = False
                    else:
                        cu_op_list.append(cu)
                        if self.trace:
                            print("\tcu operation")
                
                ## cu state ##
                # cu_id = self.PE_array.index(pe) * HW().CU_num + pe.CU_array.index(cu)
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
        self.pe_id_cu_op = erp

    def performance_analysis(self):
        for pe in self.pe_id_rd:
            for cu in pe.idle_eventQueuing_CU:
                cu.wait_transfer_time += 1 # CU在等資料傳到
            if pe.edram_rd_event:
                event = pe.edram_rd_event
                cu = pe.edram_rd_cu_idx
                cu.wait_transfer_time += 1

    def event_pe_saa(self):
        erp = set()
        for pe in self.pe_id_pe_saa:
            pe_saa_erp = []
            self.busy_pe.add(pe)
            for event in pe.pe_saa_erp:
                if event.data_is_transfer != 0: # 此event的資料正在傳輸
                    pe_saa_erp.append(event)
                    continue
                if self.trace:
                    pass
                    print("\tdo pe_saa, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event)) 

                self.done_event += 1
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1
                
                saa_amount = event.inputs

                # Energy
                pe.Or_energy += HW().Energy_or * self.input_bit * saa_amount
                pe.Bus_energy += HW().Energy_bus * self.input_bit * saa_amount
                pe.PE_shift_and_add_energy += HW().Energy_shift_and_add * saa_amount
                pe.Bus_energy += HW().Energy_bus * self.input_bit * saa_amount
                pe.Or_energy += HW().Energy_or * self.input_bit * saa_amount

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
                            self.trigger_act.add(pe)

                        elif pro_event.event_type == "edram_wr":
                            pe.edram_wr_trigger.append([pro_event, []])
                            self.trigger_edram_wr.add(pe)

                if self.ideal_replacement:
                    # Free buffer (ideal)
                    data = event.outputs
                    if data:
                        try:
                            pe.edram_buffer.buffer.remove(data)
                        except ValueError:
                            pass
                        self.check_buffer_pe_set.add(pe)

            pe.pe_saa_erp = pe_saa_erp
            if pe.pe_saa_erp:
                erp.add(pe)
        self.pe_id_pe_saa = erp
 
    def event_act(self):
        erp = set()
        for pe in self.pe_id_act:
            self.busy_pe.add(pe)
            # if pe.activation_epc <= len(pe.activation_erp):
            #     do_act_num = pe.activation_epc
            # else:
            #     do_act_num = len(pe.activation_erp)
            do_act_num = len(pe.activation_erp) # 一個cycle全部做完
            for idx in range(do_act_num):
                event = pe.activation_erp.popleft()
                if self.trace:
                    pass
                    print("\tdo activation, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                self.done_event += 1
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1

                # Energy
                pe.Or_energy += HW().Energy_or * self.input_bit
                pe.Bus_energy += HW().Energy_bus * self.input_bit
                pe.Activation_energy += HW().Energy_activation

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

            # if pe.activation_erp:
            #     erp.add(pe)
        self.pe_id_act = erp

    def event_edram_wr(self):
        erp = set()
        for pe in self.pe_id_wr:
            self.busy_pe.add(pe)
            self.check_buffer_pe_set.add(pe)

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
                pe.Bus_energy += HW().Energy_bus * self.input_bit
                pe.Edram_buffer_energy += HW().Energy_edram_buffer * self.input_bit

                ### Write data into buffer
                pe.edram_buffer.put(event.outputs)
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
                            cu_idx = cu_x + cu_y * HW().CU_num_x
                            pe.edram_rd_ir_trigger.append([pro_event, [cu_idx]])
                            self.trigger_edram_rd.add(pe)
                        elif pro_event.event_type == "edram_rd_pool":
                            pe.edram_rd_pool_trigger.append([pro_event, []])
                            self.trigger_pool_rd.add(pe)
                        elif pro_event.event_type == "data_transfer":
                            self.data_transfer_trigger.append([pro_event, []])

            if pe.edram_wr_erp:
                erp.add(pe)
        self.pe_id_wr = erp

    def event_edram_rd_pool(self):
        erp = set()
        for pe in self.pe_id_pool_rd:
            if pe.data_to_ir_ing:
                erp.add(pe)
                continue
            self.busy_pe.add(pe)
            edram_rd_pool_erp = []
            for event in pe.edram_rd_pool_erp:
                if event.data_is_transfer != 0: # 此event的資料正在傳輸
                    edram_rd_pool_erp.append(event)
                    continue

                self.busy_pe.add(pe)
                
                num_data = len(event.inputs)
                self.done_event += 1
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1
                if self.trace:
                    print("\tdo edram_rd_pool, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                
                # Energy
                pe.Edram_buffer_energy += HW().Energy_edram_buffer * self.input_bit * num_data
                pe.Bus_energy += HW().Energy_bus * self.input_bit * num_data
                pe.Pooling_energy += HW().Energy_pooling

                ### add next event counter: write
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
                
                if self.ideal_replacement:
                    # Free buffer (ideal)
                    pe_id = self.PE_array.index(pe)
                    nlayer = event.nlayer
                    for d in event.inputs:
                        pos = d[2] + d[1]*self.ordergenerator.model_info.input_w[nlayer] + d[3]*self.ordergenerator.model_info.input_w[nlayer]*self.ordergenerator.model_info.input_h[nlayer] # w + h*width + c*height*width
                        self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                        if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                            self.PE_array[pe_id].edram_buffer.buffer.remove(d)
                            self.check_buffer_pe_set.add(self.PE_array[pe_id])

            pe.edram_rd_pool_erp = edram_rd_pool_erp
            if pe.edram_rd_pool_erp:
                erp.add(pe)

        self.pe_id_pool_rd = erp

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
            pe_idx = pex + pey * HW().PE_num_x + rtx * HW().PE_num + rty * HW().PE_num * HW().Router_num_x
            pe = self.PE_array[pe_idx]

            # Energy
            pe.Edram_buffer_energy += HW().Energy_edram_buffer * self.input_bit # write

            # Buffer
            pe.edram_buffer.put(pk.data)
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
            src_pe_id = src[3] + src[2] * HW().PE_num_x + \
                        src[1] * HW().PE_num + \
                        src[0] * HW().PE_num * HW().Router_num_x
            src_pe = self.PE_array[src_pe_id]
            des = event.position_idx[1]
            des_pe_id = des[3] + des[2] * HW().PE_num_x + \
                        des[1] * HW().PE_num + \
                        des[0] * HW().PE_num * HW().Router_num_x
            des_pe = self.PE_array[des_pe_id]

            # data
            pro_event_list = event.proceeding_event
            event_type = self.Computation_order[pro_event_list[0]].event_type
            data = event.outputs

            packet = Packet(src, des, data, pro_event_list)
            self.interconnect.input_packet(packet)

            # Energy
            src_pe.Edram_buffer_energy += HW().Energy_edram_buffer * self.input_bit # read

            ### add next event counter
            for proceeding_index in pro_event_list:
                pro_event = self.Computation_order[proceeding_index]
                pro_event.current_number_of_preceding_event += 1
                pro_event.data_is_transfer += 1
                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                    if pro_event.event_type == "edram_rd_ir":
                        cuy, cux = pro_event.position_idx[4], pro_event.position_idx[5]
                        cu_idx = cux + cuy * HW().CU_num_x
                        des_pe.edram_rd_ir_trigger.append([pro_event, [cu_idx]])
                        self.trigger_edram_rd.add(des_pe)
                    elif pro_event.event_type == "edram_rd_pool":
                        des_pe.edram_rd_pool_trigger.append([pro_event, []])
                        self.trigger_pool_rd.add(des_pe)
                    elif pro_event.event_type == "pe_saa":
                        des_pe.pe_saa_trigger.append([pro_event, []])
                        self.trigger_pe_saa.add(des_pe)
            
            if self.ideal_replacement:
                # Free buffer (ideal)
                if event_type == "pe_saa":
                    self.PE_array[src_pe_id].edram_buffer.buffer.remove(data)
                    self.check_buffer_pe_set.add(self.PE_array[src_pe_id])
                else:
                    if self.ordergenerator.model_info.layer_list[event.nlayer+1].layer_type != "fully":
                        d = event.outputs
                        pos = d[2] + d[1]*self.ordergenerator.model_info.input_w[event.nlayer+1] + d[3]*self.ordergenerator.model_info.input_w[event.nlayer+1]*self.ordergenerator.model_info.input_h[event.nlayer+1] # w + h*width + c*height*width
                        self.ordergenerator.free_buffer_controller.input_require[src_pe_id][event.nlayer+1][pos] -= 1
                        if self.ordergenerator.free_buffer_controller.input_require[src_pe_id][event.nlayer+1][pos] == 0:
                            self.PE_array[src_pe_id].edram_buffer.buffer.remove(d)
                            self.check_buffer_pe_set.add(self.PE_array[src_pe_id])
                    else:
                        d = event.outputs
                        pos = d[1]
                        self.ordergenerator.free_buffer_controller.input_require[src_pe_id][event.nlayer+1][pos] -= 1
                        if self.ordergenerator.free_buffer_controller.input_require[src_pe_id][event.nlayer+1][pos] == 0:
                            self.PE_array[src_pe_id].edram_buffer.buffer.remove(d)
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
        for trigger in self.data_transfer_trigger:
            pro_event = trigger[0]
            self.data_transfer_erp.append(pro_event)
        self.data_transfer_trigger = []

        if not self.isPipeLine:
            if self.trigger_next_layer:
                self.trigger_next_layer = False
                for pe in self.trigger_edram_rd:
                    # print("\tpe", self.PE_array.index(pe), "edram_rd_ir_trigger", self.PE_array.index(pe), pe.edram_rd_ir_trigger)
                    for trigger in pe.edram_rd_ir_trigger:
                        pro_event = trigger[0]
                        cu_idx    = trigger[1][0]
                        cu = pe.CU_array[cu_idx]
                        cu.edram_rd_ir_erp.append(pro_event)
                        if cu not in pe.idle_eventQueuing_CU:
                            pe.idle_eventQueuing_CU.append(cu)
                            self.pe_id_rd.add(pe)
                    pe.edram_rd_ir_trigger = []
                self.trigger_edram_rd = set()
                for pe in self.trigger_pool_rd:
                    # print("\tpe", self.PE_array.index(pe), "trigger_pool_rd", self.PE_array.index(pe), self.trigger_pool_rd)
                    for trigger in pe.edram_rd_pool_trigger:
                        pro_event = trigger[0]
                        pe.edram_rd_pool_erp.append(pro_event)
                    pe.edram_rd_pool_trigger = []
                    self.pe_id_pool_rd.add(pe)
                self.trigger_pool_rd = set()
        else: # pipeline
            for pe in self.trigger_edram_rd:
                for trigger in pe.edram_rd_ir_trigger:
                    pro_event = trigger[0]
                    cu_idx    = trigger[1][0]
                    cu = pe.CU_array[cu_idx]
                    cu.edram_rd_ir_erp.append(pro_event)
                    if cu != pe.edram_rd_cu_idx and not cu.state:
                        if cu not in pe.idle_eventQueuing_CU:
                            pe.idle_eventQueuing_CU.append(cu)
                            self.pe_id_rd.add(pe)
                pe.edram_rd_ir_trigger = []
            self.trigger_edram_rd = set()

            for pe in self.trigger_pool_rd:
                for trigger in pe.edram_rd_pool_trigger:
                    pro_event = trigger[0]
                    pe.edram_rd_pool_erp.append(pro_event)
                pe.edram_rd_pool_trigger = []
                self.pe_id_pool_rd.add(pe)
            self.trigger_pool_rd = set()

        for pe in self.trigger_cu_op:
            event = pe.cu_op_trigger
            cuy, cux = event.position_idx[4], event.position_idx[5]
            cu_idx = cux + cuy * HW().CU_num_x
            cu = pe.CU_array[cu_idx]
            cu.cu_op_event = event
            pe.cu_op_list.append(cu)
            pe.cu_op_trigger = 0
            self.pe_id_cu_op.add(pe)
        self.trigger_cu_op = set()

        for pe in self.trigger_pe_saa:
            # print("\tpe", self.PE_array.index(pe), "trigger_pe_saa", self.PE_array.index(pe), self.trigger_pe_saa)
            for trigger in pe.pe_saa_trigger:
                pro_event = trigger[0]
                pe.pe_saa_erp.append(pro_event) 
            pe.pe_saa_trigger = []
            self.pe_id_pe_saa.add(pe)
        self.trigger_pe_saa = set()

        for pe in self.trigger_act:
            # print("\tpe", self.PE_array.index(pe), "trigger_act", self.PE_array.index(pe), self.trigger_act)
            for trigger in pe.activation_trigger:
                pro_event = trigger[0]
                pe.activation_erp.append(pro_event)
            pe.activation_trigger = []
            self.pe_id_act.add(pe)
        self.trigger_act = set()

        for pe in self.trigger_edram_wr:
            # print("\tpe", self.PE_array.index(pe), "trigger_edram_wr", self.PE_array.index(pe), self.trigger_edram_wr)
            for trigger in pe.edram_wr_trigger:
                pro_event = trigger[0]
                pe.edram_wr_erp.append(pro_event)
            pe.edram_wr_trigger = []
            self.pe_id_wr.add(pe)
        self.trigger_edram_wr = set()

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
        print("Cycles time:", HW().cycle_time, "ns\n")
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
            for i in range(len(pe.edram_buffer.buffer_size_util[1])):
                pe.edram_buffer.buffer_size_util[1][i] *= self.input_bit/8/1000
            pe.edram_buffer.maximal_usage *= self.input_bit/8/1000

        ### time history
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Buffer_time_history.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for pe in self.PE_array:
                util = pe.edram_buffer.buffer_size_util
                if len(util[0]) == 0:
                    continue
                writer.writerow(["PE"+str(self.PE_array.index(pe))])
                writer.writerow(util[0])
                writer.writerow(util[1])
        
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Buffer_utilization.csv', 'w', newline='') as csvfile:
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
