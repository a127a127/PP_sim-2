from HardwareMetaData import HardwareMetaData
from configs.ModelConfig import ModelConfig
from PE import PE

from EventMetaData import EventMetaData
from FetchEvent import FetchEvent
from Interconnect import Interconnect
from Packet import Packet

import numpy as np
from math import ceil, floor
import os, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        self.hd_info = HardwareMetaData()
        self.input_bit = self.ordergenerator.model_info.input_bit
        self.cycle_ctr = 0
        self.edram_read_cycles = \
            ceil(self.hd_info.Xbar_num * self.hd_info.Xbar_h * \
            self.input_bit * self.hd_info.eDRAM_read_latency / \
            self.hd_info.cycle_time)
        # Energy
        self.Total_energy_cycle = 0
        self.Total_energy_edram_buffer = 0
        self.Total_energy_bus = 0
        self.Total_energy_router = 0
        self.Total_energy_activation = 0
        self.Total_energy_pe_shift_and_add = 0
        self.Total_energy_cu_shift_and_add = 0
        self.Total_energy_pooling = 0
        self.Total_energy_or = 0
        self.Total_energy_adc = 0
        self.Total_energy_dac = 0
        self.Total_energy_crossbar = 0
        self.Total_energy_ir_in_cu = 0
        self.Total_energy_or_in_cu = 0
        self.Total_energy_interconnect = 0
        self.PE_array = []
        for rty_idx in range(self.hd_info.Router_num_y):
            for rtx_idx in range(self.hd_info.Router_num_x):
                for pey_idx in range(self.hd_info.PE_num_y):
                    for pex_idx in range(self.hd_info.PE_num_x):
                        pe_pos = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        pe = PE(pe_pos, ModelConfig().input_bit)
                        self.PE_array.append(pe)
        # Interconnect
        self.fetch_array = []
        self.interconnect = Interconnect(self.hd_info.Router_num_y, self.hd_info.Router_num_x, self.input_bit)
        self.interconnect_step = self.hd_info.Router_flit_size / self.input_bit * self.hd_info.cycle_time * self.hd_info.Frequency # scaling from ISAAC
        self.interconnect_step = floor(self.interconnect_step)
        self.interconnect_step = 4000
        self.data_transfer_trigger = []
        self.data_transfer_erp = []
        # Pipeline control
        if not self.isPipeLine:
            self.pipeline_layer_stage = 0
            self.pipeline_stage_record = []
            self.events_each_layer = []
            for layer in range(self.ordergenerator.model_info.layer_length):
                self.events_each_layer.append(0)
            for e in self.Computation_order:
                self.events_each_layer[e.nlayer] += 1
            self.this_layer_event_ctr = 0
            self.this_layer_cycle_ctr = 0
            self.cycles_each_layer = []
            print("events_each_layer:", self.events_each_layer)

        # Statistics
        self.color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.mem_acc_ctr = 0
        self.data_transfer_ctr = 0
        self.act_xb_ctr = 0
        self.pe_saa_stall_cycle = 0

        # Utilization
        self.energy_utilization = []
        self.xbar_utilization = []
        self.pe_state_for_plot = [[], []]
        self.cu_state_for_plot = [[], []]
        self.xb_state_for_plot = [[], []]
        self.buffer_size = []
        for i in range(len(self.PE_array)):
            self.buffer_size.append([])
        self.max_buffer_size = 0 # num of data

    def run(self):
        for e in self.Computation_order:
            if e.preceding_event_count == e.current_number_of_preceding_event:
                if e.event_type == 'edram_rd_ir':
                    pos = e.position_idx
                    rty, rtx, pey, pex, cuy, cux = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
                    idx = pex + pey * self.hd_info.PE_num_x + rtx * self.hd_info.PE_num + rty * self.hd_info.Router_num_x * self.hd_info.PE_num
                    cu_idx = cux + cuy * self.hd_info.CU_num_x
                    self.PE_array[idx].CU_array[cu_idx].edram_rd_ir_erp.append(e)
                else:
                    print("Computation order error: event\"", e.event_type, "\".")
                    print("exit...")
                    exit()

        isDone = False
        while not isDone:
            self.Total_energy_cycle = 0
            self.cycle_ctr += 1
            self.act_xb_ctr = 0
            if not self.isPipeLine:
                self.this_layer_cycle_ctr += 1
            if self.trace:
                print("cycle:", self.cycle_ctr)

            ### Event: edram_rd_ir
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    if not cu.state and not cu.state_edram_rd_ir: # 有無resource
                        for event in cu.edram_rd_ir_erp.copy(): # loop event
                            if event.data_is_transfer != 0:
                                # 此event的資料正在傳輸
                                continue
                            ## Is Data in eDRAM buffer
                            isData_ready = True
                            # inputs: [[num_input, fm_h, fm_w, fm_c]]
                            for inp in event.inputs:
                                data = inp[1:]
                                if not pe.edram_buffer.check([event.nlayer, data]):
                                    # Data not in buffer
                                    if self.trace:
                                        print("\tData not ready for edram_rd_ir. Data: layer", event.nlayer, \
                                            event.event_type, "index:", self.Computation_order.index(event), data,
                                            "position:", cu.position)
                                        #print("\tBuffer:", pe.edram_buffer.buffer)
                                    isData_ready = False
                                    break
                            if not isData_ready: # 無data
                                self.mem_acc_ctr += 1
                                #cu.edram_rd_ir_erp.remove(event)
                                event.data_is_transfer += len(event.inputs)
                                self.fetch_array.append(FetchEvent(event))
                                continue
                            else:
                                if self.trace:
                                    print("\tdo edram_rd_ir, nlayer:", event.nlayer,", cu_pos:", cu.position, ",order index:", self.Computation_order.index(event))
                                    #print("\t\tread data:", event.inputs)
                                if not self.isPipeLine:
                                    self.this_layer_event_ctr += 1

                                num_data = len(event.outputs)
                                self.Total_energy_edram_buffer += self.hd_info.Energy_edram_buffer * self.input_bit * num_data
                                self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit * num_data
                                self.Total_energy_ir_in_cu += self.hd_info.Energy_ir_in_cu * self.input_bit * num_data
                                self.Total_energy_cycle += (self.hd_info.Energy_edram_buffer + self.hd_info.Energy_bus + self.hd_info.Energy_ir_in_cu) * self.input_bit * num_data

                                # 優化: 這裏energy重複算
                                pe.Edram_buffer_energy += self.hd_info.Energy_edram_buffer * self.input_bit * num_data
                                pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit * num_data
                                pe.CU_energy += self.hd_info.Energy_ir_in_cu * self.input_bit * num_data

                                cu.state = True
                                cu.state_edram_rd_ir = True
                                cu.edram_rd_ir_erp.remove(event)
                                cu.edram_rd_event = event
                                cu.edram_rd_cycle_ctr += 1

                                if cu.edram_rd_cycle_ctr == self.edram_read_cycles: # finish edram read
                                    cu.edram_rd_cycle_ctr = 0
                                    ### add next event counter: ou
                                    for proceeding_index in event.proceeding_event:
                                        pro_event = self.Computation_order[proceeding_index]
                                        pro_event.current_number_of_preceding_event += 1

                                        if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                            if self.trace:
                                                pass
                                                #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                            pos = pro_event.position_idx
                                            cu_y, cu_x, xb_y, xb_x = pos[4], pos[5], pos[6], pos[7]
                                            cu_idx = cu_x + cu_y * self.hd_info.CU_num_x
                                            xb_idx = xb_x + xb_y * self.hd_info.Xbar_num_x
                                            cu.ou_trigger.append([pro_event, [cu_idx, xb_idx]])

                                    #if self.isFreeBuffer:
                                    # free buffer
                                    pe_id = self.PE_array.index(pe)
                                    nlayer = event.nlayer
                                    if self.ordergenerator.model_info.layer_list[nlayer].layer_type == "convolution":
                                        for d in event.outputs:
                                            pos = d[2] + d[1]*self.ordergenerator.model_info.input_w[nlayer] + d[3]*self.ordergenerator.model_info.input_w[nlayer]*self.ordergenerator.model_info.input_h[nlayer] # w + h*width + c*height*width
                                            self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                                            if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                                data = d[1:]
                                                self.PE_array[pe_id].edram_buffer.buffer.remove([nlayer, data])
                                    elif self.ordergenerator.model_info.layer_list[nlayer].layer_type == "fully":
                                        for d in event.outputs:
                                            pos = d[1]
                                            self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                                            if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                                data = d[1:]
                                                self.PE_array[pe_id].edram_buffer.buffer.remove([nlayer, data])
                                    else:
                                        print("layer type error.")
                                        exit(0)
                                #print("last_arrived_data", event.last_arrived_data)
                                break
                    elif cu.state and cu.state_edram_rd_ir: # 無resource
                        cu.edram_rd_cycle_ctr += 1
                        if cu.edram_rd_cycle_ctr == self.edram_read_cycles: # finish edram read
                            cu.edram_rd_cycle_ctr = 0
                            cu.state_edram_rd_ir = False
                            event = cu.edram_rd_event
                            ### add next event counter: ou
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1

                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        pass
                                        #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pos = pro_event.position_idx
                                    cu_y, cu_x, xb_y, xb_x = pos[4], pos[5], pos[6], pos[7]
                                    cu_idx = cu_x + cu_y * self.hd_info.CU_num_x
                                    xb_idx = xb_x + xb_y * self.hd_info.Xbar_num_x
                                    cu.ou_trigger.append([pro_event, [cu_idx, xb_idx]])

                                ### add next event counter: ou
                                for proceeding_index in event.proceeding_event:
                                    pro_event = self.Computation_order[proceeding_index]
                                    pro_event.current_number_of_preceding_event += 1

                                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                        if self.trace:
                                            pass
                                            #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                        pos = pro_event.position_idx
                                        cu_y, cu_x, xb_y, xb_x = pos[4], pos[5], pos[6], pos[7]
                                        cu_idx = cu_x + cu_y * self.hd_info.CU_num_x
                                        xb_idx = xb_x + xb_y * self.hd_info.Xbar_num_x
                                        cu.ou_trigger.append([pro_event, [cu_idx, xb_idx]])
                        else:
                            break

                    # bottleneck analysis
                    if not cu.state:
                        # CU idle
                        if not cu.edram_rd_ir_erp:
                            # CU idle, 且event pool是空的
                            cu.pure_idle_time += 1
                        else:
                            # CU idle, 有event在event pool裡面, 這些event不能開始做, 肯定是資料沒到的拉
                            cu.wait_transfer_time += 1
                    else:
                        # CU busy
                        isEventWaiting = False
                        for event in cu.edram_rd_ir_erp:
                            if event.data_is_transfer == 0:
                                # 有event資料已經ready但還不能做
                                isEventWaiting = True
                                break
                        if isEventWaiting:
                            cu.wait_resource_time += 1
                        else:
                            cu.pure_computation_time += 1

            ### Event: ou
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    for xb in cu.XB_array:
                        for event in xb.ou_erp.copy():
                            if not xb.state_ou:
                                if self.trace:
                                    pass
                                    #print("\tdo ou, xb_pos:", xb.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                                if not self.isPipeLine:
                                    self.this_layer_event_ctr += 1
                                self.Total_energy_ir_in_cu += self.hd_info.Energy_ir_in_cu * self.hd_info.OU_h
                                self.Total_energy_dac += self.hd_info.Energy_dac
                                self.Total_energy_crossbar += self.hd_info.Energy_crossbar
                                self.Total_energy_cycle += self.hd_info.Energy_ir_in_cu * self.hd_info.OU_h + \
                                                            self.hd_info.Energy_dac + \
                                                            self.hd_info.Energy_crossbar
                                self.act_xb_ctr += 1

                                # 優化: 這裏energy重複算
                                pe.CU_energy += self.hd_info.Energy_ir_in_cu * self.hd_info.OU_h + \
                                                self.hd_info.Energy_dac + \
                                                self.hd_info.Energy_crossbar

                                xb.state_ou = True
                                xb.ou_erp.remove(event)

                                ### add next event counter: adc
                                for proceeding_index in event.proceeding_event:
                                    pro_event = self.Computation_order[proceeding_index]
                                    pro_event.current_number_of_preceding_event += 1
                                    
                                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                        if self.trace:
                                            pass
                                            #print("\t\tProceeding event is triggered.", pro_event.event_type)

                                        pos = pro_event.position_idx
                                        cu_y, cu_x = pos[4], pos[5]
                                        cu_idx = cu_x + cu_y * self.hd_info.CU_num_x
                                        xb.adc_trigger.append([pro_event, [cu_idx]])

            ### Event: adc
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    for event in cu.adc_erp.copy():
                        for idx in range(len(cu.state_adc)):
                            if not cu.state_adc[idx]:
                                if self.trace:
                                    pass
                                    #print("\tdo adc, cu_pos:", cu.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                                if not self.isPipeLine:
                                    self.this_layer_event_ctr += 1

                                self.Total_energy_adc += self.hd_info.Energy_adc
                                self.Total_energy_cycle += self.hd_info.Energy_adc
                                pe.CU_energy += self.hd_info.Energy_adc

                                cu.adc_erp.remove(event)
                                cu.state_adc[idx] = True

                                ### add next event counter: cu_saa
                                for proceeding_index in event.proceeding_event:
                                    pro_event = self.Computation_order[proceeding_index]
                                    pro_event.current_number_of_preceding_event += 1

                                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                        if self.trace:
                                            pass
                                            #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx, self.Computation_order.index(pro_event))
                                        cu.cu_saa_trigger.append([pro_event, []])
                                break

            ### Event: cu_saa 
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    for event in cu.cu_saa_erp.copy():
                        if self.trace:
                            pass
                            #print("\tdo cu_saa, cu_pos:", cu.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                        if not self.isPipeLine:
                            self.this_layer_event_ctr += 1

                        cu.cu_saa_erp.remove(event)

                        need_saa = 0
                        for saa_amount in range(len(event.inputs)): # amounts of saa 
                            need_saa += 1
                            for idx in range(len(cu.state_cu_saa)):
                                if not cu.state_cu_saa[idx]:
                                    self.Total_energy_or_in_cu += self.hd_info.Energy_or_in_cu * self.input_bit # read
                                    self.Total_energy_cu_shift_and_add += self.hd_info.Energy_shift_and_add
                                    self.Total_energy_or_in_cu += self.hd_info.Energy_or_in_cu * self.input_bit # write
                                    self.Total_energy_cycle += self.hd_info.Energy_or_in_cu * self.input_bit + \
                                                                self.hd_info.Energy_shift_and_add + \
                                                                self.hd_info.Energy_or_in_cu * self.input_bit

                                    # 優化: 這裏energy重複算
                                    pe.CU_energy += self.hd_info.Energy_or_in_cu * self.input_bit + \
                                                    self.hd_info.Energy_shift_and_add + \
                                                    self.hd_info.Energy_or_in_cu * self.input_bit

                                    cu.state_cu_saa[idx] = True
                                    break
                        if need_saa > len(cu.state_cu_saa):
                            print("no enough cu saa per cycle..")
                            exit()
                        
                        ### add next event counter: pe_saa, data_transfer
                        for proceeding_index in event.proceeding_event:
                            pro_event = self.Computation_order[proceeding_index]
                            pro_event.current_number_of_preceding_event += 1

                            if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                if self.trace:
                                    pass
                                    #print("\t\tProceeding event is triggered.", pro_event.event_type)
                                if pro_event.event_type == "pe_saa":
                                    cu.pe_saa_trigger.append([pro_event, []])

                                elif pro_event.event_type == "edram_wr":
                                    pe.edram_wr_trigger.append([pro_event, []])

            ### Event: pe_saa 
            for pe in self.PE_array:
                for event in pe.pe_saa_erp.copy():
                    if event.data_is_transfer != 0: # 此event的資料正在傳輸
                        continue
                    if self.trace:
                        pass
                        #print("\tdo pe_saa, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                    if not self.isPipeLine:
                        self.this_layer_event_ctr += 1
                    
                    pe.pe_saa_erp.remove(event)
                    
                    saa_amount = len(event.inputs[0])
                    rm_data_list = event.inputs[1]
                    if saa_amount > len(pe.state_pe_saa):
                        print("Not enough pe_saa per cycle")
                        exit()

                    for s in range(saa_amount):
                        for idx in range(len(pe.state_pe_saa)):
                            if not pe.state_pe_saa[idx]: # 可能可以直接用s的idx, 不用判斷有沒有resource, 若沒有上面就會exit了
                                self.Total_energy_or += self.hd_info.Energy_or * self.input_bit
                                self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit
                                self.Total_energy_pe_shift_and_add += self.hd_info.Energy_shift_and_add
                                self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit
                                self.Total_energy_or += self.hd_info.Energy_or * self.input_bit
                                self.Total_energy_cycle += self.hd_info.Energy_or * self.input_bit + \
                                                            self.hd_info.Energy_bus * self.input_bit + \
                                                            self.hd_info.Energy_shift_and_add + \
                                                            self.hd_info.Energy_bus * self.input_bit + \
                                                            self.hd_info.Energy_or * self.input_bit
                                # 優化: 這裏energy重複算
                                pe.Or_energy += self.hd_info.Energy_or * self.input_bit
                                pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit
                                pe.Shift_and_add_energy += self.hd_info.Energy_shift_and_add
                                pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit
                                pe.Or_energy += self.hd_info.Energy_or * self.input_bit

                                pe.state_pe_saa[idx] = True
                                break

                        ### add next event counter: activation
                        for proceeding_index in event.proceeding_event:
                            pro_event = self.Computation_order[proceeding_index]
                            pro_event.current_number_of_preceding_event += 1
                            
                            if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                if self.trace:
                                    pass
                                    #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                pe.activation_trigger.append([pro_event, []])
                    #if self.isFreeBuffer:
                    # free buffer
                    for d in rm_data_list:
                        if [event.nlayer, d] in pe.edram_buffer.buffer:
                            pe.edram_buffer.buffer.remove([event.nlayer, d])

                # bottleneck analysis
                if not pe.state_pe_saa[0]:
                    # 因為都是從0開使使用, 0 idle代表全部idle
                    if not pe.pe_saa_erp:
                        # pe saa idle, 且event pool是空的
                        pe.saa_pure_idle_time += 1
                    else:
                        # pe saa idle, 有event在event pool裡面, 這些event不能開始做, 肯定是資料沒到的拉
                        pe.saa_wait_transfer_time += 1
                else:
                    # SAA busy
                    isEventWaiting = False
                    for event in pe.pe_saa_erp:
                        if event.data_is_transfer == 0:
                            # 有event資料已經ready但還不能做
                            isEventWaiting = True
                            break
                    if isEventWaiting:
                        pe.saa_wait_resource_time += 1
                    else:
                        pe.saa_pure_computation_time += 1

            ### Event: activation
            for pe in self.PE_array:
                for event in pe.activation_erp.copy():
                    for idx in range(len(pe.state_activation)):
                        if not pe.state_activation[idx]:
                            if self.trace:
                                pass
                                #print("\tdo activation, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                            if not self.isPipeLine:
                                self.this_layer_event_ctr += 1

                            self.Total_energy_or += self.hd_info.Energy_or * self.input_bit
                            self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit
                            self.Total_energy_activation += self.hd_info.Energy_activation
                            self.Total_energy_cycle += self.hd_info.Energy_bus * self.input_bit + self.hd_info.Energy_activation
                            
                            # 優化: 這裏energy重複算
                            pe.Or_energy += self.hd_info.Energy_or * self.input_bit
                            pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit
                            pe.Activation_energy += self.hd_info.Energy_activation

                            pe.state_activation[idx] = True
                            pe.activation_erp.remove(event)

                            ### add next event counter: edram_wr
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1
                                
                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        pass
                                        #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx, self.Computation_order.index(pro_event))
                                    pe.edram_wr_trigger.append([pro_event, []])
                            break

            ### Event: edram write 
            for pe in self.PE_array:
                for event in pe.edram_wr_erp.copy():
                    for idx in range(len(pe.state_edram_wr)):
                        if not pe.state_edram_wr[idx]:
                            if self.trace:
                                pass
                                print("\tdo edram_wr, pe_pos:", pe.position, "layer:", event.nlayer, \
                                ",order index:", self.Computation_order.index(event), "data:", event.outputs)
                            if not self.isPipeLine:
                                self.this_layer_event_ctr += 1

                            #self.Total_energy_or += self.hd_info.Energy_or * self.input_bit

                            self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit
                            self.Total_energy_edram_buffer += self.hd_info.Energy_edram_buffer * self.input_bit
                            self.Total_energy_cycle += self.hd_info.Energy_bus * self.input_bit + \
                                                        self.hd_info.Energy_edram_buffer * self.input_bit

                            # 優化: 這裏energy重複算
                            pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit
                            pe.Edram_buffer_energy += self.hd_info.Energy_edram_buffer * self.input_bit

                            pe.state_edram_wr[idx] = True
                            pe.edram_wr_erp.remove(event)

                            if len(event.outputs[0]) == 4 and event.outputs[0][3] == "u": # same layer transfer
                                pe.edram_buffer.put([event.nlayer, event.outputs[0]])
                            else:
                                pe.edram_buffer.put([event.nlayer+1, event.outputs[0]])

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
                                    elif pro_event.event_type == "edram_rd_pool":
                                        pe.edram_rd_pool_trigger.append([pro_event, []])
                                    elif pro_event.event_type == "data_transfer":
                                        self.data_transfer_trigger.append([pro_event, []])
                            break

            ### Event: edram_rd_pool
            for pe in self.PE_array:
                if not pe.state_edram_rd_pool:
                    for event in pe.edram_rd_pool_erp:
                        if event.data_is_transfer != 0: # 此event的資料正在傳輸
                            continue
                        isData_ready = True
                        for data in event.inputs:
                            #print(event.nlayer, data)
                            if not pe.edram_buffer.check([event.nlayer, data]):
                                # Data not in buffer
                                if self.trace:
                                    print("\tData not ready for edram_rd_pool. Data: layer", event.nlayer, event.event_type, data)
                                isData_ready = False
                                pe.edram_rd_pool_erp.remove(event)
                                break

                        if not isData_ready:
                            event.data_is_transfer += len(event.inputs)
                            self.fetch_array.append(FetchEvent(event))
                        else:
                            if self.trace:
                                print("\tdo edram_rd_pool, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                            if not self.isPipeLine:
                                self.this_layer_event_ctr += 1

                            num_data = len(event.outputs)
                            self.Total_energy_edram_buffer += self.hd_info.Energy_edram_buffer * self.input_bit * num_data
                            self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit * num_data
                            self.Total_energy_cycle += self.hd_info.Energy_edram_buffer * self.input_bit * num_data + \
                                                        self.hd_info.Energy_bus * self.input_bit * num_data

                            # 優化: 這裏energy重複算
                            pe.Edram_buffer_energy += self.hd_info.Energy_edram_buffer * self.input_bit * num_data
                            pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit * num_data

                            pe.state_edram_rd_pool = True
                            pe.edram_rd_pool_erp.remove(event)

                            ### add next event counter: pooling
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1

                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        pass
                                        #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pos = pro_event.position_idx
                                    pe.pooling_trigger.append([pro_event, []])
                            #if self.isFreeBuffer:
                            # free buffer
                            pe_id = self.PE_array.index(pe)
                            nlayer = event.nlayer
                            if self.ordergenerator.model_info.layer_list[nlayer].layer_type == "pooling":
                                for d in event.outputs:
                                    pos = d[1] + d[0]*self.ordergenerator.model_info.input_w[nlayer] + d[2]*self.ordergenerator.model_info.input_w[nlayer]*self.ordergenerator.model_info.input_h[nlayer] # w + h*width + c*height*width
                                    self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                                    if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                        self.PE_array[pe_id].edram_buffer.buffer.remove([nlayer, d])
                            else:
                                print("layer type error:", self.ordergenerator.model_info.layer_list[nlayer].layer_type)
                                exit(0)

                # bottleneck analysis
                if not pe.state_edram_rd_pool:
                    # Pooling unit idle
                    if not pe.edram_rd_pool_erp:
                        # Pooling unit idle, 且event pool是空的
                        pe.pooling_pure_idle_time += 1
                    else:
                        # Pooling unit idle, 有event在event pool裡面, 這些event不能開始做, 因為資料沒到
                        pe.pooling_wait_transfer_time += 1
                else:
                    # Pooling unit busy
                    isEventWaiting = False
                    for event in pe.edram_rd_pool_erp:
                        if event.data_is_transfer == 0:
                            # 有event資料已經ready但還不能做
                            isEventWaiting = True
                            break
                    if isEventWaiting:
                        pe.pooling_wait_resource_time += 1
                    else:
                        pe.pooling_pure_computation_time += 1

            ### Event: pooling 
            for pe in self.PE_array:
                for event in pe.pooling_erp.copy():
                    for idx in range(len(pe.state_pooling)):
                        if not pe.state_pooling[idx]:
                            if self.trace:
                                print("\tdo pooling, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                            if not self.isPipeLine:
                                self.this_layer_event_ctr += 1

                            self.Total_energy_pooling += self.hd_info.Energy_pooling
                            self.Total_energy_cycle += self.hd_info.Energy_pooling

                            pe.Pooling_energy += self.hd_info.Energy_pooling
                            
                            pe.state_pooling[idx] = True
                            pe.pooling_erp.remove(event)

                            ### add next event counter: edram_wr
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1

                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        pass
                                        #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pe.edram_wr_trigger.append([pro_event, []])
                            break

            ### Interconnect
            for s in range(self.interconnect_step):
                self.interconnect.step()
                self.Total_energy_interconnect += self.interconnect.step_energy_consumption
            # Packets arrive: Store data, trigger event
            arrived_packet = self.interconnect.get_arrived_packet()
            for pk in arrived_packet:
                if self.trace:
                    pass
                    #print("\tArrived packet:", pk)
                rty, rtx = pk.destination[0], pk.destination[1]
                pey, pex = pk.destination[2], pk.destination[3]
                pe_idx = pex + pey * self.hd_info.PE_num_x + rtx * self.hd_info.PE_num + rty * self.hd_info.PE_num * self.hd_info.Router_num_x
                pe = self.PE_array[pe_idx]

                self.Total_energy_edram_buffer += self.hd_info.Energy_edram_buffer * self.input_bit # write
                self.Total_energy_cycle += self.hd_info.Energy_edram_buffer * self.input_bit

                # 優化: 這裏energy重複算
                pe.Edram_buffer_energy += self.hd_info.Energy_edram_buffer * self.input_bit

                pe.edram_buffer.put(pk.data)
                if self.trace:
                    pass
                    #print("put packet data:", pk)

                for pro_event_idx in pk.pro_event_list:
                    if not self.isPipeLine:
                        self.this_layer_event_ctr += 1
                    pro_event = self.Computation_order[pro_event_idx]
                    pro_event.data_is_transfer -= 1

            ### Event: data_transfer
            for event in self.data_transfer_erp.copy():
                if self.trace:
                    print("\tdo data_transfer, layer:", event.nlayer, ",order index:", self.Computation_order.index(event), \
                            "pos:", event.position_idx, "data:", event.outputs)
                self.data_transfer_erp.remove(event)

                src = event.position_idx[0]
                des = event.position_idx[1]

                pro_event_list = [event.proceeding_event[0]]
                if self.Computation_order[pro_event_list[0]].event_type == "edram_rd_ir":
                    data = [event.nlayer+1, event.inputs[0]]
                elif self.Computation_order[pro_event_list[0]].event_type == "edram_rd_pool":
                    data = [event.nlayer+1, event.inputs[0]]
                elif self.Computation_order[pro_event_list[0]].event_type == "pe_saa":
                    data = [event.nlayer, event.inputs[0]]
                else:
                    print("transfer proceeding event type error.\n exit.")
                    exit(0)
                packet = Packet(src, des, data, pro_event_list)
                src_pe_id = src[3] + src[2] * self.hd_info.PE_num_x + \
                            src[1] * self.hd_info.PE_num + \
                            src[0] * self.hd_info.PE_num * self.hd_info.Router_num_x
                src_pe = self.PE_array[src_pe_id]

                self.interconnect.input_packet(packet)

                self.Total_energy_edram_buffer += self.hd_info.Energy_edram_buffer * self.input_bit # read
                self.Total_energy_cycle += self.hd_info.Energy_edram_buffer * self.input_bit

                # 優化: 這裏energy重複算
                src_pe.Edram_buffer_energy += self.hd_info.Energy_edram_buffer * self.input_bit

                ### add next event counter
                des_pe_id = des[3] + des[2] * self.hd_info.PE_num_x + \
                            des[1] * self.hd_info.PE_num + \
                            des[0] * self.hd_info.PE_num * self.hd_info.Router_num_x
                des_pe = self.PE_array[des_pe_id]
                for proceeding_index in event.proceeding_event: # 這個loop只會執行一次(一個transfer後都只接一個event)
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1
                    pro_event.data_is_transfer += 1
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            pass
                            print("\t\tProceeding event is triggered.", pro_event.event_type)
                        if pro_event.event_type == "edram_rd_ir":
                            cuy, cux = pro_event.position_idx[4], pro_event.position_idx[5]
                            cu_idx = cux + cuy * self.hd_info.CU_num_x
                            des_pe.edram_rd_ir_trigger.append([pro_event, [cu_idx]])
                        elif pro_event.event_type == "edram_rd_pool":
                            des_pe.edram_rd_pool_trigger.append([pro_event, []])
                        elif pro_event.event_type == "pe_saa":
                            des_pe.pe_saa_trigger.append([pro_event, []])
                        else:
                            print("pro_event type error:", pro_event.event_type)
                            exit()

                #if self.isFreeBuffer: # 把不再需要用到的資料free掉
                pe_id = src[3] + src[2]*self.hd_info.PE_num_x + \
                        src[1]*self.hd_info.PE_num + src[0]*self.hd_info.PE_num*self.hd_info.Router_num_x
                if len(event.outputs[0]) == 4 and event.outputs[0][3] == "u": # same layer data transfer
                    data = event.outputs[0]
                    if [event.nlayer, data] in self.PE_array[pe_id].edram_buffer.buffer:
                        self.PE_array[pe_id].edram_buffer.buffer.remove([event.nlayer, data])
                else:
                    nlayer = event.nlayer+1
                    if self.ordergenerator.model_info.layer_list[nlayer-1].layer_type != "fully":
                        for d in event.outputs:
                            pos = d[1] + d[0]*self.ordergenerator.model_info.input_w[nlayer] + d[2]*self.ordergenerator.model_info.input_w[nlayer]*self.ordergenerator.model_info.input_h[nlayer] # w + h*width + c*height*width
                            self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                            if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                self.PE_array[pe_id].edram_buffer.buffer.remove([nlayer, d])
                    else:
                        for d in event.outputs:
                            pos = d[1]
                            self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                            if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                self.PE_array[pe_id].edram_buffer.buffer.remove([nlayer, d])

            ### Fetch data from off-chip memory
            des_dict = dict()
            for FE in self.fetch_array.copy():
                FE.cycles_counter += 1
                if FE.cycles_counter == FE.fetch_cycle:
                    src  = (0, FE.event.position_idx[1], -1, -1)
                    des  = FE.event.position_idx[0:4]
                    if FE.event.event_type == "edram_rd_ir":
                        if not self.isPipeLine:
                            self.this_layer_event_ctr -= len(FE.event.inputs)
                        FE.event.preceding_event_count += len(FE.event.inputs)
                        if des not in des_dict:
                            des_dict[des] = []
                        for inp in FE.event.inputs:
                            data = inp[1:]
                            data = [FE.event.nlayer, data]
                            pro_event_idx = self.Computation_order.index(FE.event)

                            isDataInDict = False
                            for data_pro_event in des_dict[des]:
                                if data == data_pro_event[0]:
                                    data_pro_event[1].append(pro_event_idx)
                                    isDataInDict = True
                                    break
                            if not isDataInDict:
                                des_dict[des].append([data, [pro_event_idx]])
                    elif FE.event.event_type == "edram_rd_pool":
                        if not self.isPipeLine:
                            self.this_layer_event_ctr -= len(FE.event.inputs)
                        FE.event.preceding_event_count += len(FE.event.inputs)
                        if des not in des_dict:
                            des_dict[des] = []
                        for data in FE.event.inputs:
                            data = [FE.event.nlayer, data]
                            pro_event_idx = self.Computation_order.index(FE.event)
                            isDataInDict = False
                            for data_pro_event in des_dict[des]:
                                if data == data_pro_event[0]:
                                    data_pro_event[1].append(pro_event_idx)
                                    isDataInDict = True
                                    break
                            if not isDataInDict:
                                des_dict[des].append([data, [pro_event_idx]])
                    self.fetch_array.remove(FE)
            for des in des_dict:
                src = (0, des[1], -1, -1)
                for data_pro_event in des_dict[des]:
                    data, pro_event_idx = data_pro_event[0], data_pro_event[1]
                    packet = Packet(src, des, data, pro_event_idx)
                    self.interconnect.input_packet(packet)


            ### Pipeline stage control ###
            if not self.isPipeLine:
                self.pipeline_stage_record.append(self.pipeline_layer_stage)
                if self.this_layer_event_ctr == self.events_each_layer[self.pipeline_layer_stage]:
                    self.pipeline_layer_stage += 1
                    self.cycles_each_layer.append(self.this_layer_cycle_ctr)
                    self.this_layer_event_ctr = 0
                    self.this_layer_cycle_ctr = 0

            ### Trigger events ###
            ## Trigger interconnect
            for trigger in self.data_transfer_trigger.copy():
                pro_event = trigger[0]
                if not self.isPipeLine:
                    if pro_event.nlayer == self.pipeline_layer_stage:
                        self.data_transfer_erp.append(pro_event)
                        self.data_transfer_trigger.remove(trigger)
                else:
                    self.data_transfer_erp.append(pro_event)
                    self.data_transfer_trigger.remove(trigger)
            for pe in self.PE_array:
                ## Trigger activation 
                for trigger in pe.activation_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.activation_erp.append(pro_event)
                            pe.activation_trigger.remove(trigger)
                    else:
                        pe.activation_erp.append(pro_event)
                        pe.activation_trigger.remove(trigger)
                ## Trigger edram_wr 
                for trigger in pe.edram_wr_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.edram_wr_erp.append(pro_event)
                            pe.edram_wr_trigger.remove(trigger)
                    else:
                        pe.edram_wr_erp.append(pro_event)
                        pe.edram_wr_trigger.remove(trigger)
                ## Trigger edram_rd_ir
                for trigger in pe.edram_rd_ir_trigger.copy():
                    pro_event = trigger[0]
                    cu_idx = trigger[1][0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.CU_array[cu_idx].edram_rd_ir_erp.append(pro_event)
                            pe.edram_rd_ir_trigger.remove(trigger)
                    else:
                        pe.CU_array[cu_idx].edram_rd_ir_erp.append(pro_event)
                        pe.edram_rd_ir_trigger.remove(trigger)
                ## Trigger pooling 
                for trigger in pe.pooling_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.pooling_erp.append(pro_event)
                            pe.pooling_trigger.remove(trigger)
                    else:
                        pe.pooling_erp.append(pro_event)
                        pe.pooling_trigger.remove(trigger)
                ## Trigger edram_rd_ir_pool 
                for trigger in pe.edram_rd_pool_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.edram_rd_pool_erp.append(pro_event)
                            pe.edram_rd_pool_trigger.remove(trigger)
                    else:
                        pe.edram_rd_pool_erp.append(pro_event)
                        pe.edram_rd_pool_trigger.remove(trigger)
                for cu in pe.CU_array:
                    ## Trigger ou operation 
                    for trigger in cu.ou_trigger.copy():
                        pro_event = trigger[0]
                        xb_idx = trigger[1][1]
                        if not self.isPipeLine:
                            if pro_event.nlayer == self.pipeline_layer_stage:
                                cu.XB_array[xb_idx].ou_erp.append(pro_event)
                                cu.ou_trigger.remove(trigger)
                        else:
                            cu.XB_array[xb_idx].ou_erp.append(pro_event)
                            cu.ou_trigger.remove(trigger)
                    ## Trigger pe saa
                    for trigger in cu.pe_saa_trigger.copy():
                        pro_event = trigger[0]
                        if not self.isPipeLine:
                            if pro_event.nlayer == self.pipeline_layer_stage:
                                pe.pe_saa_erp.append(pro_event)
                                cu.pe_saa_trigger.remove(trigger)
                        else:
                            pe.pe_saa_erp.append(pro_event) 
                            cu.pe_saa_trigger.remove(trigger)
                    ### Trigger adc
                    for xb in cu.XB_array:
                        for trigger in xb.adc_trigger.copy():
                            pro_event = trigger[0]
                            cu_idx = trigger[1][0]
                            if not self.isPipeLine:
                                if pro_event.nlayer == self.pipeline_layer_stage:
                                    pe.CU_array[cu_idx].adc_erp.append(pro_event)
                                    xb.adc_trigger.remove(trigger)
                            else:
                                pe.CU_array[cu_idx].adc_erp.append(pro_event)
                                xb.adc_trigger.remove(trigger)
                    ### Trigger cu_saa
                    for trigger in cu.cu_saa_trigger.copy():
                        pro_event = trigger[0]
                        if not self.isPipeLine:
                            if pro_event.nlayer == self.pipeline_layer_stage:
                                cu.cu_saa_erp.append(pro_event)
                                cu.cu_saa_trigger.remove(trigger)
                        else:
                            cu.cu_saa_erp.append(pro_event)
                            cu.cu_saa_trigger.remove(trigger)
                ## Trigger pe saa (for data transfer) 
                for trigger in pe.pe_saa_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.pe_saa_erp.append(pro_event)
                            pe.pe_saa_trigger.remove(trigger)
                    else:
                        pe.pe_saa_erp.append(pro_event)
                        pe.pe_saa_trigger.remove(trigger)
                        
            ### Record State ###
            for pe in self.PE_array:
                if pe.check_state():
                    self.pe_state_for_plot[0].append(self.cycle_ctr)
                    self.pe_state_for_plot[1].append(self.PE_array.index(pe))
                for cu in pe.CU_array:
                    if cu.check_state():
                        self.cu_state_for_plot[0].append(self.cycle_ctr)
                        self.cu_state_for_plot[1].append(self.PE_array.index(pe) * self.hd_info.CU_num + \
                                                        pe.CU_array.index(cu))
                    
                    for xb in cu.XB_array:
                        if xb.check_state():
                            self.xb_state_for_plot[0].append(self.cycle_ctr)
                            self.xb_state_for_plot[1].append( \
                                self.PE_array.index(pe) * self.hd_info.CU_num + \
                                pe.CU_array.index(cu) * self.hd_info.Xbar_num + cu.XB_array.index(xb)
                                )
            ### Reset ###
            for pe in self.PE_array:
                pe.reset()
                for cu in pe.CU_array:
                    cu.reset()
                    for xb in cu.XB_array:
                        xb.reset(self.cycle_ctr)
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    if cu.state:
                        if cu.cu_saa_erp or cu.adc_erp:
                            continue
                        if cu.state_edram_rd_ir:
                            continue

                        isCUBusy = False
                        for xb in cu.XB_array:
                            if xb.ou_erp:
                                isCUBusy = True
                                break
                        if isCUBusy:
                            continue
                        cu.state = False

            self.energy_utilization.append(self.Total_energy_cycle*1e09)
            self.xbar_utilization.append(self.act_xb_ctr)
            
            ### Finish?
            isDone = True
            if self.fetch_array or self.data_transfer_erp or self.data_transfer_trigger:
                isDone = False
            if self.interconnect.busy():
                isDone = False
            for pe in self.PE_array:
                if pe.pe_saa_erp or pe.activation_erp or pe.pooling_erp or pe.edram_wr_erp or pe.edram_rd_pool_erp:
                    isDone = False
                    break
                elif pe.activation_trigger or pe.edram_wr_trigger or pe.edram_rd_pool_trigger \
                    or pe.edram_rd_ir_trigger or pe.pooling_trigger:
                    isDone = False
                    break
                for cu in pe.CU_array:
                    if cu.state or cu.edram_rd_ir_erp or cu.cu_saa_erp or cu.adc_erp:
                        isDone = False
                        break
                    elif cu.ou_trigger or cu.pe_saa_trigger or cu.cu_saa_trigger:
                        isDone = False
                        break
                    for xb in cu.XB_array:
                        if xb.adc_trigger:
                            isDone = False
                            break
                    if not isDone:
                        break
                if not isDone:
                    break

            # Buffer size utilization
            for pe_idx in range(len(self.PE_array)):
                self.buffer_size[pe_idx].append(self.PE_array[pe_idx].edram_buffer.count())
                self.max_buffer_size = max(len(pe.edram_buffer.buffer), self.max_buffer_size)


    def print_statistics_result(self):
        print("Total Cycles:", self.cycle_ctr)
        if not self.isPipeLine:
            print("Cycles each layer:", self.cycles_each_layer)
        print("Cycles time:", self.hd_info.cycle_time, "ns\n")
        
        print('memory accesss times:', self.mem_acc_ctr)
        print()

        isBottleneckAnalysis = False
        if isBottleneckAnalysis:
            print("Bottleneck analysis")
            # for pe in self.PE_array:
            #     print("pe", self.PE_array.index(pe))
            #     print("\tpe.saa_wait_transfer", pe.saa_wait_transfer)
            #     print("\tpe.saa_wait_resource", pe.saa_wait_resource)
            #     print("\tpe.pool_wait_transfer", pe.pool_wait_transfer)
            #     print("\tpe.pool_wait_resource", pe.pool_wait_resource)
            for pe in self.PE_array:
                print("pe", self.PE_array.index(pe))
                for cu in pe.CU_array:
                    print("\tcu", pe.CU_array.index(cu))
                    print("\t\tcu.pure_idle_time", cu.pure_idle_time)
                    print("\t\tcu.wait_transfer_time", cu.wait_transfer_time)
                    print("\t\tcu.wait_resource_time", cu.wait_resource_time)
                    print("\t\tcu.wait_computation_time", cu.wait_computation_time)
        
        ### non-pipeline stage
        if not self.isPipeLine:
            with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/stage.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in range(self.cycle_ctr):
                    writer.writerow([row+1, self.pipeline_stage_record[row]])

        fre = 1
        # ### Energy per 100 cycle
        # with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Energy.csv', 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     for row in range(0, self.cycle_ctr, fre):
        #         writer.writerow([row+1, self.energy_utilization[row]])
        # plt.bar(range(1, self.cycle_ctr+1), self.energy_utilization)
        # #plt.show()
        # plt.title(self.mapping_str+", "+self.scheduling_str)
        # plt.ylabel('Energy (nJ)')
        # plt.xlabel('Cycle')
        # plt.ylim([0, 20])
        # #plt.xlim([0,])
        # plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/energy_utilization.png')
        # plt.clf()

        ### PE usage
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/PE_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, len(self.pe_state_for_plot[0]), fre):
                writer.writerow([self.pe_state_for_plot[0][row], self.pe_state_for_plot[1][row]])

        plt.scatter(self.pe_state_for_plot[0], self.pe_state_for_plot[1], s=3, c='blue')
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('Cycle')
        plt.ylabel('PE index')
        plt.ylim([-1, 16])
        plt.xticks(np.arange(0,250, 20), fontsize=8)
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/PE_utilization.png')
        plt.clf()
        
        ### CU usage
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/CU_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, len(self.cu_state_for_plot[0]), fre):
                writer.writerow([self.cu_state_for_plot[0][row], self.cu_state_for_plot[1][row]])

        plt.scatter(self.cu_state_for_plot[0], self.cu_state_for_plot[1], s=2, c='blue')
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('Cycle')
        plt.ylabel('CU index')
        plt.yticks(np.arange(-1,64, 2), fontsize=6)
        plt.xticks(np.arange(0,250, 20), fontsize=8)
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/CU_utilization.png')
        plt.clf()

        ### XB usage
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/XB_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, len(self.xb_state_for_plot[0]), fre):
                writer.writerow([self.xb_state_for_plot[0][row], self.xb_state_for_plot[1][row]])

        plt.scatter(self.xb_state_for_plot[0], self.xb_state_for_plot[1], s=2, c='blue')
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('Cycle')
        plt.ylabel('XB index')
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/XB_utilization.png')
        plt.clf()
  
        ### Buffer analysis
        self.buffer_analysis()

        ### Energy breakdown
        self.energy_breakdown()

        ### Bottleneck analysis
        self.bottleneck_statistics()

    def bottleneck_statistics(self):
        ## CU
        total_pe_num = len(self.PE_array)
        idx = np.arange(total_pe_num)
        pure_idle_time, wait_transfer_time, wait_resource_time, pure_computation_time = [], [], [], []
        pure_idle_time_total, wait_transfer_time_total, wait_resource_time_total, pure_computation_time_total = 0, 0, 0, 0
        for j in range(self.hd_info.CU_num):
            pure_idle_time.append([])
            wait_transfer_time.append([])
            wait_resource_time.append([])
            pure_computation_time.append([])
            for i in range(len(self.PE_array)):
                pure_idle_time[j].append(self.PE_array[i].CU_array[j].pure_idle_time)
                wait_transfer_time[j].append(self.PE_array[i].CU_array[j].wait_transfer_time)
                wait_resource_time[j].append(self.PE_array[i].CU_array[j].wait_resource_time)
                pure_computation_time[j].append(self.PE_array[i].CU_array[j].pure_computation_time)
                pure_idle_time_total += pure_idle_time[j][-1]
                wait_transfer_time_total += wait_transfer_time[j][-1]
                wait_resource_time_total += wait_resource_time[j][-1]
                pure_computation_time_total +=pure_computation_time[j][-1]
        pure_idle_time = np.array(pure_idle_time)
        wait_transfer_time = np.array(wait_transfer_time)
        wait_resource_time = np.array(wait_resource_time)
        pure_computation_time = np.array(pure_computation_time)
        for i in range(self.hd_info.CU_num):
            plt.bar(idx+0.2*i, pure_idle_time[i], width = 0.2, color='b', edgecolor = 'white')
            plt.bar(idx+0.2*i, wait_transfer_time[i], bottom=pure_idle_time[i], color='r',  width=0.2, edgecolor = 'white')
            plt.bar(idx+0.2*i, wait_resource_time[i], bottom=pure_idle_time[i]+wait_transfer_time[i], color='g', width=0.2, edgecolor = 'white')
            plt.bar(idx+0.2*i, pure_computation_time[i], bottom=pure_idle_time[i]+wait_transfer_time[i]+wait_resource_time[i], color='y', width=0.2, edgecolor = 'white')
        plt.legend(["pure_idle", "wait_transfer", "wait_resource", "pure_computation"])
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('PE index')
        plt.ylabel('Cycle')
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Bottleneck_CU.png')
        plt.clf()

        print("CU Bottleneck analysis:")
        print("\tTotal pure_idle:", pure_idle_time_total)
        print("\tTotal wait_transfer:", wait_transfer_time_total)
        print("\tTotal wait_resource:", wait_resource_time_total)
        print("\tTotal pure_computation:", pure_computation_time_total)
        print()
        total_cu_num = total_pe_num * self.hd_info.CU_num
        print("\tAverage pure_idle, ", pure_idle_time_total/total_cu_num, end="")
        print("(" + str(pure_idle_time_total/total_cu_num/self.cycle_ctr*100) + "%)")
        print("\tAverage wait_transfer, ", wait_transfer_time_total/total_cu_num, end="")
        print("(" + str(wait_transfer_time_total/total_cu_num/self.cycle_ctr*100) + "%)")
        print("\tAverage wait_resource, ", wait_resource_time_total/total_cu_num, end="")
        print("(" + str(wait_resource_time_total/total_cu_num/self.cycle_ctr*100) + "%)")
        print("\tAverage pure_computation, ", pure_computation_time_total/total_cu_num, end="")
        print("(" + str(pure_computation_time_total/total_cu_num/self.cycle_ctr*100) + "%)")

        # PE SAA
        total_pe_num = len(self.PE_array)
        idx = np.arange(total_pe_num)
        pure_idle_time = list()
        wait_transfer_time = list()
        wait_resource_time = list()
        pure_computation_time = list()
        pure_idle_time_total = 0
        wait_transfer_time_total = 0
        wait_resource_time_total = 0
        pure_computation_time_total = 0
        for i in range(len(self.PE_array)):
            pure_idle_time.append(self.PE_array[i].saa_pure_idle_time)
            wait_transfer_time.append(self.PE_array[i].saa_wait_transfer_time)
            wait_resource_time.append(self.PE_array[i].saa_wait_resource_time)
            pure_computation_time.append(self.PE_array[i].saa_pure_computation_time)
            pure_idle_time_total += pure_idle_time[-1]
            wait_transfer_time_total += wait_transfer_time[-1]
            wait_resource_time_total += wait_resource_time[-1]
            pure_computation_time_total +=pure_computation_time[-1]
        pure_idle_time = np.array(pure_idle_time)
        wait_transfer_time = np.array(wait_transfer_time)
        wait_resource_time = np.array(wait_resource_time)
        pure_computation_time = np.array(pure_computation_time)
        plt.bar(idx, pure_idle_time, color='b',  width=0.8, edgecolor = 'white')
        plt.bar(idx, wait_transfer_time, bottom=pure_idle_time, color='r',  width=0.8, edgecolor = 'white')
        plt.bar(idx, wait_resource_time, bottom=pure_idle_time+wait_transfer_time, color='g',  width=0.8, edgecolor = 'white')
        plt.bar(idx, pure_computation_time, bottom=pure_idle_time+wait_transfer_time+wait_resource_time, color='y',  width=0.8, edgecolor = 'white')
        plt.legend(["pure_idle", "wait_transfer", "wait_resource", "pure_computation"])
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('PE index')
        plt.ylabel('Cycle')
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Bottleneck_PE_SAA.png')
        plt.clf()
        print("PE SAA Bottleneck analysis:")
        print("\tTotal pure_idle:", pure_idle_time_total)
        print("\tTotal wait_transfer:", wait_transfer_time_total)
        print("\tTotal wait_resource:", wait_resource_time_total)
        print("\tTotal pure_computation:", pure_computation_time_total)
        print()
        print("\tAverage pure_idle, ", pure_idle_time_total/total_pe_num, end="")
        print("(" + str(pure_idle_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print("\tAverage wait_transfer, ", wait_transfer_time_total/total_pe_num, end="")
        print("(" + str(wait_transfer_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print("\tAverage wait_resource, ", wait_resource_time_total/total_pe_num, end="")
        print("(" + str(wait_resource_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print("\tAverage pure_computation, ", pure_computation_time_total/total_pe_num, end="")
        print("(" + str(pure_computation_time_total/total_pe_num/self.cycle_ctr*100) + "%)")

        # Pooling
        total_pe_num = len(self.PE_array)
        idx = np.arange(total_pe_num)
        pure_idle_time = list()
        wait_transfer_time = list()
        wait_resource_time = list()
        pure_computation_time = list()
        pure_idle_time_total = 0
        wait_transfer_time_total = 0
        wait_resource_time_total = 0
        pure_computation_time_total = 0
        for i in range(len(self.PE_array)):
            pure_idle_time.append(self.PE_array[i].pooling_pure_idle_time)
            wait_transfer_time.append(self.PE_array[i].pooling_wait_transfer_time)
            wait_resource_time.append(self.PE_array[i].pooling_wait_resource_time)
            pure_computation_time.append(self.PE_array[i].pooling_pure_computation_time)
            pure_idle_time_total += pure_idle_time[-1]
            wait_transfer_time_total += wait_transfer_time[-1]
            wait_resource_time_total += wait_resource_time[-1]
            pure_computation_time_total +=pure_computation_time[-1]
        pure_idle_time = np.array(pure_idle_time)
        wait_transfer_time = np.array(wait_transfer_time)
        wait_resource_time = np.array(wait_resource_time)
        pure_computation_time = np.array(pure_computation_time)
        plt.bar(idx, pure_idle_time, color='b',  width=0.8, edgecolor = 'white')
        plt.bar(idx, wait_transfer_time, bottom=pure_idle_time, color='r',  width=0.8, edgecolor = 'white')
        plt.bar(idx, wait_resource_time, bottom=pure_idle_time+wait_transfer_time, color='g',  width=0.8, edgecolor = 'white')
        plt.bar(idx, pure_computation_time, bottom=pure_idle_time+wait_transfer_time+wait_resource_time, color='y',  width=0.8, edgecolor = 'white')
        plt.legend(["pure_idle", "wait_transfer", "wait_resource", "pure_computation"])
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('PE index')
        plt.ylabel('Cycle')
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Bottleneck_Pooling.png')
        plt.clf()
        print("Pooling Bottleneck analysis:")
        print("\tTotal pure_idle:", pure_idle_time_total)
        print("\tTotal wait_transfer:", wait_transfer_time_total)
        print("\tTotal wait_resource:", wait_resource_time_total)
        print("\tTotal pure_computation:", pure_computation_time_total)
        print()
        print("\tAverage pure_idle, ", pure_idle_time_total/total_pe_num, end="")
        print("(" + str(pure_idle_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print("\tAverage wait_transfer, ", wait_transfer_time_total/total_pe_num, end="")
        print("(" + str(wait_transfer_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print("\tAverage wait_resource, ", wait_resource_time_total/total_pe_num, end="")
        print("(" + str(wait_resource_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print("\tAverage pure_computation, ", pure_computation_time_total/total_pe_num, end="")
        print("(" + str(pure_computation_time_total/total_pe_num/self.cycle_ctr*100) + "%)")

    def energy_breakdown(self):
        self.Total_energy_cu = self.Total_energy_cu_shift_and_add + \
                                self.Total_energy_adc + \
                                self.Total_energy_dac + \
                                self.Total_energy_crossbar + \
                                self.Total_energy_ir_in_cu + \
                                self.Total_energy_or_in_cu
        self.Total_energy_pe = self.Total_energy_cu + \
                                self.Total_energy_edram_buffer + \
                                self.Total_energy_bus + \
                                self.Total_energy_activation + \
                                self.Total_energy_pe_shift_and_add + \
                                self.Total_energy_pooling + \
                                self.Total_energy_or
        self.Total_energy = self.Total_energy_pe + self.Total_energy_interconnect

        print("Energy breakdown:")
        print("\tTotal:", self.Total_energy, "nJ")
        print("\tChip level")
        print("\t\tPE: %.4e (%.2f%%)" %(self.Total_energy_pe, self.Total_energy_pe/self.Total_energy*100))
        print("\t\tInterconnect: %.4e (%.2f%%)" %(self.Total_energy_interconnect, self.Total_energy_interconnect/self.Total_energy*100))
        print()
        print("\tPE level")
        print("\t\tCU: %.4e (%.2f%%)" %(self.Total_energy_cu, self.Total_energy_cu/self.Total_energy_pe*100))
        print("\t\tEdram Buffer: %.4e (%.2f%%)" %(self.Total_energy_edram_buffer, self.Total_energy_edram_buffer/self.Total_energy_pe*100))
        print("\t\tBus: %.4e (%.2f%%)" %(self.Total_energy_bus, self.Total_energy_bus/self.Total_energy_pe*100))
        print("\t\tActivation: %.4e (%.2f%%)" %(self.Total_energy_activation, self.Total_energy_activation/self.Total_energy_pe*100))
        print("\t\tShift and Add: %.4e (%.2f%%)" %(self.Total_energy_pe_shift_and_add, self.Total_energy_pe_shift_and_add/self.Total_energy_pe*100))
        print("\t\tPooling: %.4e (%.2f%%)" %(self.Total_energy_pooling, self.Total_energy_pooling/self.Total_energy_pe*100))
        print("\t\tOR: %.4e (%.2f%%)" %(self.Total_energy_or, self.Total_energy_or/self.Total_energy_pe*100))
        print()
        print("\tCU level")
        print("\t\tShift and Add: %.4e (%.2f%%)" %(self.Total_energy_cu_shift_and_add, self.Total_energy_cu_shift_and_add/self.Total_energy_cu*100))
        print("\t\tADC: %.4e (%.2f%%)" %(self.Total_energy_adc, self.Total_energy_adc/self.Total_energy_cu*100))
        print("\t\tDAC: %.4e (%.2f%%)" %(self.Total_energy_dac, self.Total_energy_dac/self.Total_energy_cu*100))
        print("\t\tCrossbar Array: %.4e (%.2f%%)" %(self.Total_energy_crossbar, self.Total_energy_crossbar/self.Total_energy_cu*100))
        print("\t\tIR: %.4e (%.2f%%)" %(self.Total_energy_ir_in_cu, self.Total_energy_ir_in_cu/self.Total_energy_cu*100))
        print("\t\tOR: %.4e (%.2f%%)" %(self.Total_energy_or_in_cu, self.Total_energy_or_in_cu/self.Total_energy_cu*100))
        print()

        # Chip 圓餅圖
        labels = 'PE', 'Interconnect'
        value = [self.Total_energy_pe, self.Total_energy_interconnect]
        plt.pie(value , labels = labels, autopct='%1.1f%%')
        plt.axis('equal')
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Energy_breakdown_chip.png')
        plt.clf()

        # PE breakdown
        total_pe_num = len(self.PE_array)
        idx = np.arange(total_pe_num)
        CU_energy, Edram_buffer_energy = list(), list()
        Bus_energy, Shift_and_add_energy = list(), list()
        Or_energy, Activation_energy, Pooling_energy = list(), list(), list()
        for pe in self.PE_array:
            CU_energy.append(pe.CU_energy)
            Edram_buffer_energy.append(pe.Edram_buffer_energy)
            Bus_energy.append(pe.Bus_energy)
            Shift_and_add_energy.append(pe.Shift_and_add_energy)
            Or_energy.append(pe.Or_energy)
            Activation_energy.append(pe.Activation_energy)
            Pooling_energy.append(pe.Pooling_energy)
        CU_energy, Edram_buffer_energy = np.array(CU_energy), np.array(Edram_buffer_energy)
        Bus_energy, Shift_and_add_energy = np.array(Bus_energy), np.array(Shift_and_add_energy)
        Or_energy, Activation_energy = np.array(Or_energy), np.array(Activation_energy)
        Pooling_energy = np.array(Pooling_energy)

        plt.bar(idx, CU_energy,  width=0.8)
        energy_sum = CU_energy
        plt.bar(idx, Edram_buffer_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Edram_buffer_energy
        plt.bar(idx, Bus_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Bus_energy
        plt.bar(idx, Shift_and_add_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Shift_and_add_energy
        plt.bar(idx, Or_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Or_energy
        plt.bar(idx, Activation_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Activation_energy
        plt.bar(idx, Pooling_energy, bottom=energy_sum,  width=0.8)
        plt.xlabel('PE index')
        plt.ylabel('Energy (nJ)')
        plt.legend(["CU_energy", "Edram_buffer_energy", "Bus_energy", "Shift_and_add_energy", "Or_energy", "Activation_energy", "Pooling_energy"])
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Energy_breakdown_PE.png')
        plt.clf()

        # PE breakdown (percentage)
        for i in range(len(self.PE_array)):
            sum_energy = CU_energy[i] + Edram_buffer_energy[i] + Bus_energy[i] + \
                         Shift_and_add_energy[i] + Or_energy[i] + \
                         Activation_energy[i] + Pooling_energy[i]
            if sum_energy == 0:
                CU_energy[i], Edram_buffer_energy[i] = 0, 0
                Bus_energy[i], Shift_and_add_energy[i] = 0, 0
                Or_energy[i], Activation_energy[i] = 0, 0
                Pooling_energy[i] = 0
            else:
                CU_energy[i], Edram_buffer_energy[i] = CU_energy[i]*100/sum_energy, Edram_buffer_energy[i]*100/sum_energy
                Bus_energy[i], Shift_and_add_energy[i] = Bus_energy[i]*100/sum_energy, Shift_and_add_energy[i]*100/sum_energy
                Or_energy[i], Activation_energy[i] = Or_energy[i]*100/sum_energy, Activation_energy[i]*100/sum_energy
                Pooling_energy[i] = Pooling_energy[i]*100/sum_energy

        plt.bar(idx, CU_energy,  width=0.8)
        energy_sum = CU_energy
        plt.bar(idx, Edram_buffer_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Edram_buffer_energy
        plt.bar(idx, Bus_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Bus_energy
        plt.bar(idx, Shift_and_add_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Shift_and_add_energy
        plt.bar(idx, Or_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Or_energy
        plt.bar(idx, Activation_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Activation_energy
        plt.bar(idx, Pooling_energy, bottom=energy_sum,  width=0.8)
        plt.xlabel('PE index')
        plt.ylabel('Energy (%)')
        plt.legend(["CU_energy", "Edram_buffer_energy", "Bus_energy", "Shift_and_add_energy", "Or_energy", "Activation_energy", "Pooling_energy"])
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Energy_breakdown_PE_percentage.png')
        plt.clf()

    def buffer_analysis(self):
        ### Utilization
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/OnchipBuffer_C.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(self.cycle_ctr):
                c = [row+1]
                for i in range(len(self.PE_array)):
                    c.append(self.buffer_size[i][row])
                writer.writerow(c)
        self.buffer_size = np.array(self.buffer_size)
        for i in range(len(self.PE_array)):
            plt.plot(range(1, self.cycle_ctr+1), self.buffer_size[i] * self.input_bit/8/1000, label="PE"+str(i)) #, c=self.color[i])
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('Cycle')
        plt.ylabel('Buffer size(KB)')
        plt.legend(loc='best', prop={'size': 6})
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Buffer_utilization_C.png')
        plt.clf()

        ### Maximal usage
        for i in range(len(self.PE_array)):
            plt.bar(i, self.PE_array[i].edram_buffer.maximal_usage * self.input_bit/8/1000, color='b', width=0.8)
        plt.legend(["Maximal usage"])
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('PE index')
        plt.ylabel('Buffer Size(KB)')
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Buffer_maximal_usage.png')
        plt.clf()
