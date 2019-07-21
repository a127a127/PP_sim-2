import os
import csv
from PE import PE
import numpy as np
from math import ceil
from EventMetaData import EventMetaData
from FetchEvent import FetchEvent
from HardwareMetaData import HardwareMetaData

from Interconnect import Interconnect
from Packet import Packet

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Controller(object):
    def __init__(self, ordergenerator, isPipeLine, trace, mapping_str):
        self.mapping_str = mapping_str
        self.ordergenerator = ordergenerator
        self.isPipeLine = isPipeLine
        self.trace = trace
        self.Computation_order = self.ordergenerator.Computation_order
        self.trace = trace

        self.cycle_ctr = 0
        self.cycle_energy = 0
        self.cycle_time = 3.45888e-08

        self.hardware_information = HardwareMetaData()
        self.eDRAM_buffer_size = self.hardware_information.eDRAM_buffer_size
        # Leakage
        self.eDRAM_buffer_leakage = self.hardware_information.eDRAM_buffer_leakage
        self.Router_leakage = self.hardware_information.Router_leakage
        self.SA_leakage = self.hardware_information.SA_leakage
        self.Act_leakage = self.hardware_information.Act_leakage
        self.PE_SAA_leakage = self.hardware_information.PE_SAA_leakage
        self.Pool_leakage = self.hardware_information.Pool_leakage
        self.DAC_leakage = self.hardware_information.DAC_leakage
        self.MUX_leakage = self.hardware_information.MUX_leakage
        self.SA_leakage = self.hardware_information.SA_leakage
        self.Crossbar_leakage = self.hardware_information.Crossbar_leakage
        self.CU_SAA_leakage = self.hardware_information.CU_SAA_leakage

        self.total_leakage = self.eDRAM_buffer_leakage + self.Router_leakage + self.SA_leakage + \
                                self.Act_leakage + self.PE_SAA_leakage + self.Pool_leakage + self.DAC_leakage+ \
                                self.MUX_leakage + self.SA_leakage + self.Crossbar_leakage + self.CU_SAA_leakage
        # Energy
        self.edram_rd_ir_energy = self.hardware_information.edram_rd_ir_energy
        self.edram_rd_pool_energy = self.hardware_information.edram_rd_pool_energy
        self.ou_operation_energy = self.hardware_information.ou_operation_energy
        self.pe_saa_energy = self.hardware_information.pe_saa_energy
        self.cu_saa_energy = self.hardware_information.cu_saa_energy
        self.activation_energy = self.hardware_information.activation_energy
        self.pooling_energy = self.hardware_information.pooling_energy
        self.edram_wr_energy = self.hardware_information.edram_wr_energy
        self.router_energy = self.hardware_information.router_energy

        self.pe_or_energy = self.hardware_information.pe_or_energy
        self.cu_ir_energy = self.hardware_information.cu_ir_energy
        self.cu_or_energy = self.hardware_information.cu_or_energy

        self.dac_energy = self.hardware_information.dac_energy
        self.xb_energy = self.hardware_information.xb_energy
        self.sa_energy = self.hardware_information.sa_energy

        self.RT_num_y = self.hardware_information.Router_num_y
        self.RT_num_x = self.hardware_information.Router_num_x
        self.PE_num_y = self.hardware_information.PE_num_y
        self.PE_num_x = self.hardware_information.PE_num_x
        self.PE_num = self.hardware_information.PE_num
        self.CU_num_y = self.hardware_information.CU_num_y
        self.CU_num_x = self.hardware_information.CU_num_x
        self.CU_num = self.hardware_information.CU_num
        self.XB_num_y = self.hardware_information.Xbar_num_y
        self.XB_num_x = self.hardware_information.Xbar_num_x
        self.XB_num = self.hardware_information.Xbar_num


        ### energy consumption
        self.edram_rd_ir_energy_total = 0
        self.edram_rd_pool_energy_total = 0
        self.ou_operation_energy_total = 0
        self.pe_saa_energy_total = 0
        self.cu_saa_energy_total = 0
        self.activation_energy_total = 0
        self.pooling_energy_total = 0
        self.edram_wr_energy_total = 0
        self.interconnect_total = 0

        self.pe_or_energy_total = 0
        self.cu_ir_energy_total = 0
        self.cu_or_energy_total = 0


        self.input_bit = self.ordergenerator.model_information.input_bit
        self.PE_array = []
        for rty_idx in range(self.RT_num_y):
            for rtx_idx in range(self.RT_num_x):
                for pey_idx in range(self.PE_num_y):
                    for pex_idx in range(self.PE_num_x):
                        pe_pos = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        pe = PE(pe_pos, self.input_bit)
                        self.PE_array.append(pe)
        #print(self.PE_array[0].CU_array[0].XB_array[0])

                
        ### Statistics
        self.color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.mem_acc_ctr = 0
        self.data_transfer_ctr = 0 
        self.act_xb_ctr = 0
        self.pe_saa_stall_cycle = 0
        
        ## utilization
        self.energy_utilization = []
        self.xbar_utilization = []
        self.pe_state_for_plot = [[],[]]
        self.buffer_size = []
        for i in range(len(self.PE_array)):
            self.buffer_size.append([])

        self.fetch_array = []

        self.interconnect = Interconnect(self.RT_num_y, self.RT_num_x)
        self.interconnect_step = 1
        self.data_transfer_trigger = []
        self.data_transfer_erp = []

        
        ### Pipeline control ###
        if not self.isPipeLine:
            print("non-pipeline")
            self.pipeline_layer_stage = 0
            self.pipeline_stage_record = []
            self.num_layer = len(self.ordergenerator.layer_list)
            print("num_layer:", self.num_layer)
            
            self.events_each_layer = []
            for layer in range(self.num_layer):
                self.events_each_layer.append(0)
            for e in self.Computation_order:
                self.events_each_layer[e.nlayer] += 1
           
            #print(self.events_each_layer)


            self.this_layer_event_ctr = 0

        print("Total event:", len(self.Computation_order))

    def run(self):
        #cu_state_for_plot = [[],[]]
        #cu_transfer_ctr = 0
        for e in self.Computation_order:
            # traverse computation order
            # if current_number_of_preceding_event==preceding_event_count的event
            # append the event to event ready pool
            if e.preceding_event_count == e.current_number_of_preceding_event:
                if e.event_type == 'edram_rd_ir':
                    pos = e.position_idx
                    rty, rtx, pey, pex, cuy, cux = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
                    idx = pex + pey * self.PE_num_x + rtx * self.PE_num + rty * rtx * self.PE_num
                    cu_idx = cux + cuy * self.CU_num_x
                    self.PE_array[idx].CU_array[cu_idx].edram_rd_ir_erp.append(e)
                else:
                    # error
                    print("Computation order error: event\"", e.event_type, "\".")
                    print("exit")
                    exit()
        
        # for pe in self.PE_array:
        #     print(pe.position, pe.edram_rd_ir_erp)

        isDone = False
        while not isDone:

            self.cycle_energy = 0
            self.cycle_ctr += 1
            self.act_xb_ctr = 0

            if self.trace:
                print('cycle:', self.cycle_ctr)

            ### Interconnect 
            for s in range(self.interconnect_step):
                self.interconnect.step()

            # Store data into buffer, trigger event
            arrived_packet = self.interconnect.get_arrived_packet()
            for pk in arrived_packet:
                print("\tarrived packet:", pk)
                rty, rtx = pk.destination[0], pk.destination[1]
                pey, pex = pk.destination[2], pk.destination[3]
                pe_idx = pex + pey * self.PE_num_x + rtx * self.PE_num + rty * self.PE_num * self.RT_num_x
                pe = self.PE_array[pe_idx]
                
                pro_event = self.Computation_order[pk.pro_event_idx]
                
                if pro_event.event_type == "edram_rd_ir":
                    # 1. store data into buffer
                    print("\twrite data into buffer:", pk.data)
                    pe.edram_buffer.put(pk.data)
                    # 2. trigger event
                    cuy, cux = pro_event.position_idx[4], pro_event.position_idx[5]

                    cu_idx = cux + cuy * self.CU_num_x
                    cu = pe.CU_array[cu_idx]
                    pro_event.current_number_of_preceding_event += 1
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                        pe.edram_rd_ir_trigger.append([pro_event, [cu_idx]])
                elif pro_event.event_type == "pe_saa":
                    # trigger event

                    pro_event.current_number_of_preceding_event += 1
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            print("\t\tProceeding event is triggered.", pro_event.event_type)

                        pe.pe_saa_trigger.append([pro_event, []])
                                
            # packet from pe to interconnect module
            for event in self.data_transfer_erp.copy():
                if self.trace:
                    print("\tdo", event.event_type, ",layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                self.data_transfer_erp.remove(event)

                src = event.position_idx[0]
                des = event.position_idx[1]
                
                #print(des_list)
                pro_event_idx = event.proceeding_event[idx]
                if self.Computation_order[pro_event_idx].event_type == "edram_rd_ir":
                    packet = Packet(src, des, [event.nlayer+1, event.outputs[0]], pro_event_idx)
                else:
                    packet = Packet(src, des, [], pro_event_idx)
                self.interconnect.input_packet(packet)
            #print(self.interconnect.packet_in_module_ctr)


            ### Fetch data from off-chip memory
            for FE in self.fetch_array.copy():
                FE.cycles_counter += 1
                #print(FE.cycles_counter, end=' ')
                if FE.cycles_counter == FE.fetch_cycle:
                    #print('@', end='/')
                    
                    if FE.event.event_type == "edram_rd_ir":
                        pe_idx = FE.index[0]
                        cu_idx = FE.index[1]
                        for inp in FE.event.inputs:
                            data = inp[1:]
                            self.PE_array[pe_idx].edram_buffer.put([FE.event.nlayer, data])

                        self.PE_array[pe_idx].CU_array[cu_idx].edram_rd_ir_erp.insert(0, FE.event)
                    
                    elif FE.event.event_type == "edram_rd_pool":
                        for data in FE.event.inputs:
                            self.PE_array[pe_idx].edram_buffer.put([FE.event.nlayer, data])
                        self.PE_array[pe_idx].edram_rd_pool_erp.insert(0, FE.event)

                    self.fetch_array.remove(FE)
                    #self.mem_acc_ctr += 1
                    #print(XB_array[0].OnchipBuffer.arr) 

            ### Event: edram_rd_ir
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    if cu.edram_rd_ir_erp:
                        event = cu.edram_rd_ir_erp[0]
                    else:
                        continue
                    #print("\tevent:", event.event_type)
                    if not cu.state and not cu.state_edram_rd_ir:

                        ## Data in eDRAM buffer?
                        isData_ready = True
                        # inputs: [[num_input, fm_h, fm_w, fm_c]]
                        for inp in event.inputs:
                            data = inp[1:]
                            #print(event.nlayer, data)
                            if not pe.edram_buffer.check([event.nlayer, data]):
                                # Data not in buffer
                                if self.trace:
                                    print("\tData not ready for edram_rd_ir. Data: layer", event.nlayer, event.event_type, data)
                                    print("\tBuffer:", pe.edram_buffer.buffer)
                                isData_ready = False
                                cu.edram_rd_ir_erp.remove(event)
                                break
                        
                        if not isData_ready:
                            self.mem_acc_ctr += 1
                            pe_idx = self.PE_array.index(pe)
                            cu_idx = pe.CU_array.index(cu)
                            #print("PE_array index", pe_idx, "data not ready")
                            cu.state_edram_rd_ir = True
                            self.fetch_array.append(FetchEvent(event, [pe_idx, cu_idx]))

                        else:
                            ## Check how many event can be done in a cycle
                            if self.trace:  
                                print("\tdo edram_rd_ir, cu_pos:", cu.position, ",order index:", self.Computation_order.index(event))
                            if not self.isPipeLine:
                                self.this_layer_event_ctr += 1
                            
                            self.edram_rd_ir_energy_total += self.edram_rd_ir_energy
                            self.cycle_energy += self.edram_rd_ir_energy

                            cu.state = True
                            cu.state_edram_rd_ir = True
                            cu.edram_rd_ir_erp.remove(event)
                            
                            ### add next event counter: ou_operation
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1
                                
                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pos = pro_event.position_idx
                                    cu_y, cu_x, xb_y, xb_x = pos[4], pos[5], pos[6], pos[7]
                                    cu_idx = cu_x + cu_y * self.CU_num_x
                                    xb_idx = xb_x + xb_y * self.XB_num_x
                                    cu.ou_operation_trigger.append([pro_event, [cu_idx, xb_idx]])                                

            ### Event: ou_operation 
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    for xb in cu.XB_array:
                        for event in xb.ou_operation_erp.copy():
                            for idx in range(len(xb.state_ou_operation)):
                                #print(xb.state_ou_operation[idx], idx)
                                if not xb.state_ou_operation[idx]:
                                    if self.trace:
                                        print("\tdo ou_operation, xb_pos:", xb.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                                    self.ou_operation_energy_total += self.ou_operation_energy
                                    self.cu_ir_energy_total += self.cu_ir_energy
                                    self.cu_or_energy_total += self.cu_or_energy

                                    self.cycle_energy += self.ou_operation_energy
                                    self.cycle_energy += self.cu_ir_energy
                                    self.cycle_energy += self.cu_or_energy

                                    self.act_xb_ctr += 1

                                    if not self.isPipeLine:
                                        self.this_layer_event_ctr += 1
                                    xb.state_ou_operation[idx] = True
                                    xb.ou_operation_erp.remove(event)

                                    ### add next event counter: cu_saa
                                    for proceeding_index in event.proceeding_event:
                                        pro_event = self.Computation_order[proceeding_index]
                                        pro_event.current_number_of_preceding_event += 1
                                        
                                        if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                            if self.trace:
                                                print("\t\tProceeding event is triggered.", pro_event.event_type)
                                            pos = pro_event.position_idx
                                            cu_y, cu_x = pos[4], pos[5]
                                            cu_idx = cu_x + cu_y * self.CU_num_x
                                            xb.cu_saa_trigger.append([pro_event, [cu_idx]])
                                    break

            ### Event: cu_saa 
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    for event in cu.cu_saa_erp.copy():
                        for idx in range(len(cu.state_cu_saa)):
                            if not cu.state_cu_saa[idx]:
                                if self.trace:
                                    print("\tdo cu_saa, cu_pos:", cu.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                                
                                self.cu_saa_energy_total += self.cu_saa_energy
                                self.cu_or_energy_total += self.cu_or_energy

                                self.cycle_energy += self.cu_saa_energy
                                self.cycle_energy += self.cu_or_energy

                                if not self.isPipeLine:
                                    self.this_layer_event_ctr += 1
                                cu.state_cu_saa[idx] = True
                                cu.cu_saa_erp.remove(event)

                                ### add next event counter: pe_saa, data_transfer
                                for proceeding_index in event.proceeding_event:
                                    pro_event = self.Computation_order[proceeding_index]
                                    pro_event.current_number_of_preceding_event += 1

                                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                        if self.trace:
                                            print("\t\tProceeding event is triggered.", pro_event.event_type)
                                        if pro_event.event_type == "pe_saa":
                                            cu.pe_saa_trigger.append([pro_event, []])

                                        elif pro_event.event_type == "edram_wr":
                                            pe.edram_wr_trigger.append([pro_event, []])
                                break

            ### Event: pe_saa 
            for pe in self.PE_array:
                for event in pe.pe_saa_erp.copy():
                    for idx in range(len(pe.state_pe_saa)):
                        if not pe.state_pe_saa[idx]:
                            if self.trace:
                                print("\tdo pe_saa, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                            
                            self.pe_saa_energy_total += self.pe_saa_energy
                            self.pe_or_energy_total += self.pe_or_energy

                            self.cycle_energy += self.pe_saa_energy
                            self.cycle_energy += self.pe_or_energy

                            if not self.isPipeLine:
                                self.this_layer_event_ctr += 1
                            pe.state_pe_saa[idx] = True
                            pe.pe_saa_erp.remove(event)

                            ### add next event counter: activation
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1
                                
                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pe.activation_trigger.append([pro_event, []])
                            break    

            ### Event: activation 
            for pe in self.PE_array:
                for event in pe.activation_erp.copy():
                    for idx in range(len(pe.state_activation)):
                        if not pe.state_activation[idx]:
                            if self.trace:
                                print("\tdo activation, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                            
                            self.activation_energy_total += self.activation_energy
                            self.cycle_energy += self.activation_energy
                            if not self.isPipeLine:
                                self.this_layer_event_ctr += 1
                            pe.state_activation[idx] = True
                            pe.activation_erp.remove(event)

                            ### add next event counter: edram_wr
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1
                                
                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx, self.Computation_order.index(pro_event))
                                    pe.edram_wr_trigger.append([pro_event, []])
                            break

            ### Event: edram write 
            for pe in self.PE_array:
                for event in pe.edram_wr_erp.copy():
                    for idx in range(len(pe.state_edram_wr)):
                        if not pe.state_edram_wr[idx]:
                            if self.trace:
                                print("\tdo edram_wr, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                            
                            self.edram_wr_energy_total += self.edram_wr_energy
                            self.cycle_energy += self.edram_wr_energy
                            if not self.isPipeLine:    
                                self.this_layer_event_ctr += 1
                            pe.state_edram_wr[idx] = True
                            pe.edram_wr_erp.remove(event)

                            ### add next event counter: edram_rd_ir, edram_rd_pool, data_transfer
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1
                            
                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pos = pro_event.position_idx
                                    if pro_event.event_type == "edram_rd_ir":
                                        cu_y, cu_x = pos[4], pos[5]
                                        cu_idx = cu_x + cu_y * self.CU_num_x
                                        pe.edram_rd_ir_trigger.append([pro_event, [cu_idx]])

                                    elif pro_event.event_type == "data_transfer":
                                        self.data_transfer_trigger.append([pro_event, []])
                            break
            
            ### Event: edram_rd_pool 
            for pe in self.PE_array:
                if pe.edram_rd_pool_erp:
                    event = pe.edram_rd_pool_erp[0]
                else:
                    continue
                if not pe.state_edram_rd_pool:
                    
                    ## Data in eDRAM buffer?
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
                        pe_idx = self.PE_array.index(pe)
                        pe.state_edram_rd_pool = True
                        self.fetch_array.append(FetchEvent(event, [pe_idx]))

                    else:
                        ## Check how many event can be done in a cycle
                        if self.trace:
                            print("\tdo edram_rd_pool, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                        self.edram_rd_pool_energy_total += self.edram_rd_pool_energy
                        self.cycle_energy += self.edram_rd_pool_energy
                        if not self.isPipeLine:
                            self.this_layer_event_ctr += 1
                        pe.state_edram_rd_pool = True
                        pe.edram_rd_pool_erp.remove(event)
                        
                        ### add next event counter: pooling
                        for proceeding_index in event.proceeding_event:
                            pro_event = self.Computation_order[proceeding_index]
                            pro_event.current_number_of_preceding_event += 1
                            
                            if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                if self.trace:
                                    print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                pos = pro_event.position_idx
                                pe.pooling_trigger.append([pro_event, []])                                
                    
            ### Event: pooling 
            for pe in self.PE_array:
                for event in pe.pooling_erp.copy():
                    for idx in range(len(pe.state_pooling)):
                        if not pe.state_pooling[idx]:
                            if self.trace:
                                print("\tdo pooling, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                            self.pooling_energy_total += self.pooling_energy
                            self.cycle_energy += self.pooling_energy
                            if not self.isPipeLine:
                                self.this_layer_event_ctr += 1
                            pe.state_pooling[idx] = True
                            pe.pooling_erp.remove(event)

                            ### add next event counter: edram_wr
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1

                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pe.edram_wr_trigger.append([pro_event, []])
                            break

            ### Trigger events ###
            ### Trigger interconnect
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
                    for trigger in cu.ou_operation_trigger.copy():
                        pro_event = trigger[0]
                        xb_idx = trigger[1][1]
                        if not self.isPipeLine:
                            if pro_event.nlayer == self.pipeline_layer_stage:
                                cu.XB_array[xb_idx].ou_operation_erp.append(pro_event)
                                cu.ou_operation_trigger.remove(trigger)
                        else:
                            cu.XB_array[xb_idx].ou_operation_erp.append(pro_event)
                            cu.ou_operation_trigger.remove(trigger)
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
                    ### Trigger cu_saa 
                    for xb in cu.XB_array:
                        for trigger in xb.cu_saa_trigger.copy():
                            pro_event = trigger[0]
                            cu_idx = trigger[1][0]
                            if not self.isPipeLine:
                                if pro_event.nlayer == self.pipeline_layer_stage:
                                    pe.CU_array[cu_idx].cu_saa_erp.append(pro_event)
                                    xb.cu_saa_trigger.remove(trigger)
                            else:
                                pe.CU_array[cu_idx].cu_saa_erp.append(pro_event)
                                xb.cu_saa_trigger.remove(trigger)

                ### Trigger pe saa (for data transfer) 
                for trigger in pe.pe_saa_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.pe_saa_erp.append(pro_event)
                            pe.pe_saa_trigger.remove(trigger)
                    else:
                        pe.pe_saa_erp.append(pro_event)
                        pe.pe_saa_trigger.remove(trigger)
                        
            ### Record PE state ###
            for pe in self.PE_array:
                if pe.check_state():
                    self.pe_state_for_plot[0].append(self.cycle_ctr)
                    self.pe_state_for_plot[1].append(self.PE_array.index(pe))

            ### Reset ###
            for pe in self.PE_array:
                pe.reset()
                for cu in pe.CU_array:
                    cu.reset()
                    for xb in cu.XB_array:
                        xb.reset()

            for pe in self.PE_array:
                for cu in pe.CU_array:
                    if cu.state:
                        if cu.cu_saa_erp:
                            break

                        isCUBusy = False
                        for xb in cu.XB_array:
                            if xb.ou_operation_erp:
                                isCUBusy = True
                                break
                        if isCUBusy:
                            break
                        cu.state = False
                        
            self.energy_utilization.append(self.cycle_energy*1e09)
            self.xbar_utilization.append(self.act_xb_ctr)
            #print(self.cycle_energy)
            
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
                    if cu.state or cu.edram_rd_ir_erp:
                        isDone = False
                        break
                    elif cu.ou_operation_trigger or cu.pe_saa_trigger:
                        isDone = False
                        break
                    for xb in cu.XB_array:
                        if xb.cu_saa_trigger:
                            isDone = False
                            break
                    if not isDone:
                        break
                if not isDone:
                    break

            # if self.cycle_ctr > 50:
            #     isDone = True

            if not self.isPipeLine:
                self.pipeline_stage_record.append(self.pipeline_layer_stage)

                #print("this_layer_event_ctr:", self.this_layer_event_ctr, self.events_each_layer, self.pipeline_layer_stage)
                if self.this_layer_event_ctr == self.events_each_layer[self.pipeline_layer_stage]:
                    #print("pipeline_layer_stage finished:", self.pipeline_layer_stage)
                    self.pipeline_layer_stage += 1
                    self.this_layer_event_ctr = 0

            # Buffer size utilization #
            for pe_idx in range(len(self.PE_array)):
                self.buffer_size[pe_idx].append(self.PE_array[pe_idx].edram_buffer.count())
            
        #print("this_layer_event_ctr:", self.this_layer_event_ctr, self.events_each_layer)
        
        ### Buffer size ###
        self.max_buffer_size = 0 # num of data
        self.total_buffer_size = 0 
        for pe in self.PE_array:
            self.total_buffer_size += len(pe.edram_buffer.buffer)
            self.max_buffer_size = max(len(pe.edram_buffer.buffer), self.max_buffer_size)
        self.avg_buffer_size = self.total_buffer_size / len(self.PE_array)

        

    def print_statistics_result(self):
        print("Cycles time:", self.cycle_time)
        print("Cycles:", self.cycle_ctr)
        
        print("Leakage:")
        print("\tTotal:", self.total_leakage)
        print("\teDRAM_buffer_leakage:", self.eDRAM_buffer_leakage)
        print("\tRouter_leakage:", self.Router_leakage)
        print("\tSA_leakage:", self.SA_leakage)
        print("\tAct_leakage:", self.Act_leakage)
        print("\tPE_SAA_leakage:", self.PE_SAA_leakage)
        print("\tPool_leakage:", self.Pool_leakage)
        print("\tDAC_leakage:", self.DAC_leakage)
        print("\tMUX_leakage:", self.MUX_leakage)
        print("\tSA_leakage:", self.SA_leakage)
        print("\tCrossbar_leakage:", self.Crossbar_leakage)
        print("\tCU_SAA_leakage:", self.CU_SAA_leakage)

        self.edram_rd_energy_total = self.edram_rd_ir_energy_total + self.edram_rd_pool_energy_total
        self.edram_energy_total = self.edram_rd_energy_total + self.edram_wr_energy_total
        

        self.dac_energy_total = self.dac_energy/self.ou_operation_energy * self.ou_operation_energy_total
        self.xb_energy_total = self.xb_energy/self.ou_operation_energy * self.ou_operation_energy_total
        self.sa_energy_total = self.sa_energy/self.ou_operation_energy * self.ou_operation_energy_total

        self.cu_energy_total = self.ou_operation_energy_total + self.cu_ir_energy_total + self.cu_or_energy_total + self.cu_saa_energy_total
        self.pe_energy_total = self.cu_energy_total + self.edram_energy_total +  + self.pe_saa_energy_total + \
                               self.activation_energy_total+ self.pooling_energy_total + self.pe_or_energy_total
        self.energy_total = self.pe_energy_total #+ self.interconnect_energy_total

        print("--Power breakdown--")

        print("\tTotal:", self.energy_total, "J")
        print("Chip level")
        print("\tPE: %.4e (%.2f%%)" %(self.pe_energy_total, self.pe_energy_total/self.energy_total*100))
        #print("\tInterconnect: %.4e (%.2f%%)" %(self.interconnect_energy_total, self.interconnect_energy_total/self.energy_total*100))
        print()
        print("PE level")
        print("\tCU: %.4e (%.2f%%)" %(self.cu_energy_total, self.cu_energy_total/self.pe_energy_total*100))
        print("\tBuffer: %.4e (%.2f%%)" %(self.edram_energy_total, self.edram_energy_total/self.pe_energy_total*100))
        print("\tShift Add: %.4e (%.2f%%)" %(self.pe_saa_energy_total, self.pe_saa_energy_total/self.pe_energy_total*100))
        print("\tActivation: %.4e (%.2f%%)" %(self.activation_energy_total, self.activation_energy_total/self.pe_energy_total*100))
        print("\tPooling: %.4e (%.2f%%)" %(self.pooling_energy_total, self.pooling_energy_total/self.pe_energy_total*100))
        print("\tOR: %.4e (%.2f%%)" %(self.pe_or_energy_total, self.pe_or_energy_total/self.pe_energy_total*100))
        print()
        print("CU level")
        print("\tDAC: %.4e (%.2f%%)" %(self.dac_energy_total, self.dac_energy_total/self.cu_energy_total*100))
        print("\tCrossbar: %.4e (%.2f%%)" %(self.xb_energy_total, self.xb_energy_total/self.cu_energy_total*100))
        print("\tSA: %.4e (%.2f%%)" %(self.sa_energy_total, self.sa_energy_total/self.cu_energy_total*100))
        print("\tShift Add: %.4e (%.2f%%)" %(self.cu_saa_energy_total, self.cu_saa_energy_total/self.cu_energy_total*100))
        print("\tIR: %.4e (%.2f%%)" %(self.cu_ir_energy_total, self.cu_ir_energy_total/self.cu_energy_total*100))
        print("\tOR: %.4e (%.2f%%)" %(self.cu_or_energy_total, self.cu_or_energy_total/self.cu_energy_total*100))
        print()

        print('memory accesss times:', self.mem_acc_ctr)
        print('max_buffer_size', self.max_buffer_size, "(", self.max_buffer_size*2, "B)")
        print("Avg buffer size:", self.avg_buffer_size)

        #print("Transfer count:", self.network_transfer.transfer_count)


        if self.isPipeLine:
            pipe_str = "pipeline"
        else:
            pipe_str = "non_pipeline"

        ### non-pipeline stage
        if not self.isPipeLine:
            with open('./statistics/non_pipeline/'+self.mapping_str+'/stage.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in range(self.cycle_ctr):
                    writer.writerow([row+1, self.pipeline_stage_record[row]])

        fre = 100
        ### Energy per 100 cycle
        with open('./statistics/'+pipe_str+'/'+self.mapping_str+'/Energy.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, self.cycle_ctr, fre):
                writer.writerow([row+1, self.energy_utilization[row]])
        
        plt.bar(range(1, self.cycle_ctr+1), self.energy_utilization)
        #plt.show()
        plt.ylabel('Energy (nJ)')
        plt.xlabel('Cycle')
        plt.ylim([0, 20])
        #plt.xlim([0,])
        plt.savefig('./statistics/'+pipe_str+'/'+self.mapping_str+'/energy_utilization.png')
        plt.clf()
        
        ### PE usage
        with open('./statistics/'+pipe_str+'/'+self.mapping_str+'/PE_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, len(self.pe_state_for_plot[0]), fre):
                writer.writerow([self.pe_state_for_plot[0][row], self.pe_state_for_plot[1][row]])

        plt.scatter(self.pe_state_for_plot[0], self.pe_state_for_plot[1])
        plt.xlabel('Cycle')
        plt.ylabel('PE number')
        plt.ylim([-1, 10])
        plt.savefig('./statistics/'+pipe_str+'/'+self.mapping_str+'/PE_utilization.png')
        plt.clf()

        ### Xbar usage
        with open('./statistics/'+pipe_str+'/'+self.mapping_str+'/XB_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, self.cycle_ctr, fre):
                writer.writerow([row+1, self.xbar_utilization[row]])
        
        plt.bar(range(1, self.cycle_ctr+1), self.xbar_utilization)
        #plt.show()
        plt.ylabel('Xbar number')
        plt.xlabel('Cycle')
        plt.ylim([0, 10]) #
        plt.savefig('./statistics/'+pipe_str+'/'+self.mapping_str+'/XB_utilization.png') 
        plt.clf()



        ### On chip Buffer

        with open('./statistics/'+pipe_str+'/'+self.mapping_str+'/OnchipBuffer.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(self.cycle_ctr):
                c = [row+1]
                for i in range(len(self.PE_array)):
                    c.append(self.buffer_size[i][row])
                writer.writerow(c)

        for i in range(len(self.PE_array)):
            plt.plot(range(1, self.cycle_ctr+1), self.buffer_size[i]) #, c=self.color[i])
        plt.xlabel('Cycle')
        plt.ylabel('Buffer size (number of data)')
        plt.ylim([0, self.max_buffer_size+5])
        plt.savefig('./statistics/'+pipe_str+'/'+self.mapping_str+'/OnChipBuffer_size_utilization.png')
        plt.clf()

        
        # plt.plot(range(1, self.cycle_ctr+1), self.buffer_size)
        # plt.xlabel("Cycle")
        # plt.ylabel("Number of data")
        # plt.ylim([0, len(self.PE_array)+100])
        # plt.savefig("./statistics/"+pipe_str+"/BufferSize.png")
        # plt.clf()
        