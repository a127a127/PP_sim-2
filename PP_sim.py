
from configs.TestModelConfig import TestModelConfig
from configs.TestModelConfig2 import TestModelConfig2
from configs.LenetConfig import LenetConfig
from configs.Cifar10Config import Cifar10Config
from configs.CaffenetConfig import CaffenetConfig

from Mapping import DefaultMapping
from Mapping import ParallelismMapping
from Mapping import TransferMapping

from OrderGenerator import OrderGenerator
from Controller import Controller

import time, sys

from HardwareMetaData import HardwareMetaData

def main():
    mapping = int(sys.argv[1])
    scheduling = int(sys.argv[2])

    ### Model ###
    model_type = 0
    print("Model type:  ", end="")
    if model_type == 0: # TestModelConfig
        print("TestModelConfig")
        model_config = TestModelConfig()
    elif model_type == 1: # TestModelConfig2
        print("TestModelConfig2")
        model_config = TestModelConfig2()
    elif model_type == 2: # Cifar10Config
        print("Cifar10Config")
        model_config = Cifar10Config()
    elif model_type == 3: # CaffenetConfig
        print("CaffenetConfig")
        model_config = CaffenetConfig()
    elif model_type == 4: # LenetConfig
        print("LenetConfig")
        model_config = LenetConfig()
    
    ### Mapping ##
    print("Mapping policy:  ", end="")
    if mapping == 0: # DefaultMapping
        print("DefaultMapping")
        mapping_information = DefaultMapping(model_config)
        mapping_str = "DefaultMapping"
    elif mapping == 1: # ParallelismMapping
        print("ParallelismMapping")
        mapping_information = ParallelismMapping(model_config)
        mapping_str = "ParallelismMapping"
    elif mapping == 2: # TransferMapping
        print("TransferMapping") 
        mapping_information = TransferMapping(model_config)
        mapping_str = "TransferMapping"
    
    ### Scheduling ###
    print("Scheduling policy: ", end="")
    if scheduling == 0: # Non-pipeline
        print("Non-pipeline")
        isPipeLine = False
    elif scheduling == 1: # Pipeline
        print("Pipeline")
        isPipeLine = True
    print()

    ### Trace ###
    isTrace_order = False
    isTrace_controller = False

    ### Generate computation order graph ### 
    start_order_time = time.time()
    print("--- Generate computation order graph ---")

    order_generator = OrderGenerator(model_config, mapping_information, isTrace_order)

    end_order_time = time.time()
    print("--- Computation order graph is generated in %s seconds ---\n" % (end_order_time - start_order_time))
    
    ## Power and performance simulation ###
    start_simulation_time = time.time()
    print("--- Power and performance simulation---")

    controller = Controller(order_generator, isPipeLine, isTrace_controller, mapping_str)
    controller.run()

    end_simulation_time = time.time()
    print("--- Simulate in %s seconds ---\n" % (end_simulation_time - start_simulation_time))

    controller.print_statistics_result()

    input_bit = model_config.input_bit
    filter_bit = model_config.filter_bit

    # layer1
    window = 1
    filter_size = 40
    filter_num = 2
    num_ou = 8
    
    router = window * input_bit * filter_size * HardwareMetaData().Energy_router
    edram_wr = window * input_bit * filter_size * HardwareMetaData().Energy_edram_buffer # off chip transfer
    edram_rd = window * filter_size * input_bit * HardwareMetaData().Energy_edram_buffer
    edram_wr += window * filter_num * input_bit * HardwareMetaData().Energy_edram_buffer
    # transfer between layer
    edram_rd += window * input_bit * filter_num * HardwareMetaData().Energy_edram_buffer 
    edram_wr += window * input_bit * filter_num * HardwareMetaData().Energy_edram_buffer
    router += window * input_bit * filter_num * HardwareMetaData().Energy_router 

    # bus
    edram_rd_bus = window * filter_size * input_bit * HardwareMetaData().Energy_bus
    pe_saa_bus = window * filter_num * input_bit * HardwareMetaData().Energy_bus * 2 * 2 # read + write
    act_bus = window * filter_num * input_bit * HardwareMetaData().Energy_bus
    edram_wr_bus = window * filter_num * input_bit * HardwareMetaData().Energy_bus
    # transfer between layer
    edram_wr_bus += window * input_bit * filter_num * HardwareMetaData().Energy_bus ## ??? 這邊不會有bus

    act = window * filter_num * HardwareMetaData().Energy_activation
    pe_saa = window * filter_num * HardwareMetaData().Energy_shift_and_add * 2 # 會分配到兩個CU
    pe_or = window * filter_num * input_bit * HardwareMetaData().Energy_or * filter_num * 2
    ou_per_filter = 1
    cu_saa = window * filter_num * input_bit * HardwareMetaData().Energy_shift_and_add * filter_bit * ou_per_filter
    adc = num_ou * HardwareMetaData().Energy_adc
    dac = num_ou * HardwareMetaData().Energy_dac
    crossbar = num_ou * HardwareMetaData().Energy_crossbar
    cu_ir = window * filter_size * input_bit * HardwareMetaData().Energy_ir_in_cu 
    ####
    cu_ir += num_ou * HardwareMetaData().Energy_ir_in_cu * HardwareMetaData().OU_h # 與ou寬無關
    cu_or = window * filter_num * input_bit * HardwareMetaData().Energy_or_in_cu * input_bit * filter_bit * 2


    # layer2
    window = 1
    filter_size = 2
    filter_num = 2
    num_ou = 2

    edram_rd += window * filter_size * input_bit * HardwareMetaData().Energy_edram_buffer
    edram_wr += window * filter_num * input_bit * HardwareMetaData().Energy_edram_buffer
    
    edram_rd_bus += window * filter_size * input_bit * HardwareMetaData().Energy_bus
    pe_saa_bus += window * filter_num * input_bit * HardwareMetaData().Energy_bus * 1 * 2 # read + write
    act_bus += window * filter_num * input_bit * HardwareMetaData().Energy_bus
    edram_wr_bus += window * filter_num * input_bit * HardwareMetaData().Energy_bus

    act += window * filter_num * HardwareMetaData().Energy_activation
    pe_saa += window * filter_num * HardwareMetaData().Energy_shift_and_add * 1 # preceding event
    pe_or += window * filter_num * input_bit * HardwareMetaData().Energy_or * 4 * 2
    ou_per_filter = 2
    cu_saa += window * filter_num * input_bit * HardwareMetaData().Energy_shift_and_add * filter_bit * ou_per_filter
    adc += num_ou * HardwareMetaData().Energy_adc
    dac += num_ou * HardwareMetaData().Energy_dac
    crossbar += num_ou * HardwareMetaData().Energy_crossbar
    cu_ir += window * filter_size * input_bit * HardwareMetaData().Energy_ir_in_cu 
    cu_ir += num_ou * HardwareMetaData().Energy_ir_in_cu * HardwareMetaData().OU_h
    cu_or += window * filter_num * input_bit * HardwareMetaData().Energy_or_in_cu * input_bit * filter_bit * 2 * 2 # 一個filter分成兩個ou

    edram = edram_wr + edram_rd
    bus = edram_rd_bus + pe_saa_bus + act_bus + edram_wr_bus

    print("edram: %.4e" %(edram))
    print("bus: %.4e" %(bus))
    print("act: %.4e" %(act))
    print("pe_saa: %.4e" %(pe_saa))
    print("pe_or: %.4e" %(pe_or))
    print("cu_saa: %.4e" %(cu_saa))
    print("adc: %.4e" %(adc))
    print("dac: %.4e" %(dac))
    print("crossbar: %.4e" %(crossbar))
    print("cu_ir: %.4e" %(cu_ir))
    print("cu_or: %.4e" %(cu_or))
    print("router: %.4e" %(router))

if __name__ == '__main__':
    main()

