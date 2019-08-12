from HardwareMetaData import HardwareMetaData

from TestModelConfig import TestModelConfig
from TestModelConfig2 import TestModelConfig2
from LenetConfig import LenetConfig
from Cifar10Config import Cifar10Config

from CaffenetConfig import CaffenetConfig
from DefaultMapping import DefaultMapping
from DefaultMapping import ParallelismMapping
from DefaultMapping import TransferMapping

from OrderGenerator import OrderGenerator
from Controller import Controller

import time, sys

def main():
    mapping = int(sys.argv[1])
    scheduling = int(sys.argv[2])

    ### Hardware configuration ###
    hardware_information = HardwareMetaData()

    ### Model ###
    model_type = 4
    print("Model type:  ", end="")
    if model_type == 0: #TestModelConfig
        print("TestModelConfig")
        model_information = TestModelConfig()
    elif model_type == 1: #LenetConfig
        print("LenetConfig")
        model_information = LenetConfig()
    elif model_type == 2: #Cifar10Config
        print("Cifar10Config")
        model_information = Cifar10Config()
    elif model_type == 3: #CaffenetConfig
        print("CaffenetConfig")
        model_information = CaffenetConfig()
    elif model_type == 4: #TestModelConfig2
        print("TestModelConfig2")
        model_information = TestModelConfig2()
    
    ### Mapping ##
    print("Mapping policy:  ", end="")
    if mapping == 0: #DefaultMapping
        print("DefaultMapping")
        mapping_information = DefaultMapping(hardware_information, model_information)
        mapping_str = "DefaultMapping"
    elif mapping == 1: #ParallelismMapping
        print("ParallelismMapping")
        mapping_information = ParallelismMapping(hardware_information, model_information)
        mapping_str = "ParallelismMapping"
    elif mapping == 2: #TransferMapping
        print("TransferMapping") 
        mapping_information = TransferMapping(hardware_information, model_information)
        mapping_str = "TransferMapping"
    
    ### Scheduling ###
    print("Scheduling policy: ", end="")
    if scheduling == 0:
        print("Non-pipeline")
        isPipeLine = False
    elif scheduling == 1:
        print("Pipeline")
        isPipeLine = True

    ### Trace ###
    isTrace_order = False
    isTrace_controller = True

    ### Generate computation order graph ### 
    start_order_time = time.time()
    print("Generate computation order graph...")

    order_generator = OrderGenerator(model_information, hardware_information, mapping_information)

    end_order_time = time.time()
    print("--- Computation order graph is generated in %s seconds ---" % (end_order_time - start_order_time))
    if isTrace_order:
        trace_order(order_generator)
    
    ## Power and performance simulation ###
    start_simulation_time = time.time()

    controller = Controller(order_generator, isPipeLine, isTrace_controller, mapping_str)
    controller.run()

    end_simulation_time = time.time()
    print("--- Simulation in %s seconds ---" % (end_simulation_time - start_simulation_time))

    controller.print_statistics_result()

def trace_order(order_generator):
    edram_rd_ir_ctr = 0
    ou_operation_ctr = 0
    cu_saa_ctr = 0
    pe_saa_ctr = 0
    activation_ctr = 0
    pooling_ctr = 0
    edram_wr_ctr = 0
    edram_rd_pool_ctr = 0
    data_transfer_ctr = 0
    for e in order_generator.Computation_order: 
        # edram_rd_ir, ou_operation, cu_saa, pe_saa, activation, edram_wr, edram_rd_pool, data_transfer
        t = e.event_type
        if t == "edram_rd_ir":
            edram_rd_ir_ctr += 1
        elif t == "ou_operation":
            ou_operation_ctr += 1
        elif t == "cu_saa":
            cu_saa_ctr += 1
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
            print("event type error..")

    print("edram_rd_ir_ctr", edram_rd_ir_ctr)
    print("ou_operation_ctr", ou_operation_ctr)
    print("cu_saa_ctr", cu_saa_ctr)
    print("pe_saa_ctr", pe_saa_ctr)
    print("activation_ctr", activation_ctr)
    print("edram_wr_ctr", edram_wr_ctr)
    print("edram_rd_pool_ctr", edram_rd_pool_ctr)
    print("data_transfer_ctr", data_transfer_ctr)

    for e in order_generator.Computation_order:
        #if 201 in e.proceeding_event:
        if True:
            print(order_generator.Computation_order.index(e), e)
            print()

if __name__ == '__main__':
    main()

