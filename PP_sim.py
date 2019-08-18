
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

def main():
    mapping = int(sys.argv[1])
    scheduling = int(sys.argv[2])

    ### Model ###
    model_type = 1
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
    isTrace_controller = True

    ### Generate computation order graph ### 
    start_order_time = time.time()
    print("--- Generate computation order graph ---")

    order_generator = OrderGenerator(model_config, mapping_information)

    end_order_time = time.time()
    print("--- Computation order graph is generated in %s seconds ---\n" % (end_order_time - start_order_time))
    if isTrace_order:
        trace_order(order_generator)
    
    ## Power and performance simulation ###
    start_simulation_time = time.time()
    print("--- Simulation power and performance ---")

    controller = Controller(order_generator, isPipeLine, isTrace_controller, mapping_str)
    controller.run()

    end_simulation_time = time.time()
    print("--- Simulation in %s seconds ---\n" % (end_simulation_time - start_simulation_time))

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

