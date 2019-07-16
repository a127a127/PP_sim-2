from HardwareMetaData import HardwareMetaData
from DefaultMapping import DefaultMapping
from DefaultMapping import ParallelismMapping
from DefaultMapping import TransferMapping

from OrderGenerator import OrderGenerator
from Controller import Controller

from TestModelConfig import TestModelConfig
from LenetConfig import LenetConfig
from Cifar10Config import Cifar10Config
from CaffenetConfig import CaffenetConfig

import time, sys

def main():
    mapping = int(sys.argv[1])

    ### Hardware configuration ###
    hardware_information = HardwareMetaData()

    ### Model ###
    model_type = 0
    print("Model type:  ", end="")
    if model_type == 0:
        model_information = TestModelConfig()
        print("TestModelConfig")
    elif model_type == 1:
        model_information = LenetConfig()
        print("LenetConfig")
    elif model_type == 2:
        model_information = Cifar10Config()
        print("Cifar10Config")
    elif model_type == 3:
        model_information = CaffenetConfig()
        print("CaffenetConfig")
    
    ### Mapping ##
    mapping_type = mapping
    print("Mapping policy:  ", end="")
    if mapping_type == 0:
        mapping_information = DefaultMapping(hardware_information, model_information)
        print("DefaultMapping")
    elif mapping_type == 1:
        mapping_information = ParallelismMapping(hardware_information, model_information)
        print("ParallelismMapping")
    elif mapping_type == 2:
        mapping_information = TransferMapping(hardware_information, model_information)
        print("TransferMapping")
    

    ### Scheduling ###
    isPipeLine = False

    ### Trace ###
    istrace = False
    isStatistic_order = True

    ### Generate computation order graph ### 
    start_order_time = time.time()
    print("Generate computation order graph...")
    order_generator = OrderGenerator(model_information, hardware_information, mapping_information)
    end_order_time = time.time()
    print("--- Computation order graph is generated in %s seconds ---" % (end_order_time - start_order_time))
    if isStatistic_order:
        statistic_order(order_generator)
    
    ### Power and performance simulation ###
    start_simulation_time = time.time()
    controller = Controller(order_generator, isPipeLine, istrace)
    controller.run()
    end_simulation_time = time.time()
    print("--- Simulation in %s seconds ---" % (end_simulation_time - start_simulation_time))

    
def statistic_order(order_generator):
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
            print("???")

    print("edram_rd_ir_ctr", edram_rd_ir_ctr)
    print("ou_operation_ctr", ou_operation_ctr)
    print("cu_saa_ctr", cu_saa_ctr)
    print("pe_saa_ctr", pe_saa_ctr)
    print("activation_ctr", activation_ctr)
    print("edram_wr_ctr", edram_wr_ctr)
    print("edram_rd_pool_ctr", edram_rd_pool_ctr)
    print("data_transfer_ctr", data_transfer_ctr)

    # i = 0
    # for e in order_generator.Computation_order:
    #     if e.nlayer == 1:
    #         print(i, e)
    #     # if e.event_type == "edram_wr" or e.event_type == "edram_rd_ir" or e.event_type == "edram_rd_pool" or e.event_type == "data_transfer":
    #     #     pass
    #     # 
    #     # print()
    #     i += 1
    # print("data_transfer_ctr", data_transfer_ctr)


if __name__ == '__main__':
    main()


