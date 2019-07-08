from HardwareMetaData import HardwareMetaData
from DefaultMapping import DefaultMapping
from DefaultMapping import ParallelismMapping
from DefaultMapping import TransferMapping

from OrderGenerator import OrderGenerator
from Controller import Controller

from TestModelConfig import TestModelConfig
from MnistConfig import MnistConfig
from Cifar10Config import Cifar10Config
from CaffenetConfig import CaffenetConfig

def main():
    trace = False
    isPipeLine = True
    

    hardware_information = HardwareMetaData()

    ### Model ###
    #model_information = TestModelConfig()
    #model_information = MnistConfig()
    model_information = Cifar10Config()
    #model_information = CaffenetConfig()
    
    ### Mapping ##
    mapping_type = 1
    if mapping_type == 1:
        mapping_information = DefaultMapping(hardware_information, model_information)
        print("DefaultMapping")
    elif mapping_type == 2:
        mapping_information = ParallelismMapping(hardware_information, model_information)
        print("ParallelismMapping")
    elif mapping_type == 3:
        mapping_information = TransferMapping(hardware_information, model_information)
        print("TransferMapping")

    print("Mapping finish")
    order_generator = OrderGenerator(model_information, hardware_information, mapping_information)
    trace_order = False
    data_transfer_ctr = 0
    for e in order_generator.Computation_order:
        if e.event_type == "data_transfer":
            data_transfer_ctr += 1
    if trace_order:
        i = 0
        for e in order_generator.Computation_order:
            if e.nlayer == 1:
                print(i, e)
            # if e.event_type == "edram_wr" or e.event_type == "edram_rd_ir" or e.event_type == "edram_rd_pool" or e.event_type == "data_transfer":
            #     pass
            # 
            # print()
            i += 1
    print("data_transfer_ctr", data_transfer_ctr)
    controller = Controller(order_generator, isPipeLine, trace)
    controller.run()

if __name__ == '__main__':
    main()
