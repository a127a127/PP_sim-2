from HardwareMetaData import HardwareMetaData
from DefaultMapping import DefaultMapping
from OrderGenerator import OrderGenerator
from Controller import Controller

from TestModelConfig import TestModelConfig
from MnistConfig import MnistConfig

def main():
    trace = True
    isPipeLine = False

    hardware_information = HardwareMetaData()
    model_information = TestModelConfig()
    #model_information = MnistConfig()
    mapping_information = DefaultMapping(hardware_information, model_information)
    #print(mapping_information.layer_mapping_to_xbar[0][0][0][0][0][0][0][0][0])

    order_generator = OrderGenerator(model_information, hardware_information, mapping_information)
    
    if False:
        i = 0
        for e in order_generator.Computation_order:
            if e.event_type == "edram_wr" or e.event_type == "edram_rd_ir" or e.event_type == "edram_rd_pool" or e.event_type == "data_transfer":
                pass
            print(i, e)
            print()
            
            i += 1
        

    controller = Controller(order_generator, isPipeLine, trace)
    controller.run()

if __name__ == '__main__':
    main()
