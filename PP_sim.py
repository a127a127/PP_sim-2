from HardwareMetaData import HardwareMetaData
from DefaultMapping import DefaultMapping
from OrderGenerator import OrderGenerator
from Controller import Controller

from TestModelConfig import TestModelConfig
from MnistConfig import MnistConfig

def main():
    trace = False
    isPipeLine = True

    hardware_information = HardwareMetaData()
    model_information = TestModelConfig()
    #model_information = MnistConfig()
    mapping_information = DefaultMapping(hardware_information, model_information)
    #print(mapping_information.layer_mapping_to_xbar[0][0][0][0][0][0][0][0][0])

    order_generator = OrderGenerator(model_information, hardware_information, mapping_information, isPipeLine)
    
   
    i = 0
    for e in order_generator.Computation_order:
        print(i, e)
        print()
        i += 1
    

    #controller = Controller(order_generator, trace)
    #controller.run()

if __name__ == '__main__':
    main()
