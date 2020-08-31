from Mapping import LowInputReuseMapping
from Mapping import HighInputReuseMapping

from OrderGenerator import OrderGenerator
from Controller import Controller
from ModelConfig import ModelConfig
from HardwareConfig import HardwareConfig

import time, sys, os

def main():
    start_time     = time.time()

    model_name     = sys.argv[1]
    mapping        = sys.argv[2]
    scheduling     = sys.argv[3]
    partition_h = int(sys.argv[4])
    partition_w = int(sys.argv[5])

    model_config = ModelConfig(model_name)
    hw_config    = HardwareConfig(model_config)

    ### Mapping ##
    start_mapping_time = time.time()
    print("--- Mapping ---")
    print("Mapping policy:  ", end="")

    # Lenet:    (0, 0, 1, 1, 3, 2)
    # Cifar10:  (0, 0, 0, 0, 6, 5)
    # DeepID:   (0, 0, 0, 0, 8, 0)
    # Caffenet: (6, 1, 0, 1, 5, 2)
    # Overfeat: (10, 11, 1, 0, 4, 2)
    # VGG16:    (0, 7, 1, 1, 11, 0)
    CANT_USE_XB_INDEX = (0, 0, 0, 0, 0, 0)

    if mapping == "LIR":
        print("Low Input data Reuse Mapping")
        mapping_information = LowInputReuseMapping(model_config, hw_config, partition_h, partition_w, CANT_USE_XB_INDEX)
        mapping_str = "Low_Input_data_Reuse_Mapping_"+sys.argv[4]+"_"+sys.argv[5]
    elif mapping == "HIR":
        print("High Input data Reuse Mapping")
        mapping_information = HighInputReuseMapping(model_config, hw_config, partition_h, partition_w, CANT_USE_XB_INDEX)
        mapping_str = "High_Input_data_Reuse_Mapping_"+sys.argv[4]+"_"+sys.argv[5]
    elif mapping == "Count":
        from Mapping import CaculateMappedCrossbarNum
        mapping_information = CaculateMappedCrossbarNum(model_config, hw_config, 1, 1)
        exit()
    else:
        print("Wrong mapping parameter")
        exit()
    
    end_mapping_time = time.time()
    print("--- Mapping: finished in %s seconds ---\n" % (end_mapping_time - start_mapping_time))
    
    ### Scheduling ###
    print("Scheduling policy: ", end="")
    if scheduling == "Non-pipeline":
        print("Non-pipeline")
    elif scheduling == "Pipeline":
        print("Pipeline")
    else:
        print("Wrong scheduling parameter")
        exit()
    print()

    ### Buffer Replacement ###
    print("Buffer replacement policy: ", end="")
    replacement = "LRU"
    if replacement == "Ideal":
        print("Ideal")
    elif replacement == "LRU":
        print("LRU")

    ### path ###
    path = './statistics/'+model_config.Model_type+'/'+mapping_str+'/'+scheduling
    if not os.path.exists(path):
            os.makedirs(path)
    
    mapping_information.mapping_layout(path)

    ### Trace ###
    isTrace_order      = False
    isTrace_controller = False   

    ### Generate computation order graph ### 
    start_order_time = time.time()
    print("--- Generate computation order ---")
    order_generator = OrderGenerator(model_config, hw_config, mapping_information, isTrace_order)
    end_order_time = time.time()
    print("--- Computation order graph is generated in %s seconds ---\n" % (end_order_time - start_order_time))
    
    ## Power and performance simulation ###
    start_simulation_time = time.time()
    print("--- Power and performance simulation---")
    controller = Controller(model_config, hw_config, order_generator, isTrace_controller, mapping_str, scheduling, replacement, path)
    end_simulation_time = time.time()
    print("--- Simulate in %s seconds ---\n" % (end_simulation_time - start_simulation_time))
    end_time = time.time()
    print("--- Run in %s seconds ---\n" % (end_time - start_time))

if __name__ == '__main__':
    main()

