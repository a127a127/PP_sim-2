from Mapping_o import SCF
from Mapping_o import SRF
from OrderGenerator import OrderGenerator
from Controller import Controller
from ModelConfig import ModelConfig
from HardwareConfig import HardwareConfig

import time, sys, os

def main():
    start_time = time.time()
    model      = sys.argv[1]
    mapping    = sys.argv[2]
    scheduling = sys.argv[3]
    partition_h = int(sys.argv[4])
    partition_w = int(sys.argv[5])

    model_config = ModelConfig(model)
    hw_config = HardwareConfig()

    ### Mapping ##
    cant_use_pe = (13, 12, 1, 1)
    # Used PE: Lenet:6, Cifar10: 5, DeepID: 6, Caffenet: 321, Overfeat: 568, VGG16: 708
    if model == "Lenet":
        cant_use_pe = (0, 1, 1, 0)
    elif model == "Cifar10":
        cant_use_pe = (0, 1, 0, 1)
    elif model == "DeepID":
        cant_use_pe = (0, 1, 1, 0)
    elif model == "Caffenet":
        cant_use_pe = (6, 2, 0, 1)
    elif model == "Overfeat":
        cant_use_pe = (10, 12, 0, 0)
    elif model == "VGG16":
        cant_use_pe = (13, 4, 0, 0)

    start_mapping_time = time.time()
    print("--- Mapping ---")
    print("Mapping policy:  ", end="")
    if mapping == "SCF":
        print("Same Column First Mapping")
        mapping_information = SCF(model_config, hw_config, partition_h, partition_w, cant_use_pe)
        mapping_str = "Same_Column_First_Mapping"+sys.argv[4]+"_"+sys.argv[5]
    elif mapping == "SRF":
        print("Same Row First Mapping")
        mapping_information = SRF(model_config, hw_config, partition_h, partition_w, cant_use_pe)
        mapping_str = "Same_Row_First_Mapping"+sys.argv[4]+"_"+sys.argv[5]

    end_mapping_time = time.time()
    print("--- Mapping is finished in %s seconds ---\n" % (end_mapping_time - start_mapping_time))
    
    ### Scheduling ###
    print("Scheduling policy: ", end="")
    if scheduling == "Non-pipeline":
        print("Non-pipeline")
    elif scheduling == "Pipeline":
        print("Pipeline")
    else:
        print("Wrong scheduling type")
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
    
    # mapping_information.mapping_layout(path)

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

