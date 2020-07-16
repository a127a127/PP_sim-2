from Mapping import SameColumnFirstMapping
from Mapping import SameRowFirstMapping
from Mapping import SCFParallelsimMapping
from Mapping import SRFParallelsimMapping

from OrderGenerator import OrderGenerator
from ISAACOrderGenerator import ISAACOrderGenerator
from Controller import Controller
from ModelConfig import ModelConfig
from HardwareConfig import HardwareConfig

import time, sys, os

def main():
    start_time = time.time()
    model      = sys.argv[1]
    mapping    = sys.argv[2]
    scheduling = sys.argv[3]

    model_config = ModelConfig(model)
    hw_config = HardwareConfig(model_config)

    if mapping == "ISAAC":
        print("--- ISAAC Order Generator ---")

        mapping_str = "ISAAC"
    else:
        ### Mapping ##
        start_mapping_time = time.time()
        print("--- Mapping ---")
        print("Mapping policy:  ", end="")
        if mapping == "SCF":
            print("Same Column First Mapping")
            mapping_information = SameColumnFirstMapping(model_config, hw_config)
            mapping_str = "Same_Column_First_Mapping"
        elif mapping == "SRF":
            print("Same Row First Mapping")
            mapping_information = SameRowFirstMapping(model_config, hw_config)
            mapping_str = "Same_Row_First_Mapping"
        elif mapping == "SCFParal":
            print("'SCF Parallelsim Mapping"+sys.argv[4])
            paral = int(sys.argv[4])
            mapping_information = SCFParallelsimMapping(model_config, hw_config, paral)
            mapping_str = "SCFParallelsim_Mapping"+sys.argv[4]
        elif mapping == "SRFParal":
            print("SRF Parallelsim Mapping"+sys.argv[4])
            paral = int(sys.argv[4])
            mapping_information = SRFParallelsimMapping(model_config, hw_config, paral)
            mapping_str = "SRFParallelsim_Mapping"+sys.argv[4]
        else:
            print("Wrong mapping type")
            exit()
        end_mapping_time = time.time()
        print("--- Mapping is finished in %s seconds ---\n" % (end_mapping_time - start_mapping_time))

        ### Scheduling ###
        print("Scheduling policy: ", end="")
        if scheduling == "Non_pipeline":
            print("Non-pipeline")
        elif scheduling == "Pipeline":
            print("Pipeline")
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
    
    if mapping != "ISAAC":
        mapping_information.mapping_layout(path)

    ### Trace ###
    isTrace_order      = True
    isTrace_controller = False   

    ### Generate computation order graph ### 
    start_order_time = time.time()
    print("--- Generate computation order ---")
    if mapping == "ISAAC":
        order_generator = ISAACOrderGenerator(model_config, hw_config, isTrace_order)
    else:
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

