from Mapping import SameColumnFirstMapping
from Mapping import SameRowFirstMapping
from Mapping import SCFParallelsimMapping
from Mapping import SRFParallelsimMapping

from OrderGenerator import OrderGenerator
from Controller import Controller
from ModelConfig import ModelConfig

import time, sys, os

def main():
    start_time = time.time()
    mapping = sys.argv[1]
    scheduling = sys.argv[2]

    ### Mapping ##
    start_mapping_time = time.time()
    print("--- Mapping ---")
    print("Mapping policy:  ", end="")
    if mapping == "SCF":
        print("Same Column First Mapping")
        mapping_information = SameColumnFirstMapping()
        mapping_str = "Same_Column_First_Mapping"
    elif mapping == "SRF":
        print("Same Row First Mapping")
        mapping_information = SameRowFirstMapping()
        mapping_str = "Same_Row_First_Mapping"
    elif mapping == "SCFParal":
        print("'SCF Parallelsim Mapping"+sys.argv[3])
        mapping_information = SCFParallelsimMapping(int(sys.argv[3]))
        mapping_str = "SCFParallelsim_Mapping"+sys.argv[3]
    elif mapping == "SRFParal":
        print("SRF Parallelsim Mapping"+sys.argv[3])
        mapping_information = SRFParallelsimMapping(int(sys.argv[3]))
        mapping_str = "SRFParallelsim_Mapping"+sys.argv[3]
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

    ### dir ###
    path = './statistics/'+ModelConfig().Model_type+'/'+mapping_str+'/'+scheduling
    if not os.path.exists(path):
            os.makedirs(path)
    
    ### Trace ###
    isTrace_order      = True
    isTrace_controller = False   

    ### Generate computation order graph ### 
    start_order_time = time.time()
    print("--- Generate computation order graph ---")
    order_generator = OrderGenerator(mapping_information, isTrace_order)
    end_order_time = time.time()
    print("--- Computation order graph is generated in %s seconds ---\n" % (end_order_time - start_order_time))
    
    ## Power and performance simulation ###
    start_simulation_time = time.time()
    print("--- Power and performance simulation---")
    controller = Controller(order_generator, isTrace_controller, mapping_str, scheduling, replacement, path)
    end_simulation_time = time.time()
    print("--- Simulate in %s seconds ---\n" % (end_simulation_time - start_simulation_time))
    end_time = time.time()
    print("--- Run in %s seconds ---\n" % (end_time - start_time))

if __name__ == '__main__':
    main()

