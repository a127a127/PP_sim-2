#from Mapping import DefaultMapping
#from Mapping import HighParallelismMapping
from Mapping import SameColumnFirstMapping
from Mapping import SameRowFirstMapping
from Mapping import ParallelsimMapping
from OrderGenerator import OrderGenerator
from Controller import Controller

import time, sys, os
#import pickle

def main():
    start_time = time.time()
    mapping = sys.argv[1]
    scheduling = sys.argv[2]

    ### Mapping ##
    start_mapping_time = time.time()
    print("--- Mapping ---")
    print("Mapping policy:  ", end="")
    if mapping == 0:
        print("Default Mapping")
        exit()
        #mapping_information = DefaultMapping()
        #mapping_str = "Default_Mapping"
    elif mapping == 1:
        print("High Parallelism Mapping")
        exit()
        #mapping_information = HighParallelismMapping()
        #mapping_str = "High_Parallelism_Mapping"
    elif mapping == "SCF":
        print("Same Column First Mapping")
        mapping_information = SameColumnFirstMapping()
        mapping_str = "Same_Column_First_Mapping"
    elif mapping == "SRF":
        print("Same Row First Mapping")
        mapping_information = SameRowFirstMapping()
        mapping_str = "Same_Row_First_Mapping"
    elif mapping == "Paral":
        print("Parallelsim Mapping"+sys.argv[3])
        mapping_information = ParallelsimMapping(int(sys.argv[3]))
        mapping_str = "Parallelsim_Mapping"+sys.argv[3]
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
    replacement = "Ideal"
    if replacement == "Ideal":
        print("Ideal")
    elif replacement == "LRU":
        print("LRU")


    ### dir ###
    if not os.path.exists('./statistics/'+mapping_str+'/'+scheduling):
            os.makedirs('./statistics/'+mapping_str+'/'+scheduling)
    
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
    controller = Controller(order_generator, isTrace_controller, mapping_str, scheduling, replacement)
    controller.run()

    end_simulation_time = time.time()
    print("--- Simulate in %s seconds ---\n" % (end_simulation_time - start_simulation_time))
    end_time = time.time()
    print("--- Run in %s seconds ---\n" % (end_time - start_time))
    controller.print_statistics_result()

if __name__ == '__main__':
    main()

