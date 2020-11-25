from Mapping import LowInputReuseMapping
from Mapping import HighInputReuseMapping

from OrderGenerator import OrderGenerator
from ISAACOrderGenerator import ISAACOrderGenerator
from Controller import Controller
from ModelConfig import ModelConfig
from HardwareConfig import HardwareConfig
from Visualizer import Visualizer

import time, sys, os
import jsons, jsbeautifier
from tqdm import tqdm

def main():
    start_time     = time.time()

    model_name     = sys.argv[1]
    mapping        = sys.argv[2]
    scheduling     = sys.argv[3]
    partition_h = int(sys.argv[4])
    partition_w = int(sys.argv[5])

    model_config = ModelConfig(model_name)
    hw_config    = HardwareConfig()

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
    if model_name == "Lenet":
        CANT_USE_XB_INDEX = (0, 0, 1, 1, 3, 2)
    elif model_name == "Cifar10":
        CANT_USE_XB_INDEX = (0, 0, 0, 0, 6, 5)
    elif model_name == "DeepID":
        CANT_USE_XB_INDEX = (0, 0, 0, 0, 8, 0)
    elif model_name == "Caffenet":
        CANT_USE_XB_INDEX = (6, 1, 0, 1, 5, 2)
    elif model_name == "Overfeat":
        CANT_USE_XB_INDEX = (10, 11, 1, 0, 4, 2)
    elif model_name == "VGG16":
        CANT_USE_XB_INDEX = (0, 7, 1, 1, 11, 0)

    CANT_USE_XB_INDEX = (10000, 0, 1, 1, 3, 2)

    from Model import Model
    model = Model(model_config)
    for nlayer in range(model.layer_length):
        if model.layer_list[nlayer].layer_type == "convolution" or model.layer_list[nlayer].layer_type == "fully":
            strides = model.strides[nlayer]
            pad = model.pad[nlayer]
            o_height = model.input_h[nlayer+1]
            o_width = model.input_w[nlayer+1]

            print(f'  - {nlayer} {model.layer_list[nlayer].layer_type}: [{model.input_c[nlayer]}, {model.input_h[nlayer]}, {model.input_w[nlayer]}] x [{model.filter_n[nlayer]}, {model.filter_c[nlayer]}, {model.filter_h[nlayer]}, {model.filter_w[nlayer]}] {strides}, {pad} -> [{model.input_c[nlayer+1]}, {o_height}, {o_width}]')

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
    #order_generator = ISAACOrderGenerator(model_config, hw_config, isTrace_order)
    end_order_time = time.time()
    print("--- Computation order graph is generated in %s seconds ---\n" % (end_order_time - start_order_time))

    if False:
        json_name = f"{model_name}-{mapping}-{scheduling}-{partition_h}-{partition_w}"
        print(f"Dumping JSON to {json_name}.json...")
        with open(f"{json_name}.json", "w") as outfile:
            opts = jsbeautifier.default_options()
            opts.indent_with_tabs = True
            opts.indent_level = 1
            model_config_json = jsons.dumps(model_config)
            hw_config_json = jsons.dumps(hw_config)

            #json = jsons.dumps({
            #    "order_generator.Computation_order": order_generator.Computation_order,
            #})

            model_config_json = jsbeautifier.beautify(model_config_json, opts)
            hw_config_json = jsbeautifier.beautify(hw_config_json, opts)
            outfile.write(f'{{\n\t"model_config": {model_config_json},\n\t"hw_config": {hw_config_json},\n\t"order_generator.Computation_order": [\n')
            for index, event in enumerate(tqdm(order_generator.Computation_order)):
                outfile.write(f'\t\t// {index}:\n')
                outfile.write(f'\t\t{jsons.dumps(event)},\n')
            outfile.write(f'\t]\n}}\n')
        print(f"Done")

    #Visualizer.weightMappingByCO(hw_config, model_config, order_generator.Computation_order, f"{model_name}")
    #return
    #Visualizer.visualizeGif(hw_config, model_config, order_generator.Computation_order, f"{model_name}")
    #return
    
    log = {}

    ## Power and performance simulation ###
    start_simulation_time = time.time()
    print("--- Power and performance simulation---")
    controller = Controller(model_config, hw_config, order_generator, isTrace_controller, mapping_str, scheduling, path, log)
    end_simulation_time = time.time()
    print("--- Simulate in %s seconds ---\n" % (end_simulation_time - start_simulation_time))
    end_time = time.time()
    print("--- Run in %s seconds ---\n" % (end_time - start_time))

    Visualizer.visualizeSimulation2(hw_config, model_config, order_generator.Computation_order, log, f"{model_name}")

if __name__ == '__main__':
    main()

