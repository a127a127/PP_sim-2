from ModelConfig import ModelConfig
from Model import Model
from Mapping import SameColumnFirstMapping
from HardwareConfig import HardwareConfig

model_type_list = ["Lenet", "Cifar10", "DeepID", "Caffenet", "Overfeat"]
result = []

for model_type in model_type_list:
    model_config = ModelConfig(model_type)
    model_info = Model(model_config)
    hw_config = HardwareConfig(model_config)
    mapping_info = SameColumnFirstMapping(model_config, hw_config)
    throughput_list = []
    for nlayer in range(model_info.layer_length):
        layer_type = model_config.layer_list[nlayer].layer_type
        if layer_type == "convolution":
            print("conv", nlayer)
            fm_h, fm_w = model_info.input_h[nlayer+1], model_info.input_w[nlayer+1]
            f_n, f_h, f_w, f_c = model_info.filter_n[nlayer], model_info.filter_h[nlayer], model_info.filter_w[nlayer], model_info.filter_c[nlayer]
            used_xb = mapping_info.layer_used_xb[nlayer]

            # print(f_n, f_h, f_w, f_c)
            # print(used_xb)

            throughput = fm_h * fm_w * f_n * f_h * f_w * f_c // used_xb
            throughput_list.append(throughput)
    result.append(throughput_list)
for n in range(len(model_type_list)): 
    print(model_type_list[n], result[n])
for n in range(len(model_type_list)):
    throughput_list = result[n]
    last = throughput_list[-1]
    for i in range(len(throughput_list)):
        throughput_list[i] = throughput_list[i] // last

for n in range(len(model_type_list)): 
    print(model_type_list[n], result[n])