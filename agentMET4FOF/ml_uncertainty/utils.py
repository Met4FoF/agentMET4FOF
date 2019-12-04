import torch

def calc_layer_size(input_size,layer_string):
    operator = layer_string[0]
    magnitude = float(layer_string[1:])
    if operator.lower() == "d":
        output_size = int(input_size/magnitude)
    elif operator.lower() == "x":
        output_size = int(input_size*magnitude)
    return output_size

def parse_architecture_string(input_size,output_size, architecture, layer_type=torch.nn.Linear):
    layers = []
    for layer_index,layer_string in enumerate(architecture):
        if layer_index ==0:
            first_layer = layer_type(input_size, calc_layer_size(input_size, layer_string))
            layers.append(first_layer)
            #special case if there's only a single hidden layer
            if len(architecture) == 1:
                last_layer = layer_type(calc_layer_size(input_size, layer_string), output_size)
                layers.append(last_layer)
        elif layer_index == (len(architecture)-1):
            new_layer = layer_type(calc_layer_size(input_size, architecture[layer_index-1]),
                                            calc_layer_size(input_size, layer_string))
            last_layer = layer_type(calc_layer_size(input_size, layer_string), output_size)
            layers.append(new_layer)
            layers.append(last_layer)
        else:
            new_layer = layer_type(calc_layer_size(input_size, architecture[layer_index-1]),
                                            calc_layer_size(input_size, layer_string))
            layers.append(new_layer)
    return layers
