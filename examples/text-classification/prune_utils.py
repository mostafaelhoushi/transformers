import torch
import collections

def get_params_of(model, layer_class, param_name):
    params = []
    for layer_name, module in model._modules.items():
        if isinstance(module, layer_class):
            params.append(module.weight)
        
        if len(list(module.children())) > 0:
            # recurse
            params.extend(get_params_of(module, layer_class, param_name))

    return params

def prune_by_weightnorm(model, count):
    layer2weightnorm = {}
    for index, layer in enumerate(model.bert.encoder.layer):
        weights = get_params_of(layer, torch.nn.Linear, "weight")
        weightnorm = torch.stack([weight.flatten().norm() for weight in weights]).mean()
        layer2weightnorm[index] = weightnorm

    sorted_layer2weightnorm = sorted(layer2weightnorm, key=layer2weightnorm.get)
    selected_layers = sorted_layer2weightnorm[:count]
    selected_layers.sort(reverse=True)

    for layer in selected_layers:
        del model.bert.encoder.layer[layer] 

    return model


    