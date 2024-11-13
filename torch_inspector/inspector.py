from typing import Union, Optional, List, Literal
import psutil
import time
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

class LayerContainer:

    def __init__(self,
                 layer: nn.Module,
                 model_rel_path: List[str] = None
                 ):

        self.layer = layer
        self.layer_type = layer.__class__.__name__
        self.model_rel_path = model_rel_path
        ##-- Layer Information --##
        self.num_parameters = sum(p.numel() for p in layer.parameters())
        self.num_learnable_parameters = sum(p.numel() for p in layer.parameters() if p.requires_grad)

        self.params_size = sum(p.numel()*p.element_size() for p in layer.parameters())
        self.gradients_size = sum(p.numel()*p.element_size() for p in layer.parameters() if p.requires_grad)
        self.effective_additional_memory_fwd = 0. # bytes 
        self.effective_additional_memory_bwd = 0. # bytes

        self.forward_nan = False
        self.grad_in_nan = False
        self.grad_out_nan = False

        self.input_shapes = []
        self.output_shapes = []
        self.grad_in_shape = []
        self.grad_out_shape = []
        self.gpu_memory_usage = 0

        self.input_size = []
        self.output_size = []

        self.forward_latency = 0.
        self.backward_latency = 0.

    def set_input_shape(self, shape: Union[tuple, List[int]], dtype=torch.float32):
        if isinstance(shape, torch.Size):
            shape = [list(shape)]
        self.input_shapes = shape
        self.input_size = []
        elem_size = torch.tensor([], dtype=dtype).element_size()
        for in_shape in self.input_shapes:
            self.input_size.append((elem_size * torch.prod(torch.tensor(in_shape))).item())


    def set_output_shape(self, shape: Union[tuple, List[int]], dtype=torch.float32):
        if isinstance(shape, torch.Size):
            shape = [list(shape)]
        self.output_shapes = shape
        elem_size = torch.tensor([], dtype=dtype).element_size()
        self.output_size = []
        for out_shape in self.output_shapes:
            self.output_size.append((elem_size * torch.prod(torch.tensor(out_shape))).item())

    def to_dict(self) -> dict:
        """
            All memory sizes are returned in Mb.
        """
        return {
            'type': self.layer_type,
            'input_shapes': self.input_shapes,
            'output_shapes': self.output_shapes,
            'grad_in_shape': self.grad_in_shape,
            'grad_out_shape': self.grad_out_shape,
            'num_parameters': self.num_parameters,
            'num_learnable_parameters': self.num_learnable_parameters,
            'param_memory_usage': self.params_size  / (10**6),
            'gradient_memory_usage': self.gradients_size / (10**6),
            'input_memory_usage': [mem/10**6 for mem in self.input_size],
            'output_memory_usage': [mem/10**6 for mem in self.output_size],
            'effective_additional_memory_fwd': self.effective_additional_memory_fwd / (10**6),
            'effective_additional_memory_bwd': self.effective_additional_memory_bwd / (10**6),
            'forward_nan': self.forward_nan,
            'grad_in_nan': self.grad_in_nan,
            'grad_out_nan': self.grad_out_nan,
            'total_memory': (self.params_size+self.gradients_size+sum(self.input_size)) / (10**6),
            'forward_latency': self.forward_latency,
            'backward_latency': self.backward_latency,

        }


class InspectorGadgets:

    def __init__(self, 
                  model: Union[nn.Module, List[nn.Module]],
                  watch_forward: Optional[bool] = True,
                  watch_backward: Optional[bool] = True,
                  watch_depth: int = -1,
                  save_file: str = None
                ):
        if isinstance(model, nn.Module):
            self.models = [model]
        else:
            self.models = model

        devices = [next(m.parameters()).device for m in self.models]
        self.device = devices[0]
        self.save_file = save_file

        self.watch_forward = watch_forward
        self.watch_backward = watch_backward
        self.watch_depth = watch_depth

        self._curr_forward_memory_usage = None
        self._init_forward_memory_usage = None

        self._total_forward_memory_usage = 0.
        self._max_memory_usage = 0.

        self._cur_time = time.time()
        self._init_time = time.time()

        self._cur_mode: Literal['init', 'fwd', 'bwd'] = 'init'

        self.forward_hooks = []
        self.backward_hooks = []

    def start_timer(self):
        self._cur_time = time.time()
        self._init_time = time.time()

    def start(self):
        self.initialize_hooks()
        self._curr_gpu_memory_usage = self._get_memory_usage()
        self._init_gpu_memory_usage = self._get_memory_usage()
        self.start_timer()

    def finish(self):
        self.remove_backward_hooks()
        self.remove_forward_hooks()

        self._integrate_upwards()

        if self.save_file is not None:
            self._save(self.save_file)

    def __enter__(self):
        self.start()
        self.initialize_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove hooks
        self.finish()
    
    def remove_forward_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        self.forward_hooks = []
    
    def remove_backward_hooks(self):
        for hook in self.backward_hooks:
            hook.remove()
        self.backward_hooks = []

    def _hook_wrapper(self, hook_type: Literal["forward", "backward"], layer_container: LayerContainer):
        @torch.no_grad()
        def _forward_hook(module, input, output):
            layer_container.set_input_shape(input[0].cpu().detach().shape)
            layer_container.set_output_shape(output.cpu().detach().shape)

            curr_gpu_memory_usage = self._get_memory_usage()
            intermediate_values_size = curr_gpu_memory_usage - self._curr_gpu_memory_usage
            self._curr_gpu_memory_usage = curr_gpu_memory_usage

            layer_container.effective_additional_memory_fwd = intermediate_values_size

            layer_container.forward_nan = not torch.isfinite(output).all()

            if self.device in ['cuda', torch.device('cuda')]:
                torch.cuda.synchronize()
            curr_time = time.time()
            layer_container.forward_latency = (curr_time - self._cur_time) * 1000. #ms
            self._cur_time = curr_time
        @torch.no_grad()
        def _backward_hook(module, grad_input, grad_output):
            layer_container.grad_in_shape = [gr_in.shape for gr_in in grad_input if gr_in is not None]
            layer_container.grad_out_shape = [gr_out.shape for gr_out in grad_output]

            layer_container.grad_in_nan = not torch.isfinite(grad_input[0].cpu().detach()).all() if grad_input[0] is not None else False
            layer_container.grad_out_nan = not torch.isfinite(grad_output[0].cpu().detach()).all()

            backprop_mem_usage = self._get_memory_usage()
            layer_container.effective_additional_memory_bwd = backprop_mem_usage - self._curr_gpu_memory_usage
            self._curr_gpu_memory_usage = backprop_mem_usage

            if self.device =='cuda':
                torch.cuda.synchronize()
            curr_time = time.time()
            layer_container.backward_latency = (curr_time - self._cur_time) * 1000. #ms
            self._cur_time = curr_time

        if hook_type == "forward":
            return _forward_hook
        elif hook_type == "backward":
            return _backward_hook
        else:
            raise ValueError(f"Invalid hook type: {hook_type}")
    
    def initialize_hooks(self):
        # Register hooks
        m_layers = []
        for m in self.models:
            layers = self._hook_layers(m)
            m_layers.append(layers)
        self.layers = m_layers

    def forward_mode(self):
        self.set_mode('fwd')
    def backward_mode(self):
        self.set_mode('bwd')

    def set_mode(self, mode: Literal['init', 'fwd', 'bwd']):
        self.mode = mode
        self._curr_gpu_memory_usage = self._get_memory_usage()
        self._init_gpu_memory_usage = self._get_memory_usage()
        self.start_timer()
    def _hook_layers(self, model: nn.Module):
        def _hook_layers_recurse(module: nn.Module):
            layer_dict = {}
            for name, layer in module.named_children():
                # If the layer has children, recursively build a dictionary
                layer_cont = LayerContainer(layer)
                layer_dict[name] = {
                    "info": layer_cont,
                    "children": {}
                }
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                    continue

                if self.watch_forward:
                    fwd_hook = layer.register_forward_hook(self._hook_wrapper("forward", layer_cont))
                    self.forward_hooks.append(fwd_hook)
                if self.watch_backward:
                    bwd_hook = layer.register_full_backward_hook(self._hook_wrapper("backward", layer_cont))
                    self.backward_hooks.append(bwd_hook)
                child_dict = _hook_layers_recurse(layer)
                if len(child_dict) > 0:
                    layer_dict[name]["children"].update(child_dict)

            return layer_dict
        
        layer_dict = {}
        model_cont = LayerContainer(model)
        model_name = model.__class__.__name__
        layer_dict[model_name] = {
            "info": model_cont,
            "children": {}
        }
        layer_dict[model_name]["children"] = _hook_layers_recurse(model)

        return layer_dict

    def _get_memory_usage(self):
        if self.device in ['cuda', torch.device('cuda')]:
            return torch.cuda.memory_allocated()
        if self.device in ['cpu', torch.device('cpu')]:
            return psutil.Process().memory_info().rss
        
    def _save(self, file_path: str):
        """
            Recursively search through the nested dictionary and
            convert each LayerContainer object to a dictionary using to_dict() method.
            Then, dump the dictionary to a JSON file.
        """
        def to_dict(obj):
            if isinstance(obj, LayerContainer):
                return obj.to_dict()
            if isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_dict(i) for i in obj]
            return obj
        file_path = Path(file_path)
        suffix = file_path.suffix
        if suffix != '':
            file_path = file_path.parent / file_path.stem
        for i, m in enumerate(self.models):
            dict_to_save = self.layers[i]
            save_path = f"{file_path}_{m.__class__.__name__}.json"
            data = to_dict(dict_to_save)
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=4)

    def _integrate_upwards(self):
        """
            Due to the nested structure of torch models, some values cannot be computed directly
            but have to be computed from an integration over its children layers.

            The dictionary is recursively searched  in a bottom-up manner and sums up the 
            required values from the children to the parent layer.

            This includes: forward_latency, backward_latency, effective_additional_memory_fwd, effective_additional_memory_bwd,
        """
        def _integrate_upwards_recurse(layer_dict):

            fwd_lat_tot = 0.0; bwd_lat_tot = 0.0
            mem_fwd_tot = 0.0; mem_bwd_tot = 0.0

            if "children" in layer_dict.keys():
                if len(layer_dict['children']) > 0:
                    for name, sub_layer_dict in layer_dict['children'].items():
                        fwd_lat, bwd_lat, mem_fwd, mem_bwd = _integrate_upwards_recurse(layer_dict['children'])
                        fwd_lat_tot += fwd_lat
                        bwd_lat_tot += bwd_lat
                        mem_fwd_tot += mem_fwd
                        mem_bwd_tot += mem_bwd
                    layer_cont = layer_dict['info']
                    layer_cont.forward_latency = fwd_lat_tot
                    layer_cont.backward_latency = bwd_lat_tot
                    layer_cont.effective_additional_memory_fwd = mem_fwd_tot
                    layer_cont.effective_additional_memory_bwd = mem_bwd_tot
                    return fwd_lat_tot, bwd_lat_tot, mem_fwd_tot, mem_bwd_tot
                else:
                    layer_cont = layer_dict['info']
                    return layer_cont.forward_latency, layer_cont.backward_latency, layer_cont.effective_additional_memory_fwd, layer_cont.effective_additional_memory_bwd
            else:
                for _, new_layer_dict in layer_dict.items():
                    fwd_lat, bwd_lat, mem_fwd, mem_bwd = _integrate_upwards_recurse(new_layer_dict)
                    fwd_lat_tot += fwd_lat
                    bwd_lat_tot += bwd_lat
                    mem_fwd_tot += mem_fwd
                    mem_bwd_tot += mem_bwd
                return fwd_lat_tot, bwd_lat_tot, mem_fwd_tot, mem_bwd_tot
        for m_layers in self.layers:
            _ = _integrate_upwards_recurse(m_layers)


