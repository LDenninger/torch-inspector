<div align="center">
<h1>Inspector Gadgets</h1>
<p>The inspector for your PyTorch model to get a better in-depth understanding.</p>
<img src="resources/title.png" alt="Title picture">
</div>

The `torch_inspector` package provides the `InspectorGadgets` module which one can register PyTorch models with and inspect it during run-time without adjusting any source code of the model and thus also works seamlessly with external code. After registering the PyTorch models, the forward and backward pass of the model can be performed as-is, such that one can inspect the behaviour of the model in original environment and does not require any specific test bed. During the forward and backward pass Inspector Gadgets captures on a per-layer basis the number of (learnable) parameters, input/output shapes, gradient shapes, the respective memory usage of each Tensor, the additional memory usage beyond parameters and gradients and latency. Furthermore, it implements NaN and Inf checks for all intermediate values.

**Disclaimer:** This project is still in development and bugs might occur. If you observe failure cases please create an issue or notify me. As of now, it is only intended to give in-depth insight and not produce a nice overview of models. Finally, it is planned to add further functionalities, such as custom checks, better visualization, better approximation of latency etc.

## Installation
The project only requires an existing installation of PyTorch and NumPy. To install this project, simply clone it to your machine and run:
```shell
pip install -e .
```
## Usage
There are currently two ways to run Inspector Gadgets, either as a context manager or as standalone module.
The context manager simply automates the `start()` and `finish()` function.

**Standalone:**
```python
from torch_inspector import InspectorGadgets
model = torchvision.models.resnet18(pretrained=True).train()
dummy_input = torch.randn(1, 3, 224, 224)
# Initialize Inspector Gadgets with your model and the file the results should be saved to
inspector = InspectorGadgets(model, save_file='examples/insight')
# Has to be called before running the model
inspector.start()
output = model(dummy_input)
# (Recommended) Get correct memory usage and latency for backward in the last layer
inspector.backward_mode()
loss = torch.mean(output)
loss.backward()
# Removes all hooks and releases every link to the model
inspector.finish()
```
**Context Manager:**
```python
model = torchvision.models.resnet18(pretrained=True).train()
dummy_input = torch.randn(1, 3, 224, 224)

with InspectorGadgets(model, save_file='examples/insight') as inspector:
    output = model(dummy_input)
    inspector.backward_mode()
    loss = torch.mean(output)
    loss.backward()
```
## Output Format
The raw output file can get quite large but provides you with in-depth information for each layer. The overall structure is quite simple:
```json
{
    <model name>: {
        "info": {
            <in-depth information>
        },
        "children" : {
            <block name>: {
                "info": {...}
                "children": {...}
            },
            ...
        }
    }
}
```
Each module is accompanied by the `"info"` block providing all required information in the format:
```json
{
    "type": str,
    "input_shapes": List[List[int]],
    "output_shapes": List[List[int]],
    "grad_in_shape": List[List[int]],
    "grad_out_shape": List[List[int]],
    "num_parameters": int,
    "num_learnable_parameters": int,
    "param_memory_usage": float,
    "gradient_memory_usage": float,
    "input_memory_usage": List[float],
    "output_memory_usage": List[float],
    "effective_additional_memory_fwd": float,
    "effective_additional_memory_bwd": float,
    "forward_nan": bool,
    "grad_in_nan": bool,
    "grad_out_nan": bool,
    "total_memory": float,
    "forward_latency": float,
    "backward_latency": float
}
```
`type`: Class name of the module <br/>
`input_shapes`: List of shapes of all inputs <br/>
`output_shapes`: List of shapes of all outputs <br/>
`grad_in_shape`: List of shapes of all gradients w.r.t. the input <br/>
`grad_out_shape`: List of shapes of all gradients w.r.t. the input <br/>
`num_parameters`: Number of parameters <br/>
`num_learnable_parameters`: Number of parameters with required gradient <br/>
`param_memory_usage`: Memory usage of all parameters <br/>
`gradient_memory_usage`: Memory usage of all gradients <br/>
`input_memory_usage`: Memory usage of the input <br/>
`output_memory_usage`: Memory usage of the input <br/>
`effective_additional_memory_fwd`: Measured additional memory requirement after forward pass <br/>
`effective_additional_memory_bwd`: Measured additional memory requirement after forward pass <br/>
`forward_nan`: NaN or Inf in output <br/>
`grad_in_nan`: NaN or Inf in gradients w.r.t. the input <br/>
`grad_out_nan`: NaN or Inf in gradients w.r.t. the input <br/>
`total_memory`: Total memory for parameters, gradients, and input <br/>
`forward_latency`: Latency of the forward pass <br/>
`backward_latency`: Latency of the backward pass <br/>

Examples and an example output can be found in the `./examples` directory.

## Issues
1. Currently, the latency of higher-level modules in the nested structure of PyTorch is computed from the summation of its sub-modules. This does not consider any low-level parallelization and the asynchronous behaviour of the Cuda-backend which results in greatly over-estimated latencies especially for high-level modules and modules with many sub-modules.
2. The current output format is somewhat confusing and might require some better output format.
## License
`torch_inspector` is licensed under BSD-3.
## Author
Luis Denninger <l_denninger@uni-bonn.de>