import os
import sys; sys.path.insert(0, os.getcwd())

import torch
import torchvision
from torch_inspector.inspector import InspectorGadgets

model = torchvision.models.resnet18(pretrained=True).train()
model = model.train()
dummy_input = torch.randn(1, 3, 224, 224)

inspector = InspectorGadgets(model, save_file='examples/insight')

inspector.start()
output = model(dummy_input)

inspector.backward_mode()
loss = torch.mean(output)
loss.backward()

inspector.finish()