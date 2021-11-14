import torch
'''''''''
class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = torch.nn.Linear(2,10)
        self.fc2 = torch.nn.Linear(10,2)

    def forward(self, x):
        return self.fc2(self.fc1(x))

model = net()
trace_model = torch.jit.trace(model, torch.rand((1,2)))
torch.jit.save(trace_model, "net_trace.pt")

#==================================================
class net(torch.jit.ScriptModule):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = torch.nn.Linear(2,10)
        self.fc2 = torch.nn.Linear(10,2)

    @torch.jit.script_method
    def forward(self, x):
        return self.fc2(self.fc1(x))

model = net()
torch.jit.save(model, "net_script.pt")
'''''
#jit_model = torch.jit.load("net_script.pt")
#output = jit_model(torch.ones((1,2)))
#print(output)

#================================ onnx =================================
class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = torch.nn.Linear(2,10)
        self.fc2 = torch.nn.Linear(10,2)

    def forward(self, x):
        return self.fc2(self.fc1(x))

model = net()
torch_input = torch.ones((1,2))
torch_output = model(torch_input)
input_name = ["fc1"]
output_name = ["fc2"]
torch.onnx.export(model, torch_input, "net.onnx", input_names=input_name, output_names=output_name)

import onnx
import numpy as np
import onnxruntime

onnx_model = onnx.load("net.onnx")
onnx.checker.check_model(onnx_model)

seesion = onnxruntime.InferenceSession("net.onnx")
onnx_input = {seesion.get_inputs()[0].name: np.ones((1,2)).astype(np.float32)}
onnx_output = seesion.run(None, onnx_input)
np.testing.assert_allclose(torch_output.data.numpy(), onnx_output[0], rtol=1e-03, atol=1e-05)





