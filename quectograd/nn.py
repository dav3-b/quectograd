import random
from quectograd.engine import Value

class Module:

  def parameters(self):
    return []
  
  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0.0

class Neuron(Module):
  
  def __init__(self, num_input, nonlin="tanh"):
    self.w = [Value(random.uniform(-1,1)) for _ in range(num_input)]
    self.b = Value(0.0)
    self.nonlin = nonlin

  def __call__(self, x):
    # forward pass
    # w * x + b
    assert len(self.w) == len(x), "Dimension mismatch!"
    activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
    if self.nonlin == "relu":
      return activation.relu()
    elif self.nonlin == "tanh":
      return activation.tanh()
    else:
      return None
  
  def parameters(self):
    return self.w + [self.b]
  
  def __repr__(self):
    dict_neuron = {"weights": self.w, "bias": self.b, "nonlin": self.nonlin}
    return f"{dict_neuron}"

class Layer(Module):

  def __init__(self, num_input_neurons, num_output_neurons, output_layer=True):
    self.num_input_neurons = num_input_neurons
    if output_layer == False:  
      self.neurons = [Neuron(num_input_neurons, "relu") for _ in range(num_output_neurons)]
    else:
      self.neurons = [Neuron(num_input_neurons) for _ in range(num_output_neurons)]

  def __call__(self, x):
    assert self.num_input_neurons == len(x), "Dimension mismatch!"
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]
  
  def __repr__(self):
    dict_layer = {}
    for i in range(len(self.neurons)):
      name = f"neuron{i}"
      dict_layer[name] = self.neurons[i]
    return f"{dict_layer}"

class MLP(Module):

  def __init__(self, num_input_layer, num_output_layer):
    sz = [num_input_layer] + num_output_layer
    self.num_input_layer = num_input_layer
    self.layers = []
    for i in range(len(num_output_layer)):
      if i != len(num_output_layer) - 1:
        self.layers.append(Layer(sz[i], sz[i + 1], False))      
      else:
        self.layers.append(Layer(sz[i], sz[i + 1]))      

  def __call__(self, x):
    assert self.num_input_layer == len(x), "Dimension mismatch!"
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for l in self.layers for p in l.parameters()]
  
  def __repr__(self, debug=False):
    dict_mlp = {}
    for i in range(len(self.layers)):
      if i != len(self.layers) - 1:
        name = f"hidden_layer{i}"
      else:
        name = "output_layer"
      dict_mlp[name] = self.layers[i]
    
    return f"{dict_mlp}"

# Debug sub-classes
class NeuronDebug(Neuron):
  
  def __init__(self, num_input, nonlin="tanh"):
    super().__init__(num_input, nonlin)

  def __repr__(self):
    dict_neuron = {"weights": self.w, "bias": self.b, "nonlin": self.nonlin}
    return f"Neuron(\n  {dict_neuron}\n)"

class LayerDebug(Layer):

  def __init__(self, num_input_neurons, num_output_neurons, output_layer=True):
    super().__init__(num_input_neurons, num_output_neurons, output_layer)

  def __repr__(self):
    dict_layer = {}
    for i in range(len(self.neurons)):
      name = f"neuron{i}"
      dict_layer[name] = self.neurons[i]
    return f"Layer(\n {dict_layer}\n)"

class MLPDebug(MLP):

  def __init__(self, num_input_layer, num_output_layer):
    super().__init__(num_input_layer, num_output_layer)
  
  def __repr__(self):
    dict_mlp = {}
    for i in range(len(self.layers)):
      if i != len(self.layers) - 1:
        name = f"hidden_layer{i}"
      else:
        name = "output_layer"
      dict_mlp[name] = self.layers[i]
    
    return f"MPL(\n  {dict_mlp}\n)"
