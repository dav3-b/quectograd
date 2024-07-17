import math

class Value:

  def __init__(self, data, _children=(), dtype=float):
    # public
    assert dtype == float or dtype == int, "Only supporting int/float data types"
    self.data = data
    self.grad = 0.0
    # private
    self._prev = set(_children)
    self._backward = lambda: None
    self.dtype = dtype
  
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other))

    def _add_backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    
    out._backward = _add_backward
    return out
  
  def __radd__(self, other):
    return self + other

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return ohter + (-self)

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out =  Value(self.data * other.data, (self, other))

    def _mul_backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

    out._backward = _mul_backward
    return out

  def __rmul__(self, other):
    return self * other

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self, ))

    def _pow_backward():
      self.grad += other * (self.data**(other - 1)) * out.grad

    out._backward = _pow_backward
    return out

  def __truediv__(self, other):
    return self * (other**-1)
  
  def __rtruediv__(self, other):
    return other * (self**-1)

  def __setitem__(self, index, value):
    row, col = index
    assert self.dtype == type(value), "Wrong data type, change it!"
    self.data[row][col] = value

  def __getitem__(self, index):
    row, col = index
    return self.data[row][col]

  def __repr__(self):
    return f"Value({self.data})"

  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ))

    def _exp_backward():
      self.grad += out.data * out.grad

    out._backward = _exp_backward
    return out

  def tanh(self):
    x = self.data
    tanh = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(tanh, (self, ))

    def _tanh_backward():
      self.grad += (1 - tanh**2) * out.grad

    out._backward = _tanh_backward
    return out

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self, ))

    def _relu_backward():
      self.grad += (out.data > 0) * out.grad

    out._backward = _relu_backward
    return out

  def backward(self):
    # topological order all of the children in the graph
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)

    build_topo(self)
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()

  def item(self):
    return self.data
  
  def zeros(*size, dtype=float):
    assert len(size) <= 2, "Until now the only dimension supported are: 1D and 2D."
    assert dtype == float or dtype == int, "Only supporting int/float data types"
    
    x = 0.0
    if dtype == int:
      x = 0
    
    if len(size) == 1:
      return Value([x for r in range(size[0])], dtype=dtype)
    elif len(size) == 2:
      return Value([[x for c in range(size[1])] for r in range(size[0])], dtype=dtype)
  
  def ones(*size, dtype=float):
    assert len(size) <= 2, "Until now the only dimension supported are: 1D and 2D."
    assert dtype == float or dtype == int, "Only supporting int/float data types"
    
    x = 1.0
    if dtype == int:
      x = 1
    
    if len(size) == 1:
      return Value([x for r in range(size[0])], dtype=dtype)
    elif len(size) == 2:
      return Value([[x for c in range(size[1])] for r in range(size[0])], dtype=dtype)

# Sub-class for debugging porpuse
DEBUG = False

class ValueDebug(Value):

  def __init__(self, data, _children=(), dtype=float, _op='', label=''):
    super().__init__(data, _children, dtype)
    # public
    self.label = label
    # private
    self._op = _op
  
  def __add__(self, other):
    other = other if isinstance(other, ValueDebug) else ValueDebug(other)
    out = ValueDebug(self.data + other.data, (self, other), '+')

    def _add_backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
      if DEBUG == True:
        print(f"add_backward: ({self.data}, {other.data})")
    
    out._backward = _add_backward
    return out

  def __mul__(self, other):
    other = other if isinstance(other, ValueDebug) else ValueDebug(other)
    out =  ValueDebug(self.data * other.data, (self, other), '*')

    def _mul_backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
      if DEBUG == True:
        print(f"mul_backward: ({self.data}, {other.data})")

    out._backward = _mul_backward
    return out
  
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = ValueDebug(self.data**other, (self, ), "pow")

    def _pow_backward():
      self.grad += other * (self.data**(other - 1)) * out.grad

    out._backward = _pow_backward
    return out
  
  def __repr__(self):
    return f"ValueDebug(data={self.data})"
  
  def exp(self):
    x = self.data
    out = ValueDebug(math.exp(x), (self, ), "exp")

    def _exp_backward():
      self.grad += out.data * out.grad

    out._backward = _exp_backward
    return out
  
  def tanh(self):
    x = self.data
    tanh = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = ValueDebug(tanh, (self, ), "tanh")

    def _tanh_backward():
      self.grad += (1 - tanh**2) * out.grad
      if DEBUG == True:
        print(f"tanh_backward: ({self.data})")

    out._backward = _tanh_backward
    return out
  
  def relu(self):
    out = ValueDebug(0 if self.data < 0 else self.data, (self, ), "relu")

    def _relu_backward():
      self.grad += (out.data > 0) * out.grad

    out._backward = _relu_backward
    return out

