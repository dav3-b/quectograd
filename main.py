from quectograd.engine import Value
from quectograd.nn import Neuron, Layer, MLP

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

q = MLP(3, [4, 4, 1])

step_size = 0.1

for i in range(20):
  # forward pass
  ypred = [q(x) for x in xs]
  loss = sum((y_out - y_gt)**2 for y_gt, y_out in zip(ys, ypred))

  # backward pass
  #for p in q.parameters():
  #  p.grad = 0.0
  q.zero_grad()
  
  loss.backward()
  
  # update
  for p in q.parameters():
    p.data += -step_size * p.grad

  print(f"Loss[{i}] = {loss.data}")

print(f"Ground truth: {ys}")
print(f"Predictions: {ypred}")
