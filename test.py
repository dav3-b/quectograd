from quectograd.engine import Value
from quectograd.nn import Neuron, Layer, MLP

# quectograd test

#a = ValueDebug(2.0, label='a')
#b = ValueDebug(-3.0, label='b')
#c = ValueDebug(10.0, label='c')
#d = a * b + c; d.label = 'd'
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = a * b
d = e + c;
print(d)
print(d._prev)
d.backward()
print(f"d.grad = {d.grad}")
print(f"e.grad = {e.grad}")
print(f"a.grad = {a.grad}")
print(f"b.grad = {b.grad}")
print(f"c.grad = {c.grad}")

x = [2.0, 3.0, -1.0]
n = Neuron(len(x))
res1 = n(x)
print(res1)

z = Layer(len(x), 3)
res2 = z(x)
print(res2)

q = MLP(len(x), [4, 4, 1])
res3 = q(x)
print(res3)

m = Value.zeros((3, 3))
print(m)
m[0, 0] = 1.0
print(m)

m = Module()
m.zero_grad()

print("--------------------------------------")
n = Neuron(len(x))
print(n)
#print(n(x))

print("--------------------------------------")
l = Layer(len(x),2)
print(l)
#print(l(x))

print("--------------------------------------")
model = MLP(len(x), [3, 2, 1])
print(model)
#print(model(x))

print("--------------------------------------")
n_debug = NeuronDebug(len(x))
print(n_debug)
print(n_debug(x))

