import onnx
model = onnx.load('output.onnx')
ops = set()
for node in model.graph.node:
  ops.add(node.op_type)
print(ops)
