import drjit as dr
from drjit.auto import TensorXf

pos = dr.zeros(TensorXf, (400, 400, 400))
active = dr.zeros(TensorXf, (400, 400, 400))
index = dr.zeros(TensorXf, (400, 400, 400))
mean = dr.zeros(TensorXf, (400, 400, 400))
value = dr.zeros(TensorXf, (400, 400, 400))

dr.scatter_reduce(dr.ReduceOp.Add, mean.array, value.array, index.array)
