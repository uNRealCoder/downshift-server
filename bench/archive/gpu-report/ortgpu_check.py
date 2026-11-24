import os, sys
lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
os.add_dll_directory(lib)
os.environ["PATH"] = lib + os.pathsep + os.environ["PATH"]
import onnxruntime as ort
from onnx import helper, TensorProto as T
g = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g",
    [helper.make_tensor_value_info("x", T.FLOAT, [1, 4])], [helper.make_tensor_value_info("y", T.FLOAT, [1, 4])])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 8
s = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("ort", ort.__version__, "active:", s.get_providers())
