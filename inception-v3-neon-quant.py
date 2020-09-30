import os
import time
import numpy as np
import pyarmnn as ann
from util import load_test_image

model_dir ="../tvm-bench/inception_v3_quant"
tflite_model_file = os.path.join(model_dir, "inception_v3_quant.tflite")
tflite_model_buf = open(tflite_model_file, "rb").read()

dtype="uint8"
image_data = load_test_image(dtype)

parser = ann.ITfLiteParser()
network = parser.CreateNetworkFromBinaryFile(tflite_model_file)

options = ann.CreationOptions()
rt = ann.IRuntime(options)
preferredBackends = [ ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]

opt_network, _ = ann.Optimize(network, preferredBackends, rt.GetDeviceSpec(), ann.OptimizerOptions())
net_id, _ = rt.LoadNetwork(opt_network)

input_names = parser.GetSubgraphInputTensorNames(0)
input_binding_info = parser.GetNetworkInputBindingInfo(0, input_names[0])
input_tensors = ann.make_input_tensors([input_binding_info], [image_data])

output_names = parser.GetSubgraphOutputTensorNames(0)
output_binding_info = parser.GetNetworkOutputBindingInfo(0, output_names[0])
output_tensors = ann.make_output_tensors([output_binding_info])


repeat=10
numpy_time = np.zeros(repeat)
for i in range(0,repeat):
    start_time = time.time()

    rt.EnqueueWorkload(0, input_tensors, output_tensors) # Run inference
    #out_tensor = ann.workload_tensors_to_ndarray(output_tensors)[0][0]

    elapsed_ms = (time.time() - start_time) * 1000
    numpy_time[i] = elapsed_ms


out_tensor = ann.workload_tensors_to_ndarray(output_tensors)[0][0]
print("armnn MobileNet v2 quant %-19s (%s)" % ("%.2f ms" % np.mean(numpy_time), "%.2f ms" % np.std(numpy_time)))
