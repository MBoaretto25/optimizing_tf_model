import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile


print("[INFO] start...")

## Parameters of the model
file_path = "path/of/the/model.pb"
trt_file_path = "path/of/the/optimized/model_trt.pb"
test_image_path = "test/image/image.jpg"
input_size = (100, 100)
output_names = "output_node_names"


## Start optimization with tensor rt
print("[INFO] load frozen model...")
with gfile.FastGFile(file_path, "rb") as pb_file:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(pb_file.read())
    tf.import_graph_def(graph_def, name="")
    print("[INFO] start optimization...")
    trt_graph = trt.create_inference_graph(
        input_graph_def=file_path,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 32,
        precision_mode='FP16',
        minimum_segment_size=50
    )

print("[INFO] save optimized model as {}".format(trt_file_path))
with open(trt_file_path, 'wb') as f:
    f.write(trt_graph.SerializeToString())


## Load TRT optimized model from saved file
print("[INFO] load optimized model")
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
input_nodes = "input_nodes"
output_nodes = "outputs_nodes"

model = tf.Session(config=tf_config)
with gfile.FastGFile(file_path, "rb") as pb_file:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(pb_file.read())
    tf.import_graph_def(graph_def, name="")

tf_input = model.graph.get_tensor_by_name(input_nodes)
tf_output = model.graph.get_tensor_by_name(output_nodes)

print("[INFO] model loaded!")
## Do something with model...