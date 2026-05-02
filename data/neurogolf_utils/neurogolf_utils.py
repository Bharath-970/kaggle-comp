# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module containing utilities for the IJCAI-ECAI 2026 NeuroGolf Challenge."""

import itertools
import json
import math
import pathlib
import traceback

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnx_tool
import onnxruntime


display = IPython.display.display
FileLink = IPython.display.FileLink

_BATCH_SIZE, _CHANNELS, _HEIGHT, _WIDTH = 1, 10, 30, 30
_NEUROGOLF_DIR = "/kaggle/input/competitions/neurogolf-2026/"
_COLORS = [
    (0, 0, 0),
    (30, 147, 255),
    (250, 61, 49),
    (78, 204, 48),
    (255, 221, 0),
    (153, 153, 153),
    (229, 59, 163),
    (255, 133, 28),
    (136, 216, 241),
    (147, 17, 49),
    (240, 240, 240),
    (146, 117, 86)
]
_DATA_TYPE = onnx.TensorProto.FLOAT
_EXCLUDED_OP_TYPES = ["LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"]
_FILESIZE_LIMIT_IN_BYTES = 1.44 * 1024 * 1024
_GRID_SHAPE = [_BATCH_SIZE, _CHANNELS, _HEIGHT, _WIDTH]
_IR_VERSION, _OPSET_IMPORTS = 10, [onnx.helper.make_opsetid("", 10)]
_TASK_ZERO = {
    "train": [{
        "input": [
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
        "output": [
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 5, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 0, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 0, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 0, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 0, 5, 5],
            [5, 1, 1, 1, 1, 1, 1, 0, 5, 5],
            [5, 5, 0, 0, 0, 0, 0, 0, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
    }],
    "test": [{
        "input": [
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 4, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 4, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 5, 5, 5],
            [5, 5, 4, 5, 5, 5, 4, 5, 5, 5],
            [5, 5, 4, 5, 5, 5, 4, 5, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
        "output": [
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 4, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 4, 0, 5],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 5, 5, 5],
            [5, 5, 4, 0, 0, 0, 4, 0, 5, 5],
            [5, 5, 4, 0, 5, 5, 4, 0, 5, 5],
            [5, 5, 4, 4, 4, 4, 4, 0, 5, 5],
            [5, 5, 5, 0, 0, 0, 0, 0, 5, 5],
        ],
    }],
    "arc-gen": [{
        "input": [
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 2, 2, 2, 2, 2, 2, 5, 5],
            [5, 5, 2, 5, 5, 5, 5, 2, 5, 5],
            [5, 5, 2, 5, 5, 5, 5, 2, 5, 5],
            [5, 5, 2, 5, 5, 5, 5, 2, 5, 5],
            [5, 5, 2, 5, 5, 5, 5, 2, 5, 5],
            [5, 5, 2, 2, 2, 2, 2, 2, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
        "output": [
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 2, 2, 2, 2, 2, 2, 5, 5],
            [5, 5, 2, 0, 0, 0, 0, 2, 0, 5],
            [5, 5, 2, 0, 5, 5, 5, 2, 0, 5],
            [5, 5, 2, 0, 5, 5, 5, 2, 0, 5],
            [5, 5, 2, 0, 5, 5, 5, 2, 0, 5],
            [5, 5, 2, 2, 2, 2, 2, 2, 0, 5],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
    }],
}


def calculate_memory(filename):
    model = onnx.load(filename)
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    init_names = {init.name for init in graph.initializer}
    io_names = {t.name for t in list(graph.input) + list(graph.output)}
    if io_names.intersection(init_names): return None
    if model.functions: return None
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}: return None
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH,
                             onnx.AttributeProto.GRAPHS]:
                return None
    memory = 0
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not item.type.HasField("tensor_type"): continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"): return None
        num_elements = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param"): return None
            if not dim.HasField("dim_value"): return None
            num_elements *= dim.dim_value
        if item.name in ['input', 'output']: continue
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        memory += num_elements * np.dtype(np_dtype).itemsize
    return memory


def check_network(filename):
  file_path = pathlib.Path(filename)
  if not file_path.is_file():
    print(f"Error: File {filename} does not exist.")
    return False
  if (filesize := file_path.stat().st_size) > _FILESIZE_LIMIT_IN_BYTES:
    print(f"Error: Filesize {filesize} exceeds {_FILESIZE_LIMIT_IN_BYTES}.")
    return False
  return True


def convert_to_numpy(example):
  benchmark = {}
  example_shape = (1, _CHANNELS, _HEIGHT, _WIDTH)
  for mode in ["input", "output"]:
    benchmark[mode] = np.zeros(example_shape, dtype=np.float32)
    grid = example[mode]
    if max(len(grid), len(grid[0])) > 30: return None
    for r, _ in enumerate(grid):
      for c, color in enumerate(grid[r]):
        benchmark[mode][0][color][r][c] = 1.0
  return benchmark


def convert_from_numpy(benchmark):
  example = []
  _, channels, height, width = benchmark.shape
  for row in range(height):
    cells = []
    for col in range(width):
      colors = [c for c in range(channels) if benchmark[0][c][row][col] == 1]
      cells.append(colors[0] if len(colors) == 1 else (11 if colors else 10))
    while cells and cells[-1] == 10:
      cells.pop(-1)
    example.append(cells)
  while example and not example[-1]:
    example.pop(-1)
  return example


def score_network(m):
  model = onnx_tool.loadmodel(m, {'verbose': False, 'constant_folding': True})
  g = model.graph
  g.graph_reorder_nodes()
  g.shape_infer(None)
  g.profile()
  if not g.valid_profile:
    print("Error: Invalid profile.")
    return None, None, None
  for key in g.nodemap.keys():
    if g.nodemap[key].op_type.upper() in _EXCLUDED_OP_TYPES:
      print(f"Error: Op type {g.nodemap[key].op_type} is not permitted.")
      return None, None, None
    if g.nodemap[key].memory < 0 or g.nodemap[key].params < 0:
      print(f"Error: Negative memory or params value detected.")
      return None, None, None
    if min(g.nodemap[key].macs) < 0:
      print(f"Error: Negative MACs value detected.")
      return None, None, None
  memory = calculate_memory(m)
  if memory is None:
    print("Error: Shapes not statically defined.")
    return None, None, None
  return int(sum(g.macs)), memory, int(g.params)


def load_examples(task_num):
  """Loads relevant data from ARC-AGI and ARC-GEN."""
  if not task_num:
    return _TASK_ZERO
  with open(_NEUROGOLF_DIR + f"task{task_num:03d}.json") as f:
    examples = json.load(f)
  return examples


def run_network(session, benchmark_input):
  result = session.run(["output"], {"input": benchmark_input})
  return (result[0] > 0.0).astype(float)


def show_examples(examples, bgcolor=(255, 255, 255)):
  # Determine the dimensions of the image to be rendered.
  width, height, offset = 0, 0, 1
  for example in examples:
    grid, output = example["input"], example["output"]
    width += len(grid[0]) + 1 + len(output[0]) + 4
    height = max(height, max(len(grid), len(output)) + 4)
  # Determine the contents of the image.
  image = [[bgcolor for _ in range(width)] for _ in range(height)]
  for example in examples:
    grid, output = example["input"], example["output"]
    grid_width, output_width = len(grid[0]), len(output[0])
    for r, row in enumerate(grid):
      for c, cell in enumerate(row):
        image[r + 2][offset + c + 1] = _COLORS[cell]
    offset += grid_width + 1
    for r, row in enumerate(output):
      for c, cell in enumerate(row):
        image[r + 2][offset + c + 1] = _COLORS[cell]
    offset += output_width + 4
  # Draw the image.
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_axes([0, 0, 1, 1])
  ax.imshow(np.array(image))
  # Draw the horizontal and vertical lines.
  offset = 1
  for example in examples:
    grid, output = example["input"], example["output"]
    grid_width, grid_height = len(grid[0]), len(grid)
    output_width, output_height = len(output[0]), len(output)
    ax.hlines([r + 1.5 for r in range(grid_height+1)],
              xmin=offset+0.5, xmax=offset+grid_width+0.5, color="black")
    ax.vlines([offset + c + 0.5 for c in range(grid_width+1)],
              ymin=1.5, ymax=grid_height+1.5, color="black")
    offset += grid_width + 1
    ax.hlines([r + 1.5 for r in range(output_height+1)],
              xmin=offset+0.5, xmax=offset+output_width+0.5, color="black")
    ax.vlines([offset + c + 0.5 for c in range(output_width+1)],
              ymin=1.5, ymax=output_height+1.5, color="black")
    offset += output_width + 2
    ax.vlines([offset+0.5], ymin=-0.5, ymax=height-0.5, color="black")
    offset += 2
  ax.set_xticks([])
  ax.set_yticks([])


def show_legend():
  image = [[(255, 255, 255) for _ in range(21)] for _ in range(5)]
  for idx, color in enumerate(_COLORS[:10]):
    image[1][2 * idx + 1] = color
  for idx, color in enumerate(_COLORS[10:]):
    for col in range(3):
      image[3][12 * idx + col + 3] = color
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_axes([0, 0, 1, 1])
  ax.imshow(np.array(image))
  for idx, _ in enumerate(_COLORS[:10]):
    color = "white" if idx in [0, 9] else "black"
    ax.text(2 * idx + 0.9, 1.1, str(idx), color=color)
  ax.text(3.4, 3.1, "no color", color="black")
  ax.text(5.75, 3.1, "<--- special colors to indicate one-hot encoding errors --->", color="black")
  ax.text(14.85, 3.1, "too many colors", color="white")
  ax.set_xticks([])
  ax.set_yticks([])


def single_layer_conv2d_network(weight_fn, kernel_size):
  kernel_offsets = range(-kernel_size // 2 + 1, kernel_size // 2 + 1)
  kernel_shape = [kernel_size, kernel_size]
  w_shape = [_CHANNELS, _CHANNELS, kernel_size, kernel_size]
  pads = [kernel_size // 2] * 4
  weight_cells = itertools.product(range(_CHANNELS), range(_CHANNELS),
                                   kernel_offsets, kernel_offsets)
  weights = [weight_fn(o, i, (r, c)) for (o, i, r, c) in weight_cells]

  x = onnx.helper.make_tensor_value_info("input", _DATA_TYPE, _GRID_SHAPE)
  y = onnx.helper.make_tensor_value_info("output", _DATA_TYPE, _GRID_SHAPE)
  w = onnx.helper.make_tensor("W", _DATA_TYPE, w_shape, weights)
  node_def = onnx.helper.make_node("Conv", ["input", "W"], ["output"],
                                   kernel_shape=kernel_shape, pads=pads)
  graph_def = onnx.helper.make_graph([node_def], "graph", [x], [y], [w])
  model_def = onnx.helper.make_model(graph_def, ir_version=_IR_VERSION,
                                     opset_imports=_OPSET_IMPORTS)
  return model_def


def verify_network(network, task_num, examples):
  filename = "task{:03d}.onnx".format(task_num)
  onnx.save(network, filename)
  if not check_network(filename): return
  try:
    session = onnxruntime.InferenceSession(filename)
  except onnxruntime.ONNXRuntimeError as e:
    print(f"Error: Unable to load ONNX model: {e}")
    return
  arc_agi_right, arc_agi_wrong, arc_agi_expected = verify_subset(session, examples["train"] + examples["test"])
  arc_gen_right, arc_gen_wrong, arc_gen_expected = verify_subset(session, examples["arc-gen"])
  print(f"Results on ARC-AGI examples: {arc_agi_right} pass, {arc_agi_wrong} fail")
  print(f"Results on ARC-GEN examples: {arc_gen_right} pass, {arc_gen_wrong} fail")
  print()
  macs, memory, params = score_network(filename)
  if macs is None or memory is None or params is None:
    print("Error: Your network performance could not be measured")
  elif arc_agi_wrong + arc_gen_wrong == 0:
    print("Your network IS READY for submission!")
    print()
    print("Performance stats (memory values reported here are approximate):")
    onnx_tool.model_profile(filename)
    points = max(1.0, 25.0 - math.log(max(1.0, macs + memory + params)))
    print()
    print(f"It appears to require {macs} MACs + {memory} bytes + {params} params, yielding {points:.3f} points.")
    print()
    print("Next steps:")
    print(f" * Click the link below to download {filename} onto your local machine.")
    print(" * Create a zip file containing that network along with all others.")
    print(" * Submit that zip file to the Kaggle competition so that it can be officially scored.")
    print()
    display(FileLink(filename))
  else:
    print("Your network IS NOT ready for submission.")
    expected = None
    expected = arc_agi_expected if arc_agi_expected is not None else expected
    expected = arc_gen_expected if arc_gen_expected is not None else expected
    if expected is None: return
    benchmark = convert_to_numpy(expected)
    actual = {}
    actual["input"] = expected["input"]
    actual["output"] = convert_from_numpy(run_network(session, benchmark["input"]))
    print("The expected result is shown in green; your actual result is shown in red.")
    show_examples([expected], bgcolor=(200, 255, 200))
    show_examples([actual], bgcolor=(255, 200, 200))


def verify_subset(session, example_subset):
  right, wrong, expected, error = 0, 0, None, ""
  for example in example_subset:
    benchmark = convert_to_numpy(example)
    if not benchmark: continue
    try:
      user_output = run_network(session, benchmark["input"])
      if np.array_equal(user_output, benchmark["output"]):
        right += 1
      else:
        expected = example
        wrong += 1
    except onnxruntime.ONNXRuntimeError:
      error = traceback.format_exc()
      wrong += 1
  if error: print(f"Error: {error}")
  return right, wrong, expected

