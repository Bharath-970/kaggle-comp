from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

STATIC_SHAPE = [1, 10, 30, 30]
DEFAULT_OPSET = 10
IR_VERSION = 10


def _tensor_value_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, STATIC_SHAPE)


def _validate_color_weight_matrix(weight_matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(weight_matrix, dtype=np.float32)
    if array.shape != (10, 10):
        raise ValueError("weight_matrix must have shape [10, 10]")
    return array


def build_identity_model(model_name: str = "identity_color") -> onnx.ModelProto:
    node = helper.make_node("Identity", ["input"], ["output"], name="identity")
    graph = helper.make_graph(
        nodes=[node],
        name=model_name,
        inputs=[_tensor_value_info("input")],
        outputs=[_tensor_value_info("output")],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-native-builder",
        opset_imports=[helper.make_operatorsetid("", DEFAULT_OPSET)],
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model)
    return model


def build_pointwise_color_map_model(
    weight_matrix: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    model_name: str = "pointwise_color_map",
) -> onnx.ModelProto:
    weights_2d = _validate_color_weight_matrix(weight_matrix)
    weights_4d = weights_2d.reshape(10, 10, 1, 1)

    initializers = [numpy_helper.from_array(weights_4d.astype(np.float32), name="W")]
    conv_inputs = ["input", "W"]

    if bias is not None:
        bias_array = np.asarray(bias, dtype=np.float32)
        if bias_array.shape != (10,):
            raise ValueError("bias must have shape [10]")
        initializers.append(numpy_helper.from_array(bias_array, name="B"))
        conv_inputs.append("B")

    conv_node = helper.make_node(
        "Conv",
        conv_inputs,
        ["output"],
        name="pointwise_conv",
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )

    graph = helper.make_graph(
        nodes=[conv_node],
        name=model_name,
        inputs=[_tensor_value_info("input")],
        outputs=[_tensor_value_info("output")],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-native-builder",
        opset_imports=[helper.make_operatorsetid("", DEFAULT_OPSET)],
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model)
    return model


def build_single_conv_model(
    weight_tensor: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int | None = None,
    model_name: str = "single_conv",
) -> onnx.ModelProto:
    weights = np.asarray(weight_tensor, dtype=np.float32)
    if weights.ndim != 4:
        raise ValueError("weight_tensor must have shape [out_ch, in_ch, k, k]")

    kernel_h, kernel_w = int(weights.shape[2]), int(weights.shape[3])
    if kernel_h != kernel_w or kernel_h % 2 == 0:
        raise ValueError("Only odd square kernels are supported")
    if stride != 1:
        raise ValueError("Only stride=1 is supported for static 30x30 output")

    pad = kernel_h // 2 if padding is None else padding
    if pad != kernel_h // 2:
        raise ValueError("Padding must preserve 30x30 output shape")

    initializers = [numpy_helper.from_array(weights, name="W_conv")]
    conv_inputs = ["input", "W_conv"]

    if bias is not None:
        bias_array = np.asarray(bias, dtype=np.float32)
        if bias_array.shape != (weights.shape[0],):
            raise ValueError("bias must have shape [out_ch]")
        initializers.append(numpy_helper.from_array(bias_array, name="B_conv"))
        conv_inputs.append("B_conv")

    conv_node = helper.make_node(
        "Conv",
        conv_inputs,
        ["output"],
        name="single_conv",
        kernel_shape=[kernel_h, kernel_w],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad],
    )

    graph = helper.make_graph(
        nodes=[conv_node],
        name=model_name,
        inputs=[_tensor_value_info("input")],
        outputs=[_tensor_value_info("output")],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-native-builder",
        opset_imports=[helper.make_operatorsetid("", DEFAULT_OPSET)],
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model)
    return model


def build_maxpool_model(
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int | None = None,
    model_name: str = "maxpool",
) -> onnx.ModelProto:
    if kernel_size % 2 == 0:
        raise ValueError("Only odd kernel sizes are supported")
    if stride != 1:
        raise ValueError("Only stride=1 is supported for static 30x30 output")

    pad = kernel_size // 2 if padding is None else padding
    if pad != kernel_size // 2:
        raise ValueError("Padding must preserve 30x30 output shape")

    pool_node = helper.make_node(
        "MaxPool",
        ["input"],
        ["output"],
        name="maxpool",
        kernel_shape=[kernel_size, kernel_size],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad],
    )

    graph = helper.make_graph(
        nodes=[pool_node],
        name=model_name,
        inputs=[_tensor_value_info("input")],
        outputs=[_tensor_value_info("output")],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-native-builder",
        opset_imports=[helper.make_operatorsetid("", DEFAULT_OPSET)],
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model)
    return model


def build_minpool_model(
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int | None = None,
    model_name: str = "minpool",
) -> onnx.ModelProto:
    if kernel_size % 2 == 0:
        raise ValueError("Only odd kernel sizes are supported")
    if stride != 1:
        raise ValueError("Only stride=1 is supported for static 30x30 output")

    pad = kernel_size // 2 if padding is None else padding
    if pad != kernel_size // 2:
        raise ValueError("Padding must preserve 30x30 output shape")

    neg_node = helper.make_node("Neg", ["input"], ["neg_input"], name="minpool_neg")
    pool_node = helper.make_node(
        "MaxPool",
        ["neg_input"],
        ["pooled_neg"],
        name="minpool_max",
        kernel_shape=[kernel_size, kernel_size],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad],
    )
    out_node = helper.make_node("Neg", ["pooled_neg"], ["output"], name="minpool_out")

    graph = helper.make_graph(
        nodes=[neg_node, pool_node, out_node],
        name=model_name,
        inputs=[_tensor_value_info("input")],
        outputs=[_tensor_value_info("output")],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-native-builder",
        opset_imports=[helper.make_operatorsetid("", DEFAULT_OPSET)],
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model)
    return model


def build_depthwise_shift_model(
    dx: int,
    dy: int,
    *,
    model_name: str = "depthwise_shift",
) -> onnx.ModelProto:
    radius = max(abs(dx), abs(dy))
    kernel_size = radius * 2 + 1
    center = radius

    weights = np.zeros((10, 1, kernel_size, kernel_size), dtype=np.float32)
    row = center - dy
    col = center - dx
    if row < 0 or row >= kernel_size or col < 0 or col >= kernel_size:
        raise ValueError("Shift is out of supported kernel bounds")

    for channel in range(10):
        weights[channel, 0, row, col] = 1.0

    initializers = [numpy_helper.from_array(weights, name="W_shift")]

    conv_node = helper.make_node(
        "Conv",
        ["input", "W_shift"],
        ["output"],
        name="depthwise_shift",
        kernel_shape=[kernel_size, kernel_size],
        strides=[1, 1],
        pads=[radius, radius, radius, radius],
        group=10,
    )

    graph = helper.make_graph(
        nodes=[conv_node],
        name=model_name,
        inputs=[_tensor_value_info("input")],
        outputs=[_tensor_value_info("output")],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-native-builder",
        opset_imports=[helper.make_operatorsetid("", DEFAULT_OPSET)],
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model)
    return model


def build_shift_then_color_map_model(
    dx: int,
    dy: int,
    weight_matrix: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    model_name: str = "shift_then_color_map",
) -> onnx.ModelProto:
    weights_2d = _validate_color_weight_matrix(weight_matrix)
    shift_radius = max(abs(dx), abs(dy))
    shift_kernel = shift_radius * 2 + 1
    center = shift_radius

    shift_weights = np.zeros((10, 1, shift_kernel, shift_kernel), dtype=np.float32)
    row = center - dy
    col = center - dx
    if row < 0 or row >= shift_kernel or col < 0 or col >= shift_kernel:
        raise ValueError("Shift is out of supported kernel bounds")

    for channel in range(10):
        shift_weights[channel, 0, row, col] = 1.0

    color_weights = weights_2d.reshape(10, 10, 1, 1).astype(np.float32)

    initializers = [
        numpy_helper.from_array(shift_weights, name="W_shift"),
        numpy_helper.from_array(color_weights, name="W_color"),
    ]

    color_inputs = ["shifted", "W_color"]
    if bias is not None:
        bias_array = np.asarray(bias, dtype=np.float32)
        if bias_array.shape != (10,):
            raise ValueError("bias must have shape [10]")
        initializers.append(numpy_helper.from_array(bias_array, name="B_color"))
        color_inputs.append("B_color")

    shift_node = helper.make_node(
        "Conv",
        ["input", "W_shift"],
        ["shifted"],
        name="depthwise_shift",
        kernel_shape=[shift_kernel, shift_kernel],
        strides=[1, 1],
        pads=[shift_radius, shift_radius, shift_radius, shift_radius],
        group=10,
    )

    color_node = helper.make_node(
        "Conv",
        color_inputs,
        ["output"],
        name="pointwise_color_map",
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )

    graph = helper.make_graph(
        nodes=[shift_node, color_node],
        name=model_name,
        inputs=[_tensor_value_info("input")],
        outputs=[_tensor_value_info("output")],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-native-builder",
        opset_imports=[helper.make_operatorsetid("", DEFAULT_OPSET)],
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model)
    return model


def build_maxpool_then_color_map_model(
    kernel_size: int,
    weight_matrix: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int | None = None,
    model_name: str = "maxpool_then_color_map",
) -> onnx.ModelProto:
    if kernel_size % 2 == 0:
        raise ValueError("Only odd kernel sizes are supported")
    if stride != 1:
        raise ValueError("Only stride=1 is supported for static 30x30 output")

    pad = kernel_size // 2 if padding is None else padding
    if pad != kernel_size // 2:
        raise ValueError("Padding must preserve 30x30 output shape")

    weights_2d = _validate_color_weight_matrix(weight_matrix)
    color_weights = weights_2d.reshape(10, 10, 1, 1).astype(np.float32)

    initializers = [numpy_helper.from_array(color_weights, name="W_color")]
    color_inputs = ["pooled", "W_color"]

    if bias is not None:
        bias_array = np.asarray(bias, dtype=np.float32)
        if bias_array.shape != (10,):
            raise ValueError("bias must have shape [10]")
        initializers.append(numpy_helper.from_array(bias_array, name="B_color"))
        color_inputs.append("B_color")

    pool_node = helper.make_node(
        "MaxPool",
        ["input"],
        ["pooled"],
        name="maxpool",
        kernel_shape=[kernel_size, kernel_size],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad],
    )

    color_node = helper.make_node(
        "Conv",
        color_inputs,
        ["output"],
        name="pointwise_color_map",
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )

    graph = helper.make_graph(
        nodes=[pool_node, color_node],
        name=model_name,
        inputs=[_tensor_value_info("input")],
        outputs=[_tensor_value_info("output")],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-native-builder",
        opset_imports=[helper.make_operatorsetid("", DEFAULT_OPSET)],
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model)
    return model


def build_minpool_then_color_map_model(
    kernel_size: int,
    weight_matrix: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int | None = None,
    model_name: str = "minpool_then_color_map",
) -> onnx.ModelProto:
    if kernel_size % 2 == 0:
        raise ValueError("Only odd kernel sizes are supported")
    if stride != 1:
        raise ValueError("Only stride=1 is supported for static 30x30 output")

    pad = kernel_size // 2 if padding is None else padding
    if pad != kernel_size // 2:
        raise ValueError("Padding must preserve 30x30 output shape")

    weights_2d = _validate_color_weight_matrix(weight_matrix)
    color_weights = weights_2d.reshape(10, 10, 1, 1).astype(np.float32)

    initializers = [numpy_helper.from_array(color_weights, name="W_color")]
    color_inputs = ["minpooled", "W_color"]

    if bias is not None:
        bias_array = np.asarray(bias, dtype=np.float32)
        if bias_array.shape != (10,):
            raise ValueError("bias must have shape [10]")
        initializers.append(numpy_helper.from_array(bias_array, name="B_color"))
        color_inputs.append("B_color")

    neg_node = helper.make_node("Neg", ["input"], ["neg_input"], name="minpool_neg")
    pool_node = helper.make_node(
        "MaxPool",
        ["neg_input"],
        ["neg_pooled"],
        name="minpool_max",
        kernel_shape=[kernel_size, kernel_size],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad],
    )
    out_node = helper.make_node("Neg", ["neg_pooled"], ["minpooled"], name="minpool_out")
    color_node = helper.make_node(
        "Conv",
        color_inputs,
        ["output"],
        name="pointwise_color_map",
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )

    graph = helper.make_graph(
        nodes=[neg_node, pool_node, out_node, color_node],
        name=model_name,
        inputs=[_tensor_value_info("input")],
        outputs=[_tensor_value_info("output")],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-native-builder",
        opset_imports=[helper.make_operatorsetid("", DEFAULT_OPSET)],
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model)
    return model


def build_flip_model(
    *,
    axis: int,
    model_name: str = "flip",
) -> onnx.ModelProto:
    if axis not in (2, 3):
        raise ValueError("axis must be 2 (height) or 3 (width)")

    indices = np.arange(29, -1, -1, dtype=np.int64)
    initializers = [numpy_helper.from_array(indices, name="flip_indices")]

    gather_node = helper.make_node(
        "Gather",
        ["input", "flip_indices"],
        ["output"],
        name="flip_gather",
        axis=axis,
    )

    graph = helper.make_graph(
        nodes=[gather_node],
        name=model_name,
        inputs=[_tensor_value_info("input")],
        outputs=[_tensor_value_info("output")],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-native-builder",
        opset_imports=[helper.make_operatorsetid("", DEFAULT_OPSET)],
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model)
    return model


def build_transpose_hw_model(
    *,
    model_name: str = "transpose_hw",
) -> onnx.ModelProto:
    transpose_node = helper.make_node(
        "Transpose",
        ["input"],
        ["output"],
        name="transpose_hw",
        perm=[0, 1, 3, 2],
    )

    graph = helper.make_graph(
        nodes=[transpose_node],
        name=model_name,
        inputs=[_tensor_value_info("input")],
        outputs=[_tensor_value_info("output")],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-native-builder",
        opset_imports=[helper.make_operatorsetid("", DEFAULT_OPSET)],
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model)
    return model


def save_model(model: onnx.ModelProto, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path.as_posix())
    return path
