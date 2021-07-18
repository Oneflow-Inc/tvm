import os
import copy
import warnings

import numpy as np
import tvm
from tvm.ir import IRModule
from tvm.relay.analysis.analysis import check_basic_block_normal_form
from tvm.topi.utils import get_const_tuple

from ... import nd as _nd
from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import loops as _loops
from .. import op as _op
from .. import qnn as _qnn
from .. import ty as _ty
from .. import vision as _vision
from .common import (
    AttrCvt,
    Renamer,
    fold_constant,
    get_name,
    get_relay_op,
    infer_channels,
    infer_shape,
    infer_type,
    infer_value,
    new_var,
)

__all__ = ["from_oneflow"]

FLOW_2_NP_DTYPE = {
    2: np.float32,
    3: np.float64,
    6: np.int64,
    5: np.int32,
    4: np.int8,
    7: np.uint8,
    9: np.float16
}

_identity_list = []


def is_input_op(node):
    # 用来判断该节点的op是否为input_conf
    return node.WhichOneof("op_type") == "input_conf"


def is_user_op(node):
    # 用来判断该节点的op是否为user_conf
    return node.WhichOneof("op_type") == "user_conf"


def is_output_op(node):
    # 用来判断该节点的op是否为return_conf
    return node.WhichOneof("op_type") == "return_conf"


def is_param_op(node):
    # 用来判断该节点的op是否为variable_conf
    return node.WhichOneof("op_type") == "variable_conf"


def get_node_info(node):
    """
    获取node基本信息: shape、data_type
    """
    # 获取形状，转为list->tuple
    shape = tuple(node.input_conf.blob_conf.shape.dim)
    # 获取数据类型
    dtype = node.input_conf.blob_conf.data_type
    if dtype in list(FLOW_2_NP_DTYPE.keys()):
        data_type = FLOW_2_NP_DTYPE[dtype]
    else:
        raise IndexError('Please check the data type of your node: %s' % node.name)
    return shape, data_type


def parse_attr(attr):
    # 解析node_attr
    # TODO: 可能数据类型有遗漏
    attrs = {}
    for a in attr:
        attr_str = str(attr[a])

        if attr_str[0:7] == "at_list":
            attr_str_ = attr_str.split(" ")[0]

            if attr_str_ == "at_list_float":
                attrs[a] = np.array(list(attr[a].at_list_float.val)).astype(np.float32)
            elif attr_str_ == "at_list_int32":
                attrs[a] = np.array(list(attr[a].at_list_int32.val)).astype(np.int32)
            elif attr_str_ == "at_list_int64":
                attrs[a] = np.array(list(attr[a].at_list_int64.val)).astype(np.int64)

        elif attr_str.split(":")[0] == "at_string":
            attrs[a] = attr[a].at_string

        elif attr_str.split(" ")[0] == "at_shape":
            attrs[a] = tuple(list(attr[a].at_shape.dim))

        else:
            attr_str_ = attr_str.split(":")[0]
            if attr_str_ == "at_bool":
                attrs[a] = np.array(attr[a].at_bool).astype(np.bool)
            elif attr_str_ == "at_double":
                attrs[a] = np.array(attr[a].at_double).astype(np.float64)
            elif attr_str_ == "at_float":
                attrs[a] = np.array(attr[a].at_float).astype(np.float32)
            elif attr_str_ == "at_int32":
                attrs[a] = np.array(attr[a].at_int32).astype(np.int32)
            elif attr_str_ == "at_int64":
                attrs[a] = np.array(attr[a].at_int64).astype(np.int64)
    return attrs


def fix_outputs(op_name, outputs):
    if op_name.lower() == "Dropout":
        if len(outputs) == 1:
            return outputs
        # TODO(zhreshold): support dropout mask?
        outputs = outputs[:-1]
    return outputs


def shape_of(x, dtype="int64"):
    ttype = infer_type(x).checked_type
    if not _ty.is_dynamic(ttype):
        shape = list(ttype.shape)
        return _expr.const(shape, dtype)
    return _op.shape_of(x, dtype)


def get_pad_pair(input1d, kernel1d, stride1d, mode):
    """infer pad size"""
    if input1d % stride1d == 0:
        pad = max(kernel1d - stride1d, 0)
    else:
        pad = max(kernel1d - (input1d % stride1d), 0)
    pad_before = pad // 2
    pad_after = pad - pad_before
    if "lower" in mode:
        return [pad_after, pad_before]
    return [pad_before, pad_after]


def autopad(
    data,
    strides,
    kernel_shape,
    dilations,
    ndim,
    pad_type="constant",
    deconv=False,
    mode="same_upper",
    pad_value=0.0,
):
    """
    Perform autopadding with dynamic input shapes
    """
    # get attributes as constants
    strides = _op.const(np.array(strides), dtype="int64")
    dilated_kernel_shape = _op.const(
        np.array(
            [(kernel - 1) * dilation + 1 for kernel, dilation in zip(kernel_shape, dilations)]
        ),
        dtype="int64",
    )
    # get input shape
    shape = _op.strided_slice(shape_of(data, dtype="int64"), [2], [ndim])

    # set up integer constants
    zero = _op.const(0, dtype="int64")
    one = _op.const(1, dtype="int64")
    two = _op.const(2, dtype="int64")

    # Calculate total padding
    mod = _op.mod(shape, strides)

    left = _op.maximum(dilated_kernel_shape - strides, zero)
    right = _op.maximum(dilated_kernel_shape - mod, zero)

    total_pad = _op.where(_op.equal(mod, zero), left, right)
    if deconv:
        total_pad = _op.const(np.array(kernel_shape), dtype="int64") - one - total_pad

    # split total padding into before and after
    pad_before = _op.floor_divide(total_pad, two)
    pad_after = total_pad - pad_before

    # combine
    if "lower" in mode:
        pad = _op.concatenate(
            [_op.reshape(pad_after, [-1, 1]), _op.reshape(pad_before, [-1, 1])], axis=1
        )
    else:
        pad = _op.concatenate(
            [_op.reshape(pad_before, [-1, 1]), _op.reshape(pad_after, [-1, 1])], axis=1
        )

    # pad N and C with zeros
    pad = _op.concatenate([_op.const(np.zeros([2, 2], dtype="int64"), dtype="int64"), pad], axis=0)

    if isinstance(pad_value, (float, int)):
        pad_value = _op.const(pad_value)

    return _op.nn.pad(data, fold_constant(pad), pad_value, pad_type)


# OneFlow的op_name自带了2d, 如: max_pool_2d
# def dimension_picker(prefix, suffix=""):
#     """Check that dimensions are supported.(pool)"""

#     def _impl(attr):
#         kernel = attr["pool_size"]
#         if len(kernel) == 1:
#             return prefix + "1d" + suffix
#         if len(kernel) == 2:
#             return prefix + "2d" + suffix
#         if len(kernel) == 3:
#             return prefix + "3d" + suffix
#         msg = "Only 1D, 2D, and 3D kernels are supported for operator {}."
#         op_name = prefix + "1d/2d/3d"
#         raise tvm.error.OpAttributeInvalid(msg.format(op_name))

#     return _impl


def dimension_constraint():
    # TODO: 仅仅针对了pool
    def _dim_check(attrs):
        if len(attrs["pool_size"]) in [1, 2, 3]:
            return True
        return False

    return _dim_check, "Only 1d, 2d and 3d kernel supported."


class OneFlowOpConverter:
    """A helper class for holding oneflow op converters."""

    @classmethod
    def get_converter(cls):
        """Get converter matches given opset.
        Parameters
        ----------
        
        Returns
        -------
        converter, which should be `_impl_vx`. Number x is the biggest
            number smaller than or equal to opset belongs to all support versions.
        """
        # TODO: version用来控制是用哪个函数
        version = 1
        if hasattr(cls, "_impl_v{}".format(version)):
            return getattr(cls, "_impl_v{}".format(version))
        raise NotImplementedError(
            "version {} of {} not implemented".format(version, cls.__name__)
        )


class Pool(OneFlowOpConverter):
    """A helper class for pool op converters."""

    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        data = inputs[0]
        input_shape = infer_shape(data)
        input_dtype = infer_type(data).checked_type.dtype
        ndim = len(input_shape)
        if "padding" in attr:
            attr["padding"] = attr["padding"].decode("utf-8")
            if attr["padding"].lower() in ("same_upper", "same_lower"):
                if cls.name == "avg_pool":
                    pad_tuple = []
                    for axis in range(len(input_shape) - 2):
                        axis_shape = input_shape[2 + axis]
                        stride = attr.get("strides", [1] * ndim)[axis]
                        # TODO: oneflow里面没有kernel_shape, 应该是pool_size
                        kernel = attr["pool_size"][axis]
                        pad = get_pad_pair(axis_shape, kernel, stride, attr["padding"])
                        pad_tuple.append(pad)
                    pad_tuple = tuple([val for pair in zip(*pad_tuple) for val in pair])
                    # TODO: oneflow的没有pads
                    attr["pads"] = pad_tuple
                else:
                    # Warning: Pool does not yet support dynamic shapes,
                    # one will need to run dynamic_to_static on this model after import
                    if "int" in input_dtype:
                        pad_val = np.iinfo(np.dtype(input_dtype)).min
                    else:
                        pad_val = np.finfo(np.dtype(input_dtype)).min
                    data = autopad(
                        data,
                        attr.get("strides", [1] * (ndim - 2)),
                        attr["pool_size"],
                        [1] * ndim,
                        ndim,
                        pad_value=pad_val,
                        mode=attr["padding"].lower(),
                    )
            # TODO: 暂时没有遇到这俩个选项
            # elif attr["padding"].lower() == "valid":
            #     attr["pads"] = tuple([0 for i in range(ndim - 2)])
            # elif attr["padding"].lower() == "notset":
            #     pass
            else:
                msg = 'Value {} in attribute "padding" of operator {} is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr["padding"], cls.name))
            attr.pop("padding")

        # TODO: 找出oneflow中对应的attr
        # if "storage_order" in attr:
        #     attr["layout"] = onnx_storage_order2layout(
        #         attr["storage_order"], dims=(len(input_shape) - 2), op_name=cls.name
        #     )
        # else:
        #     attr["layout"] = onnx_default_layout(dims=(len(input_shape) - 2), op_name=cls.name)

        return AttrCvt(
            op_name=cls.name,
            transforms={
                "kernel_shape": "pool_size",
                "pads": ("padding", 0),
                "dilations": ("dilation", 1),
            },
            ignores=["storage_order"],
            custom_check=dimension_constraint(),
        )([data], attr, params)


class Conv(OneFlowOpConverter):
    # TODO: 关于2d的判断
    """Operator converter for Conv."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # Use shape of input to determine convolution type.
        data = inputs[0]
        kernel = inputs[1]
        input_shape = infer_shape(data)
        ndim = len(input_shape)

        kernel_type = infer_type(inputs[1])
        kernel_shapes = [get_const_tuple(kernel_type.checked_type.shape)]

        if "kernel_size" not in attr:
            attr["kernel_size"] = kernel_shapes[0][2:]

        # TODO: 暂时没有在conv见到类似pool中的对应auto_pad的attr
        if "padding" in attr:
            attr["padding"] = attr["padding"].decode("utf-8")
            if attr["padding"].lower() in ("same_upper", "same_lower"):
                # Warning: Convolution does not yet support dynamic shapes,
                # one will need to run dynamic_to_static on this model after import
                data = autopad(
                    data,
                    attr.get("strides", [1] * (ndim - 2)),
                    attr["kernel_shape"],
                    # TODO: 不清楚是否对应
                    attr.get("dilation_rate", [1] * (ndim - 2)),
                    ndim,
                    mode=attr["padding"],
                )
            # elif attr["auto_pad"].lower() == "vaild":
            #     attr["pads"] = [0 for i in range(ndim - 2)]
            # elif attr["auto_pad"].lower() == "notset":
            #     pass
            else:
                msg = 'Value {} in attribute "auto_pad" of operator Conv is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr["padding"]))
            attr.pop("padding")

        group_conv1d = False
        if cls.name == "conv1d" and attr.get("groups") != 1:
            group_conv1d = True
            # Expand input from NCW to NCHW
            data = _op.expand_dims(data, axis=2)
            # Expand kernel from OIW to OIHW
            kernel = _op.expand_dims(kernel, axis=2)
            # Add new value to kernel_shape, strices, dilation, pads, if needed
            attr["kernel_size"] = [1] + list(attr["kernel_size"])
            if "strides" in attr:
                attr["strides"] = [1] + list(attr["strides"])
            # TODO: 还没有找到对应的
            # if "dilations" in attr:
            #     attr["dilations"] = [1] + list(attr["dilations"])
            # if "pads" in attr:
            #     attr["pads"] = [0, attr["pads"][0], 0, attr["pads"][1]]

        out = AttrCvt(
            op_name=cls.name,
            transforms={
                "kernel_shape": "kernel_size",
                "dilations": ("dilation", 1),
                "pads": ("padding", 0),
                "group": ("groups", 1),
            },
            custom_check=dimension_constraint(),
        )([data, kernel], attr, params)

        # If this was a group_conv1d, squish output back to NCW.
        if group_conv1d:
            out = _op.squeeze(out, axis=[2])

        use_bias = len(inputs) == 3
        if use_bias:
            out = _op.nn.bias_add(out, inputs[2])
        return out


class BatchNorm(OneFlowOpConverter):
    """Operator converter for BatchNorm."""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        out = AttrCvt(
            op_name="batch_norm", ignores=["spatial", "is_test", "consumed_inputs", "momentum"]
        )(inputs, attrs, params)
        return out[0]


class InstanceNorm(OneFlowOpConverter):
    """Operator converter for InstanceNorm."""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return AttrCvt(op_name="instance_norm")(inputs, attrs, params)


class MatMul(OneFlowOpConverter):
    # TODO
    """Operator converter for MatMul."""


class Add(OneFlowOpConverter):
    """Operator converter for Add."""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) == 2, "op {} take 2 inputs, {} given".format(cls.name, len(inputs))
        
        return get_relay_op("add")(*inputs)


class MaxPool(Pool):
    """Operator converter for MaxPool"""

    name = "max_pool"


class AveragePool(Pool):
    """Operator converter for AveragePool."""

    name = "avg_pool"


class Reshape(OneFlowOpConverter):
    """Operator converter for Reshape."""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return _op.reshape(inputs[0], attrs["shape"])


def get_convert_map():
    # TODO: 记录实现的oneflow2relay op
    return {
        # defs/math
        "bias_add": Add.get_converter(),
        "log": Renamer("log"),
        "acos": Renamer("acos"),
        "acosh": Renamer("acosh"),
        "asin": Renamer("asin"),
        "asinh": Renamer("asinh"),
        "atan": Renamer("atan"),
        "atanh": Renamer("atanh"),
        "cos": Renamer("cos"),
        "cosh": Renamer("cosh"),
        "sin": Renamer("sin"),
        "sinh": Renamer("sinh"),
        "tan": Renamer("tan"),
        "tanh": Renamer("tanh"),
        "pow": Renamer("power"),
        "exp": Renamer("exp"),
        # defs/activation
        "sigmoid": Renamer("sigmoid"),
        "relu": Renamer("relu"),
        # defs/nn
        "conv2d": Conv.get_converter(),
        "max_pool_2d": MaxPool.get_converter(),
        "dropout": AttrCvt("dropout", {"ratio": "rate"}, ignores=["is_test"]),
        # defs/tensor
        "matmul": MatMul.get_converter(),
        # defs/others
        # TODO: softmax交叉熵
        "sparse_softmax_cross_entropy": None,
        "reshape": Reshape.get_converter(),
    }


class oneflow_input(object):
    """
    Dual purpose list or dictionary access object
    copy from ./onnx.py
    """
    def __init__(self):
        self.input_keys = []
        self.input_dict = {}

    def __getitem__(self, item):
        if isinstance(item, int):
            if item > (len(self.input_keys) - 1):
                return None
            return self.input_dict[self.input_keys[item]]
        if isinstance(item, str):
            if item not in self.input_keys:
                return None
            return self.input_dict[item]
        if isinstance(item, slice):
            keys = self.input_keys[item]
            return [self.input_dict[key] for key in keys]

        raise ValueError("Only integer, string, and slice accesses allowed.")

    def __setitem__(self, item, value):
        if isinstance(item, int):
            self.input_dict[self.input_keys[item]] = value
        elif isinstance(item, str):
            self.input_keys.append(item)
            self.input_dict[item] = value
        else:
            raise ValueError("Only integer and string indexed writes allowed.")

    def keys(self):
        return self.input_keys

    def __len__(self):
        return len(self.input_keys)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.input_keys):
            output = self.input_dict[self.input_keys[self.n]]
            self.n += 1
            return output

        raise StopIteration


class OneflowGraph(object):
    """
    A helper class for handling Relay expression

    Parameters
    ----------
    shape : dict of str to tuple, optional
        The input shape to the graph
    dtype : dict of str to str
        The input types to the graph
    """
    def __init__(self, shape, dtype, nodes, model_dir_path) -> None:
        self._nodes = {}
        self._params = {}
        self._inputs = {}
        self._num_input = 0
        self._num_param = 0
        self._shape = {}
        self._input_names = []
        self._dtype = {}
        self._model_array = {}

        model = oneflow.checkpoint.get(model_dir_path)
        # model_array是以layer_name为key，以dict('path', 'params')为value的dict
        for layer in model:
            layer_p = {}
            layer_p['path'] = model[layer].file_path # 模型参数所在路径
            layer_p['params'] = model[layer].numpy() # 模型各层ndarray
            self._model_array[str(layer)] = layer_p

        for node_name in nodes:
            node = nodes[node_name]
            if is_user_op(node):
                for input_name in node.user_conf.input:
                    node_input_name = node.name + '-' + input_name

                    node_input_path = getattr(node.user_conf.input[input_name], 's')
                    if len(node_input_path) == 1:
                        node_input_path = os.path.join(model_dir_path, node_input_path[0])
                    else:
                        pass

                    if node_input_path in shape and node_input_name not in self._shape:
                        self._shape[node_input_name] = shape[node_input_name]
                    if node_input_path in dtype and node_input_name not in self._dtype:
                        self._dtype[node_input_name] = dtype[node_input_name]

                    if node_input_name not in self._nodes:
                        self._nodes[node_input_name] = new_var(
                            node_input_name, 
                            shape=self._shape[node_input_name],
                            dtype=self._dtype[node_input_name]
                        )

                for output_name in node.user_conf.output:
                    node_output_name = node.name + '-' + output_name

                    node_output_path = getattr(node.user_conf.output[output_name], 's')
                    if len(node_output_path) == 1:
                        node_output_path = os.path.join(model_dir_path, node_output_path[0])
                    else:
                        pass

                    if node_output_path in shape and node_output_name not in self._shape:
                        self._shape[node_output_name] = shape[node_output_name]
                    if node_output_path in dtype and node_output_name not in self._dtype:
                        self._dtype[node_output_name] = dtype[node_output_name]
                    
                    if node_output_name not in self._nodes:
                        self._nodes[node_output_name] = new_var(
                            node_output_name, 
                            shape=self._shape[node_output_name],
                            dtype=self._dtype[node_output_name]
                        )
        

    def _parse_input(self, node, node_input, model_dir_path):
        inputs = oneflow_input()

        for input_name in node_input:
            node_input_name = node.name + '-' + input_name

            node_input_path = getattr(node_input[input_name], 's')
            if len(node_input_path) == 1:
                node_input_path = os.path.join(model_dir_path, node_input_path[0])
            else:
                pass

            node_input_shape = self._shape[node_input_name]
            node_input_dtype = self._dtype[node_input_name]

            if node_input_name != "":
                if node_input_name not in self._nodes:
                    self._nodes[node_input_name] = new_var(
                        node_input_name,
                        shape=node_input_shape,
                        dtype=node_input_dtype
                    )
                inputs[node_input_name] = self._nodes[node_input_name]
            else:
                inputs[node_input_name] = None

        return inputs


    def from_oneflow(self, nodes, model_dir_path):
        """
        Parameters
        ----------
        nodes : dict, keys: node.name, value: node
            contain the graph
        model_dir_path: str
            The path of parameter

        Returns
        -------
        mod : tvm.IRModule
            The returned relay module
        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        for node_name in nodes:
            node = nodes[node_name]
            if is_user_op(node):
                node_input = node.user_conf.input
                """
                举例 lenet from document：
                input_name node_input[input_key].s[0]:

                conv1-weight conv1-weight/out                直接导入
                conv1-in Input_0/out                         可以从lbn处获得,已在__init__处理
                    得到conv1-out
                
                conv1-bias_add-b conv1-bias/out              直接导入
                conv1-bias_add-a conv1/out_0                 conv1-out结果,可以从lbn处获得
                    得到conv1-bias_add-out conv1-bias_add/out_0
                
                conv1-activation-in conv1-bias_add/out_0     conv1-bias_add-out结果,可以从lbn处获得
                ......
                """
                # 可以直接导入的，并存入self._params
                for input_name in node_input:
                    # 创建参数节点名字
                    node_init_name = node_name + '-' + input_name

                    # 获取节点参数存储位置
                    node_init_path = getattr(node_input[input_name], 's')

                    if len(node_init_path) == 1:
                        node_init_path = os.path.join(model_dir_path, node_init_path[0])
                    else:
                        # TODO: 可能有多个输入路径
                        pass

                    # 先在已提前存好的参数里面导入参数
                    for name in self._model_array:
                        layer = self._model_array[name]
                        if str(node_init_path) == layer['path']:
                            node_init_array = layer['params']
                            self._params[node_init_name] = node_init_array
                            if node_init_name not in self._nodes:
                                self._nodes[node_init_name] = new_var(
                                    node_init_name,
                                    shape=node_init_array.shape,
                                    dtype=node_init_array.dtype
                                )

        # 获取中间计算过程的oneflow2relay op，为后面转换中间计算过程的op做准备
        convert_map = get_convert_map()
        unsupported_ops = set()

        # 获取计算图输入部分的节点,这一部分的是由用户指定
        for node_name in nodes:
            node = nodes[node_name]
            # 开始转换input_node
            if is_input_op(node):
                # 判断该结点是不是input节点，是，则进入该if分支
                node_shape, node_dtype = get_node_info(node)
                if node_name in self._nodes:
                    # 若进入该分支，代表该节点已经被记录
                    continue
                else:
                    # 若进入该分支，代表该节点为input
                    self._num_input += 1
                    self._input_names.append(node_name)
                    if node_name in self._shape:
                        # 若进入该分支，直接提取该节点之前存好的shape
                        # 原因：shape做过初始化
                        node_shape = self._shape[node_name]
                    else:
                        warnings.warn('Input %s has unknown dimension shapes' % node_name)
                    
                    if isinstance(self._dtype, dict):
                        # 若进入该分支，直接提取该节点之前存好的dtype
                        # 原因：dtype做过初始化
                        dtype = self._dtype[node_name] if node_name in self._dtype else node_dtype
                    else:
                        dtype = node_dtype

                    self._nodes[node_name] = new_var(
                        node_name,
                        shape=node_shape,
                        dtype=dtype
                    )
                
                self._inputs[node_name] = self._nodes[node_name]

        self._output_path = []
        for node_name in nodes:
            node = nodes[node_name]
            if is_output_op(node):
                output_path = getattr(node.return_conf, "in")
                self._output_path.append(os.path.join(model_dir_path, output_path))

        for node_name in nodes:
            node = nodes[node_name]
            # 开始转换中间计算过程的op(user_op)
            if is_user_op(node):
                # 这里应该是op的type，而不是神经网络中用户指定的层的名字
                op_name = node.user_conf.op_type_name
                if(
                    # TODO: 这个if语句需要根据op转换的具体工作做修正
                    op_name not in convert_map
                    and op_name not in _identity_list
                ):
                    unsupported_ops.add(op_name)

            # 如果遇到不能转换的op，报错
            if unsupported_ops:
                msg = "The following operators are not supported for frontend OneFlow: "
                msg += ", ".join(unsupported_ops)
                raise tvm.error.OpNotImplemented(msg)
            
            # 开始转换
            if is_user_op(node):
                op_name = node.user_conf.op_type_name
                op_attr = parse_attr(node.user_conf.attr)

                # 构建该op的input,中间变量也要包含进来
                node_inputs = self._parse_input(
                    node,
                    node.user_conf.input,
                    model_dir_path=model_dir_path
                )
                
                # node_outputs需要的都在_parse_input中被处理了
                node_outputs = []
                for output_name in node.user_conf.output:
                    node_output_name = str(node_name) + '-' + str(output_name)

                    node_output_path = getattr(node.user_conf.output[output_name], 's')
                    if len(node_output_path) == 1:
                        node_output_path = os.path.join(model_dir_path, node_output_path[0])
                    else:
                        pass
                    node_outputs.append(node_output_name)

                node_outputs = fix_outputs(op_name, node_outputs)
                op_attr["tvm_custom"] = {}
                # TODO: onnx.py这里实际的name是空，没有看明白
                op_attr["tvm_custom"]["name"] = ''
                op_attr["tvm_custom"]["num_outputs"] = len(node_outputs)

                # 转换核心语句
                op = self._convert_operator(op_name, node_inputs, op_attr)

                # 判断网络有多少个输出，并相应做出调整
                if not isinstance(op, _expr.TupleWrapper):
                    outputs_num = 1
                else:
                    outputs_num = len(op)

                assert (len(node_outputs) == outputs_num), "Number of output mismatch {} vs {} in {}.".format(
                    len(node_outputs), outputs_num, op_name
                )

                if outputs_num == 1:
                    op = fold_constant(op)
                else:
                    op = _expr.TupleWrapper(fold_constant(op.astuple()), len(op))

                # TODO: 关于可选输出与输出的清洗，oneflow可能暂时不需要
                if outputs_num > 1:
                    pass

                # 转换
                if outputs_num == 1:
                    self._nodes[node_outputs[0]] = op
                else:
                    for k, i in zip(list(node_outputs), range(len(node_outputs))):
                        self._nodes[k] = op[i]

        outputs = []
        for input_name in node.user_conf.input:
            node_input_name = str(node_name) + '-' + str(input_name)

            node_input_path = getattr(node.user_conf.output[input_name], 's')
            if len(node_input_path) == 1:
                node_input_path = os.path.join(model_dir_path, node_input_path[0])
            else:
                pass

            if node_input_path in self._output_path:
                outputs.append(node_input_name)
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        # 转换为relay IR
        free_vars = analysis.free_vars(outputs)
        nodes = {v: k for k, v in self._nodes.items()}
        free_vars = [nodes[var] for var in free_vars]
        for i_name in self._params:
            if i_name in free_vars and i_name not in self._inputs:
                self._inputs[i_name] = self._nodes[i_name]
        # Create a function from our output expression and all input variables.
        func = _function.Function([v for _, v in self._inputs.items()], outputs)
        return IRModule.from_expr(func), self._params


    def _convert_operator(self, op_name, node_inputs, op_attr):
        """
        Parameters
        ----------
        op_name : str
            Operator name, such as conv2d、relu
        node_inputs : list of tvm.relay.function.Function
            List of inputs.
        op_attr : dict
            Dict of operator attributes

        Returns
        -------
        sym : tvm.relay.function.Function
            Converted relay function
        """
        convert_map = get_convert_map()
        if op_name in _identity_list:
            sym = get_relay_op(op_name)(*node_inputs, **op_attr)
        elif op_name in convert_map:
            # conver_map[op_name]用来获取是哪一个op
            # node_inputs: oneflow_input类
            # op_attr
            sym = convert_map[op_name](node_inputs, op_attr, self._params)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        return sym


def from_oneflow(eval_job, model_dir_path, shape=None, dtype=None):
    """
    Parameters
    ----------
    eval_job : job function, type='predict'
    model_dir_path: str
        The path of parameter
    shape : dict of str to tuple, optional
        The input shape to the graph, keys: node_param_path, values: shape
    dtype : str or dict of str to str
        The input types to the graph, keys: node_param_path, values: dtype

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation
    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    """
    try:
        import oneflow
        import oneflow.experimental as flow

        flow.enable_eager_execution()
        oneflow.config.enable_legacy_model_io(False)

        # 判断模型参数是否可以正常导入
        if 'snapshot_done' not in os.listdir(model_dir_path):
            raise IndexError("\'snapshot_name\' is not in the model path, \
            please determine whether the model has been trained")

    except ImportError:
        raise ImportError("please check that OneFlow is installed")

    # 获取job函数的所有可能信息，用于得到用户的job，导出计算图
    job_set = flow.get_job_set()

    # 创建一个以node.name为key，以node为value的字典，避免后续大量for循环查找浪费时间
    nodes = {}
    shape = {}
    dtype = {}
    for j in job_set.job:
        if j.job_conf.job_name == eval_job.__name__:
            for node in eval_job.net.op:
                nodes[node.name] = node
            # 不需要跑出来中间变量，这里都存储好了
            for lbn in eval_job.helper.lbn2logical_blob_desc:
                lbd = eval_job.helper.lbn2logical_blob_desc[lbn]
                node_path = os.path.join(model_dir_path, lbn)
                node_shape = tuple(lbd.shape.dim)
                node_dtype = lbd.data_type
                shape[node_path] = node_shape
                dtype[node_path] = FLOW_2_NP_DTYPE[node_dtype]

    g = OneflowGraph(shape, dtype, nodes, model_dir_path)

    # Use the graph proto as a scope so that ops can access other nodes if needed.
    mod, params = g.from_oneflow(nodes=nodes, model_dir_path=model_dir_path)
    return mod, params
