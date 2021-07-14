import copy
import warnings
import inspect
import itertools
import collections

import numpy as np
import tvm
from tvm.ir import tensor_type

from .. import expr as _expr
from .. import function as _function
from .. import analysis

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

__all__ = ['from_oneflow']

_identity_list = []

FLOW_PROTO_DTYPE = {
    2: oneflow.float32,
    3: oneflow.float64,
    6: oneflow.int64,
    5: oneflow.int32,
    4: oneflow.int8,
    7: oneflow.uint8,
    9: oneflow.float16,
}
# TODO 4: 或许需要做一下扩充

# Built-in supported domains
ONNX_DOMAIN = ""
AI_ONNX_ML_DOMAIN = "ai.onnx.ml"

# Default opset version for onnx domain
PREFERRED_OPSET = 10

NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]
NCHW_TO_HWCN = [2, 3, 1, 0]


def is_user_op(node):
    return node.WhichOneof("op_type") == "user_conf"


def get_op_conf(node):
    conf_type = node.WhichOneof("op_type")
    conf = getattr(node, conf_type)
    return conf


def get_op_type(node):
    if is_user_op(node):
        return node.user_conf.op_type_name
    return node.WhichOneof("op_type")[:-5]


def get_op_attr(node):
    return node.user_conf.attr.keys() if is_user_op(node) else []


def get_info(lbd):
    shape = list(lbd.shape.dim)
    # 这里的data_type是个标量，应当做一下处理
    if lbd.data_type in list(FLOW_PROTO_DTYPE.keys()):
        dtype = FLOW_PROTO_DTYPE[str(lbd.data_type)]
        return shape, dtype
    else:
        raise KeyError(str(lbd.data_type))


def _get_convert_map():
    """
    TODO 4: OneFlow被TVM支持的OP字典，
    字典的Key是Oneflow OP的类型名字，而字典的Value就是转换之后的Relay IR
    可能需要自定义OP
    查询oneflow的OP
    """
    return {
        # defs/math
        "Floor": Renamer("floor"),
        "Ceil": Renamer("ceil"),
        "Round": Renamer("round"),
        "IsNaN": Renamer("isnan"),
        "Sqrt": Renamer("sqrt"),
        "Exp": Renamer("exp"),
        "Log": Renamer("log"),
        "Acos": Renamer("acos"),
        "Acosh": Renamer("acosh"),
        "Asin": Renamer("asin"),
        "Asinh": Renamer("asinh"),
        "Atan": Renamer("atan"),
        "Atanh": Renamer("atanh"),
        "Cos": Renamer("cos"),
        "Cosh": Renamer("cosh"),
        "Sin": Renamer("sin"),
        "Sinh": Renamer("sinh"),
        "Tan": Renamer("tan"),
        "Tanh": Renamer("tanh"),
        "Pow": Renamer("power"),
        
        # defs/nn

        # defs/others
    }


def get_flow_node_attr(node, name):
    assert node.WhichOneof("op_type") == "user_conf"
    attr_msg = node.user_conf.attr[name]
    attr_type = attr_msg.WhichOneof("value")

    if attr_type == "at_shape":
        return list(getattr(attr_msg, attr_type).dim)
    elif attr_type[:7] == "at_list":
        return list(getattr(attr_msg, attr_type).val)
    else:
        return getattr(attr_msg, attr_type)


def get_inputs(node):
    # TODO 3: 观察该inputs与onnx的输出是否一致，修改
    if is_user_op(node):
        ibns = flow_op.ibn4op_type(get_op_type(node))
        if ibns is None:
            return list(
                itertools.chain(*[x.s for x in node.user_conf.input.values()])
            )
        ipts = []
        for ibn in ibns:
            for key, val in node.user_conf.input.items():
                if key == ibn:
                    assert len(val.s) == 1
                    ipts.append(val.s[0])
                    break
            else:
                raise ValueError(
                    "ibn {} of node {} (type {}) not found".format(
                        ibn, node.name, get_op_type(node)
                    )
                )
        return ipts
    else:
        conf = get_op_conf(node)
        # it cannot cover all legacy op but it's enough
        if hasattr(conf, "in"):
            op_in = getattr(conf, "in")
            if isinstance(op_in, str):
                return [op_in]
            else:
                return op_in
        else:
            return []


def get_outputs(node):
    if is_user_op(node):
        obns = flow_op.obn4op_type(get_op_type(node))
        if obns is None:
            assert all([len(x.s) == 1 for x in node.user_conf.output.values()])
            return [x.s[0] for x in node.user_conf.output.values()]
        outputs = []
        for obn in obns:
            for key, val in node.user_conf.output.items():
                if key == obn:
                    assert len(val.s) == 1
                    outputs.append(val.s[0])
                    break
            else:
                raise ValueError(
                    "obn {} of node {} (type {}) not found".format(
                        obn, node.name, get_op_type(node)
                    )
                )
    else:
        conf = get_op_conf(node)
        # it cannot cover all legacy op but it's enough
        if hasattr(conf, "out"):
            out = getattr(conf, "out")
            if isinstance(out, str):
                outputs = [out]
            else:
                outputs = out
        else:
            outputs = []
        outputs = ["{}/{}".format(node.name, output) for output in outputs]
    return outputs


class flow_op:
    """TODO 3: 阅读Class to implement the decorator to register handlers that map oneflow to onnx."""

    _OPSETS = collections.OrderedDict()
    _MAPPING = None
    _OP_TYPE_2_IBN = {}
    _OP_TYPE_2_OBN = {}
    name_set = set()

    def __init__(
        self,
        name,
        onnx_op=None,
        domain=ONNX_DOMAIN,
        flow_ibns=None,
        flow_obns=None,
        **kwargs
    ):
        """Called decorator from decorator.
        :param name: The name of the oneflow operator.
        :param domain: The domain the operator belongs to, defaults to onnx.
        :param kwargs: Dictionary that are passed to the handler. A key 'onnx_op' will change the operator name.
        """
        if not isinstance(name, list):
            name = [name]
        self.name = name
        if not isinstance(onnx_op, list):
            onnx_op = [onnx_op] * len(name)
        self.onnx_op = onnx_op
        self.domain = domain
        self.kwargs = kwargs
        self.flow_ibns = flow_ibns
        self.flow_obns = flow_obns

    def __call__(self, func):
        opset = flow_op._OPSETS.get(self.domain)
        if not opset:
            opset = []
            flow_op._OPSETS[self.domain] = opset
        for k, v in inspect.getmembers(func, inspect.ismethod):
            if k.startswith("Version_"):
                version = int(k.replace("Version_", ""))
                while version >= len(opset):
                    opset.append({})
                opset_dict = opset[version]
                for i, name in enumerate(self.name):
                    opset_dict[name] = (v, self.onnx_op[i], self.kwargs)
                    flow_op.name_set.add(name)
                    if self.flow_ibns is not None:
                        flow_op._OP_TYPE_2_IBN[name] = self.flow_ibns
                    if self.flow_obns is not None:
                        flow_op._OP_TYPE_2_OBN[name] = self.flow_obns
        return func

    @staticmethod
    def ibn4op_type(op_type):
        return flow_op._OP_TYPE_2_IBN.get(op_type, None)

    @staticmethod
    def obn4op_type(op_type):
        return flow_op._OP_TYPE_2_OBN.get(op_type, None)

    @staticmethod
    def get_opsets():
        return flow_op._OPSETS

    @staticmethod
    def CreateMapping(max_onnx_opset_version, extra_opsets):
        """Create the final mapping dictionary by stacking domains and opset versions.
        :param max_onnx_opset_version: The highest onnx opset the resulting graph may use.
        :param extra_opsets: Extra opsets the resulting graph may use.
        """
        mapping = {ONNX_DOMAIN: max_onnx_opset_version}
        if extra_opsets:
            for extra_opset in extra_opsets:
                mapping[extra_opset.domain] = extra_opset.version
        ops_mapping = {}
        for domain, opsets in flow_op.get_opsets().items():
            for target_opset, op_map in enumerate(opsets):
                m = mapping.get(domain)
                if m:
                    if target_opset <= m and op_map:
                        ops_mapping.update(op_map)

        flow_op._MAPPING = ops_mapping
        return ops_mapping


class oneflow_input:
    """TODO 3: Dual purpose list or dictionary access object.似乎可以复用"""

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


class OneflowOpConverter:
    """
    A helper class for holding OneFlow op converters
    Parameters
    ----------
    shape : dict of str to tuple, optional
        The input shape to the graph
    dtype : str or dict of str to str
        The input types to the graph
    """
    def __init__(self, shape, data_type):
        self._nodes = {}
        self._params = {}
        self._inputs = {}
        self._renames = {}
        self._num_input = 0
        self._num_param = 0
        self._shape = shape.copy() if shape else {}
        self._input_names = []
        self._dtype = data_type
    
    def from_oneflow(self, graph, get_output_expr=False):
        """
        Construct Relay expression from OneFlow graph

        Parameters
        ----------
        graph : graph
            The loaded oneflow graph
        
        Returns
        -------
        mod : tvm.IRModule
            The returned relay module
        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """

        # 初始化，将模型中出现过的模型参数保存起来
        # TODO 3: 在job中找到模型参数，如果不行，导入model参数
        # 下面是直接从model导入的模型参数
        for model_key in list(model.keys()):
            layer = model[model_key]
            tensor_name = layer.name

            if not tensor_name.strip():
                raise ValueError("Tensor's name is required.")
            array = layer.numpy()
            
            self._params[tensor_name] = array
            self._nodes[tensor_name] = new_var(
                tensor_name,
                shape=self._params[tensor_name].shape,
                dtype=self._params[tensor_name].dtype,
            )

        # 解析oneflow graph的输入
        for lbn in graph.helper.lbn2logical_blob_desc:
            lbd = graph.helper.lbn2logical_blob_desc[lbn]
            # i是输入，我们需要得到shape、data_type
            # 注意，i_name = lbn
            i_shape, d_type = get_info(lbd)
            # 判断i这个输入是权重参数还是输入
            if lbn in self._params:
                # i是参数
                self._num_param += 1
                self._params[lbn] = self._params.pop(lbn)
                self._nodes[lbn] = new_var(
                    lbn, 
                    shape=self._params[lbn].shape, 
                    dtype=self._params[lbn].dtype
                )
            elif lbn in self._nodes:
                # 输入节点已经在Relay IR中了就不用处理了
                continue
            else:
                # 真正的输入节点
                self._num_input += 1
                self._input_names.append(lbn)
                if lbn in self._shape:
                    i_shape = self._shape[lbn]
                # else:
                #     # if "?" in str(i_shape):
                #     #     warning_msg = (
                #     #         "Input %s has unknown dimension shapes: %s. "
                #     #         "Specifying static values may improve performance"
                #     #         % (i_name, str(i_shape_name))
                #     #     )
                #     #     warnings.warn(warning_msg)
                #     # TODO -1: 需要根据oneflow的指定这段warning
                if isinstance(self._dtype, dict):
                    # 需要确保输入的shape为字典
                    dtype = self._dtype[lbn] if lbn in self._dtype else d_type
                else:
                    dtype = d_type
                self._nodes[lbn] = new_var(lbn, shape=i_shape, dtype=dtype)
            self._inputs[lbn] = self._nodes[lbn]
        
        # 获取不支持的算子
        convert_map = _get_convert_map() # 这个函数返回所有支持的算子，字典
        unsupported_ops = set()

        # TODO -1:该if语句需要根据oneflow op.name的真实情况做调整
        for node in graph.net.op:
            op_name = get_op_type(node)
            if(
                op_name not in convert_map
                and op_name not in _identity_list
            ):
                unsupported_ops.add(op_name)

        # 输出不支持的算子
        if unsupported_ops:
            msg = "The following operators are not supported for frontend OneFlow: "
            msg += ", ".join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

        # 正式转换
        for node in graph.net.op:
            op_name = get_op_type(node)

            # 解析attribute参数，转换为keys为names / values为attr 的字典
            attr = self._parse_attr(get_op_attr(node))

            # TODO 3: 创建oneflow输入对象，查看onnx.py 中与上get_inputs()的区别
            inputs = oneflow_input()

            # 填充oneflow输入对象
            # 查看onnx里面node.input和生成的inputs是什么
            for i in node.input:
                if i != "":
                    # TODO 2: self._renames.get(i, i)用来获取ONNX Graph每个节点的输入 ???
                    inputs[i] = self._nodes[self._renames.get(i, i)]
                else:
                    inputs[i] = None

            i_name = self._parse_value_proto(node)
            node_output = self._fix_outputs(op_name, node.output)

            attr["tvm_custom"] = {}
            attr["tvm_custom"]["name"] = i_name
            attr["tvm_custom"]["num_outputs"] = len(node_output)

            op = self._convert_operator(op_name, inputs, attr)

            # 判断op的输出
            if not isinstance(op, _expr.TupleWrapper):
                outputs_num = 1
            else:
                outputs_num = len(op)

            if outputs_num == 1:
                op = fold_constant(op)
            else:
                op = _expr.TupleWrapper(fold_constant(op.astuple()), len(op))

            if outputs_num > 1:
                pass   # TODO 4: 可选输出
            elif outputs_num == 1:
                self._nodes[node_output[0]] = fold_constant(op)
            else:
                op = _expr.TupleWrapper(fold_constant(op.astuple()), len(op))
                for k, i in zip(list(node_output), range(len(node_output))):
                    self._nodes[k] = op[i]
        
        # TODO 3: 解析输出，同上input
        outputs = [self._nodes[self._parse_value_proto(i)] for i in graph.output]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        if get_output_expr:
            return outputs
        else:
            # 按照graph顺序
            free_vars = analysis.free_vars(outputs)
            nodes = {v: k for k, v in self._nodes.items()}
            free_vars = [nodes[var] for var in free_vars]
            for i_name in self._params:
                if i_name in free_vars and i_name not in self._inputs:
                    self._inputs[i_name] = self._nodes[i_name]
            # 根据我们的输出表达式和所有输入变量创建一个函数。 
            func = _function.Function([v for k, v in self._inputs.items()], outputs)
            # 把这个函数用IRModule包起来返回，并同时返回权重参数
            return tvm.ir.IRModule.from_expr(func), self._params


    def _convert_operator(self, op_name, inputs, attrs):
        """Convert OneFlow operator into a Relay operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.
        Parameters
        ----------
        op_name : str
            Operator name, such as Convolution, FullyConnected
        inputs : list of tvm.relay.function.Function
            List of inputs.
        attrs : dict
            Dict of operator attributes

        Returns
        -------
        sym : tvm.relay.function.Function
            Converted relay function
        """
        convert_map = _get_convert_map()
        if op_name in _identity_list:
            # TODO 2: 观察这里的onnx.py中op_name
            sym = get_relay_op(op_name)(*inputs, **attrs)
        elif op_name in convert_map:
            sym = convert_map[op_name](inputs, attrs, self._params)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        return sym
    
    def _parse_attr(self, attr):
        """TODO -1: 如何更简洁，将node.attribute转换为以op_name为keys的字典"""
        attrs = {}
        for a in attr:
            attr_str = str(attr[a])

            if attr_str[0:7] == "at_list":
                attr_str_ = attr_str.split(" ")[0]

                if attr_str_ == "at_list_float":
                    attrs[a] = attr[a].at_list_float
                elif attr_str_ == "at_list_int32":
                    attrs[a] = attr[a].at_list_int32
                elif attr_str_ == "at_list_int64":
                    attrs[a] = attr[a].at_list_int64

            elif attr_str.split(":")[0] == "at_string":
                attrs[a] = attr[a].at_string

            else:
                attr_str_ = attr_str.split(":")[0]
                if attr_str_ == "at_bool":
                    attrs[a] = attr[a].at_bool
                elif attr_str_ == "at_double":
                    attrs[a] = attr[a].at_double
                elif attr_str_ == "at_float":
                    attrs[a] = attr[a].at_float
                elif attr_str_ == "at_int32":
                    attrs[a] = attr[a].at_int32
                elif attr_str_ == "at_int64":
                    attrs[a] = attr[a].at_int64
        return attrs

    def _parse_value_proto(self, value_node):
        """返回node的名字"""
        try:
            name = value_node.name
        except AttributeError:
            raise AttributeError
        return name

    def _fix_outputs(self, op_name, node_output):
        """针对dropout做输出调整，但是不清楚加入该函数原因"""
        if op_name == "Dropout":
            if len(node_output) == 1:
                return node_output
            # TODO(zhreshold): support dropout mask?
            outputs = node_output[:-1]
        return outputs


def from_onnx(model, shape=None, opset=1, d_type="float32"):
    """
    Parameters
    ----------
    model : dict
        OneFlow Model
    shape : dict of str to tuple, optional
        The input shape to the graph
    dtype : str or dict of str to str
        The input types to the graph

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation
    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    """
    import os

    try:
        import oneflow
        import oneflow.experimental as flow
        import oneflow.experimental.nn as nn

        flow.env.init()
        flow.enable_eager_execution()

    except ImportError as e:
        raise(str(e))

    assert type(model)==dict, "the type of oneflow model should be a dict"

    g = OneflowOpConverter(shape, d_type)
    mod, params = g.from_oneflow(model, opset)

    return mod, params
