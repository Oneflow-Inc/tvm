import os
import copy
import warnings

import numpy as np
import tvm
from tvm.ir import IRModule
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

FLOW_2_STR_DTYPE = {
    2: "float32",
    3: "float64",
    6: "int64",
    5: "int32",
    4: "int8",
    7: "uint8",
    9: "float16"
}

NUMPY_2_STR_DTYPE = {
    "<class \'numpy.float32\'>": "float32",
    "<class \'numpy.float64\'>": "float64",
    "<class \'numpy.int64\'>": "int64",
    "<class \'numpy.int32\'>": "int32",
    "<class \'numpy.int8\'>": "int8",
    "<class \'numpy.uint8\'>": "uint8",
    "<class \'numpy.float16\'>": "float16"
}

_identity_list = []


def is_input_op(node):
    # 用来判断该节点的op是否为input
    return node.WhichOneof("op_type") == "input_conf"


def is_user_op(node):
    # 用来判断该节点的op是否为input
    return node.WhichOneof("op_type") == "user_conf"


def is_output_op(node):
    # 用来判断该节点的op是否为input
    return node.WhichOneof("op_type") == "return_conf"


def get_node_info(node):
    # 获取名字
    name = node.name
    # 获取形状，转为list
    shape = list(node.input_conf.blob_conf.shape.dim)
    # 获取数据类型
    dtype = node.input_conf.blob_conf.data_type
    if dtype in list(FLOW_2_STR_DTYPE.keys()):
        data_type = FLOW_2_STR_DTYPE[dtype]
    else:
        raise IndexError('Please check the data type of your node: %s' % name)
    return name, shape, data_type


def parse_attr(attr):
    # 解析node_attr
    # TODO: 可能需要转换数据类型
    attrs = {}
    for a in attr:
        attr_str = str(attr[a])

        if attr_str[0:7] == "at_list":
            attr_str_ = attr_str.split(" ")[0]

            if attr_str_ == "at_list_float":
                attrs[a] = list(attr[a].at_list_float.val)
            elif attr_str_ == "at_list_int32":
                attrs[a] = list(attr[a].at_list_int32.val)
            elif attr_str_ == "at_list_int64":
                attrs[a] = list(attr[a].at_list_int64.val)

        elif attr_str.split(":")[0] == "at_string":
            attrs[a] = attr[a].at_string

        elif attr_str.split(" ")[0] == "at_shape":
            attrs[a] = list(attr[a].at_shape.dim)

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


def get_convert_map():
    # 获取已实现的oneflow2relay op
    return {}


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
    dtype : str or dict of str to str
        The input types to the graph
    """
    def __init__(self, shape, dtype) -> None:
        self._nodes = {}
        self._params = {}
        self._inputs = {}
        self._renames = {}
        self._num_input = 0
        self._num_param = 0
        self._shape = shape.copy() if shape else {}
        self._input_names = []
        self._dtype = dtype


    def _parse_input(self, node_input):
        # TODO: 需要跟onnx.py结果作比较
        inputs = oneflow_input()
        for i in node_input:
            if i != "":
                inputs[i] = self._nodes[self._renames.get(i, i)]
            else:
                inputs[i] = None
        return inputs


    def _parse_output(self, op_name, node_output):
        # TODO: 不是很理解这里在干什么，并且需要跟onnx.py结果作比较
        if op_name.lower() == "dropout":
            if len(node_output) == 1:
                return node_output
            outputs = node_output[:-1]
        return outputs


    def from_onnx(self, job, model_dir_path):
        """
        Parameters
        ----------
        job : job function
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
        import oneflow
        import oneflow.experimental as flow

        flow.enable_eager_execution()
        oneflow.config.enable_legacy_model_io(False)

        # 获取模型各层名字、大小、参数矩阵、路径、参数数据类型
        model = oneflow.checkpoint.get(model_dir_path)

        # 获取job函数的所有可能信息，用于得到用户的job，导出计算图
        job_set = flow.get_job_set()

        # 获取计算图参数的相关信息并存入类中
        for layer in model:
            # str(layer)是该层的名字，如：conv1-weight、dense2-bias
            # 这一步用于排除训练完毕的标识
            if str(layer) is 'System-Train-TrainStep-train_job':
                break

            name = str(layer)                # 获取该层的名字，str
            array = model[layer].numpy()     # 获取该层参数，ndarray
            shape = model[layer].shape       # 获取该层参数的形状，tuple
            dtype = NUMPY_2_STR_DTYPE[str(array.dtype)]
            # 获取该层参数的数据类型，str，"int32"等

            # file_path = model[layer].file_path    # 模型参数所在的路径，str

            self._params[name] = array
            self._nodes[name] = new_var(
                name,
                shape=shape,
                dtype=dtype
            )

        # 获取中间计算过程的oneflow2relay op，为后面转换中间计算过程的op做准备
        convert_map = get_convert_map()
        unsupported_ops = set()

        # 获取计算图输入部分的节点
        for j in job_set.job:
            if j.job_conf.job_name == job.__name__:
                for node in job.net.op:
                    # 开始转换input_node
                    if is_input_op(node):
                        # 判断该结点是不是input节点，是，则进入该if分支
                        # TODO: node_dtype的数据类型是什么?
                        node_name, node_shape, node_dtype = get_node_info(node)
                        # 判断该节点是否是模型参数，在上一段代码中已经存入了模型参数
                        if node_name in self._params:
                            # 若进入该分支，代表该节点为模型参数
                            self._num_param += 1
                            # 从params中剔除，加进nodes中
                            self._params[node_name] = self._params.pop(node_name)
                            self._nodes[node_name] = new_var(
                                node_name,
                                shape=node_shape,
                                dtype=node_dtype
                            )
                        elif node_name in self._nodes:
                            # 若进入该分支，代表该节点已经被记录
                            continue
                        else:
                            # 若进入该分支，代表该节点为input
                            self._num_input += 1
                            self._input_names.append(node_name)
                            if node_name in self._shape:
                                # 若进入该分支，直接提取该节点之前存好的shape
                                # 原因：shape由用户做过初始化
                                name_shape = self._shape[node_name]
                            else:
                                # 若进入该分支，则需要自己或许并存入shape
                                warnings.warn('Input %s has unknown dimension shapes' % node_name)
                            
                            if isinstance(self._dtype, dict):
                                # 若进入该分支，直接提取该节点之前存好的dtype
                                # 原因：dtype由用户做过初始化
                                dtype = self._dtype[node_name] if node_name in self._dtype else node_dtype
                            else:
                                dtype = node_dtype

                            self._nodes[node_name] = new_var(
                                node_name,
                                shape=node_shape,
                                dtype=dtype
                            )
                        
                        self._inputs[node_name] = self._nodes[node_name]

        # TODO: 可能可以不需要重开一个for，先这样写着
        for j in job_set.job:
            if j.job_conf.job_name == job.__name__:
                for node in job.net.op:
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

                        # 构建该op的input
                        node_inputs = self._parse_input(node.user_conf.input)

                        # 构建该op的output
                        node_outputs = self._parse_output(op_name, node.user_conf.output)

                        op_attr["tvm_custom"] = {}
                        op_attr["tvm_custom"]["name"] = node.name
                        op_attr["tvm_custom"]["num_outputs"] = len(node_outputs)
                        
                        # 转换核心语句
                        op = self._convert_operator(op_name, node_inputs, op_attr)

                        # 判断网络有多少个输出，并相应做出调整
                        if not isinstance(op, _expr.TupleWrapper):
                            outputs_num = 1
                        else:
                            outputs_num = len(op)

                        if outputs_num == 1:
                            op = fold_constant(op)
                        else:
                            op = _expr.TupleWrapper(fold_constant(op.astuple()), len(op))

                        # TODO: 针对多个输出，需要做相应处理
                        if outputs_num > 1:
                            pass
                        
                        # 转换
                        if outputs_num == 1:
                            self._nodes[node_outputs[0]] = op
                        else:
                            for k, i in zip(list(node_outputs), range(len(node_outputs))):
                                self._nodes[k] = op[i]
        
        # TODO: 同上，同时也需要与onnx.py的作对比
        outputs = []
        for j in job_set.job:
            if j.job_conf.job_name == job.__name__:
                for node in job.net.op:
                    if is_output_op(node):
                        outputs.append(self._nodes[node.name])
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        # 转换为relay IR
        free_vars = analysis.free_vars(outputs)
        nodes = {v: k for k, v in self._nodes.items()}
        free_vars = [nodes[var] for var in free_vars]
        for i_name in self._params:
            if i_name in free_vars and i_name not in self._inputs:
                self._inputs[i_name] = self._nodes[i_name]
        # Create a function from our output expression and all input variables.
        func = _function.Function([v for k, v in self._inputs.items()], outputs)
        return IRModule.from_expr(func), self._params


    def _convert_operator(self, op_name, node_inputs, op_attr):
        """
        Parameters
        ----------
        op_name : str
            Operator name, such as Convolution, FullyConnected
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
            sym = convert_map[op_name](node_inputs, op_attr, self._params)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        return sym


def from_onnx(job, model_dir_path, shape=None, dtype="float32"):
    """
    Parameters
    ----------
    job : job function
    model_dir_path: str
        The path of parameter
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
        pass

    g = OneflowGraph(shape, dtype)

    # Use the graph proto as a scope so that ops can access other nodes if needed.
    mod, params = g.from_onnx(job, model_dir_path)
    return mod, params
