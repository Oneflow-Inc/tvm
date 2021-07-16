import os
import copy
import warnings
from tempfile import TemporaryFile

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
    # 获取名字
    # TODO: 写的太麻烦了
    name = node.name
    if is_input_op(node):
        # 获取形状，转为list->tuple
        shape = list(node.input_conf.blob_conf.shape.dim)
        # 获取数据类型
        dtype = node.input_conf.blob_conf.data_type
        if dtype in list(FLOW_2_NP_DTYPE.keys()):
            data_type = FLOW_2_NP_DTYPE[dtype]
        else:
            raise IndexError('Please check the data type of your node: %s' % name)
    elif is_user_op(node):
        # 获取形状，转为list->tuple
        shape = list(node.user_conf.blob_conf.shape.dim)
        # 获取数据类型
        dtype = node.user_conf.blob_conf.data_type
        if dtype in list(FLOW_2_NP_DTYPE.keys()):
            data_type = FLOW_2_NP_DTYPE[dtype]
        else:
            raise IndexError('Please check the data type of your node: %s' % name)
    elif is_output_op(node):
        # 获取形状，转为list->tuple
        shape = list(node.return_conf.blob_conf.shape.dim)
        # 获取数据类型
        dtype = node.return_conf.blob_conf.data_type
        if dtype in list(FLOW_2_NP_DTYPE.keys()):
            data_type = FLOW_2_NP_DTYPE[dtype]
        else:
            raise IndexError('Please check the data type of your node: %s' % name)
    return name, tuple(shape), data_type


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


def get_convert_map():
    # TODO: 已实现的oneflow2relay op
    return {
        # defs/math
        # defs/activation
        # defs/experimental
        # defs/nn
        # defs/tensor
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
    dtype : str or dict of str to str
        The input types to the graph
    """
    def __init__(self, shape, dtype) -> None:
        self._nodes = {}
        self._params = {}
        self._inputs = {}
        self._num_input = 0
        self._num_param = 0
        self._shape = shape.copy() if shape else {}
        self._input_names = []
        self._dtype = dtype


    def _parse_input(self, node_name, node_shape, node_dtype, node_input, model_dir_path):
        inputs = oneflow_input()

        for input_key in list(node_input.keys()):
            input_name = str(node_name) + '-' + str(input_key)

            if input_name != "":
                # 判断该节点是否是模型参数，在前面已经存入了模型参数
                if input_name in self._params:
                    # 若进入该分支，代表该节点为模型参数
                    # 从params中剔除，加进nodes中
                    self._params[input_name] = self._params.pop(input_name)
                    self._nodes[input_name] = new_var(
                        input_name,
                        shape=node_shape,
                        dtype=node_dtype
                    )
                    inputs[input_name] = self._nodes[input_name]
                elif input_name.split("_")[0] in self._params and \
                    node_input[input_key].s[0].split("/")[0] in os.listdir(model_dir_path):
                    array_path = os.path.join(model_dir_path, node_input[input_key].s[0])
                    array = np.fromfile(array_path).reshape(node_shape)
                    self._nodes[input_name] = new_var(
                        input_name,
                        shape=node_shape,
                        dtype=array.dtype
                    )
                    inputs[input_name] = self._nodes[input_name]
                else:
                    """
                    举例 lenet from document：
                    input_name node_input[input_key].s[0]:

                    conv1-weight conv1-weight/out                可以直接导入
                        该层已经在self._params存储，上方代码为重新导入
                        该node已经在第一个分支中被注册
                    conv1-in Input_0/out                         未知(应当是用户指定)
                    
                    conv1-bias_add-b conv1-bias/out              可以直接导入
                        将被命名为input_name的字符串按'_'分割
                        若分割后前者为'conv1-bias'(已经被存进self._params)
                        且node_input[input_key].s[0]存在于model_dir_path下
                        则可以认为该节点参数已经被导入到self._params
                        该node已经在上一分支中被注册
                    conv1-bias_add-a conv1/out_0                 conv1-out结果
                    
                    conv1-activation-in conv1-bias_add/out_0     conv1-bias_add-out结果
                    
                    pool1-x conv1-activation/out_0               activation-op结果
                    ......
                    """
                    # 本分支用于处理没有存入self._params的node
                    if input_name in self._nodes:
                        # 可能有的node被注册过
                        inputs[input_name] = self._nodes[input_name]
                    else:
                        array_path = os.path.join(model_dir_path, node_input[input_key].s[0])
                        array = np.fromfile(array_path).reshape(node_shape)
                        self._nodes[input_name] = new_var(
                            input_name,
                            shape=node_shape,
                            dtype=array.dtype
                        )
                        inputs[input_name] = self._nodes[input_name]
            else:
                inputs[input_name] = None

        return inputs


    def from_oneflow(self, job, model_dir_path):
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
        """
        举例 lenet from document
        node_name:

        conv1-weight
        conv1-bias
        conv2-weight
        conv2-bias
        dense1-weight
        dense1-bias
        dense2-weight
        dense2-bias
        """
        for layer in model:
            # str(layer)是该层的名字，如：conv1-weight、dense2-bias
            name = str(layer)                # 获取该层的名字，str
            array = model[layer].numpy()     # 获取该层参数，ndarray
            shape = model[layer].shape       # 获取该层参数的形状，tuple
            dtype = array.dtype
            # 获取该层参数的数据类型，np.int32等

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
                        node_name, node_shape, node_dtype = get_node_info(node)
                        if node_name in self._nodes:
                            # 若进入该分支，代表该节点已经被记录
                            continue
                        else:
                            # 若进入该分支，代表该节点为input
                            self._num_input += 1
                            self._input_names.append(node_name)
                            if node_name in self._shape:
                                # 若进入该分支，直接提取该节点之前存好的shape
                                # 原因：shape由用户做过初始化
                                node_shape = self._shape[node_name]
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
                        node_name, node_shape, node_dtype = get_node_info(node)

                        # 构建该op的input
                        node_inputs= self._parse_input(
                            node_name, node_shape, node_dtype,
                            node.user_conf.input, 
                            model_dir_path=model_dir_path
                        )

                        # onnx.py中node.user_conf中为out的名字列表
                        node_outputs = []
                        for output_key in list(node.user_conf.output.keys()):
                            output_name = str(node_name) + '-' + str(output_key)
                            node_outputs.append(output_name)
                        node_outputs = fix_outputs(op_name, node_outputs)

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

                        assert (len(node_outputs) == outputs_num), "Number of output mismatch {} vs {} in {}.".format(
                            len(node_outputs), outputs_num, op_name
                        )

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
        
        # TODO: 同上，同时也需要与onnx.py的作对比，需要重写
        """
        举例: node.return_conf
        in: "Input_1/out"
        out: "out"

        in: "dense2-bias_add/out_0"
        out: "out"
        """
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


def from_oneflow(job, model_dir_path, shape=None, dtype="float32"):
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

    # TODO: 有一种新思路，在这里跑一遍模型，存好相应的模型参数文件夹，后面直接调用就行了
    # 此方法需要改写一开始的self._params的保存
    g = OneflowGraph(shape, dtype)

    # Use the graph proto as a scope so that ops can access other nodes if needed.
    mod, params = g.from_oneflow(job, model_dir_path)
    return mod, params
