# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=import-self, invalid-name, unused-argument
"""Unit tests for various models and operators"""
import os
import sys

import numpy as np
import pytest
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay
from tvm.contrib import graph_executor

import oneflow as flow

MODEL_HOME = "test_model"


def mkdir(path):
    # init
    path = path.strip()
    path = path.rstrip("\\")

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("{} is already here".format(path))


def rmdir(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.removedirs(path)


def assert_shape(out1, out2):
    if out1.shape != out2.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(out1.shape, out2.shape))


class OneFlowGraph(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def build(self, x):
        out = self.m(x)
        return out


def get_oneflow_output(model, inputs: flow.Tensor):
    flow_output = model(inputs).numpy()
    return flow_output


def get_tvm_output(graph, model_path, inputs: flow.Tensor, target="cuda", dtype="float32"):
    inputs_numpy = inputs.numpy()
    if target == "llvm":
        device = tvm.cpu(0)
    elif target == "cuda":
        device = tvm.cuda(0)

    mod, params = relay.frontend.from_oneflow(graph, model_path)
    with tvm.transform.PassContext(opt_level=10):
        intrp = relay.build_module.create_executor("graph", mod, device, target)
    tvm_output = intrp.evaluate()(tvm.nd.array(inputs_numpy.astype(dtype)), **params).asnumpy()
    return tvm_output


def varifly_conv(model, name="", rtol=1e-5, atol=1e-5):
    conv_model = model.conv
    weight = conv_model.weight
    bias = conv_model.bias

    # weights
    node_name = name + "conv.weight"
    node_path = os.path.join(MODEL_HOME, node_name)
    mkdir(node_path)
    weight.numpy().tofile(os.path.join(node_path, "out"))

    # bias
    if bias is not None:
        node_name = name + "conv.bias"
        node_path = os.path.join(MODEL_HOME, node_name)
        mkdir(node_path)
        bias.numpy().tofile(os.path.join(node_path, "out"))

    # snapshot_done
    with open(os.path.join(MODEL_HOME, "snapshot_done"), "w") as f:
        f.write("")

    inputs = flow.Tensor(np.random.rand(1, 3, 224, 224), dtype=flow.float32, device=weight.device)
    graph = OneFlowGraph(model)
    graph._compile(inputs)

    out_flow = get_oneflow_output(graph, inputs)
    out_tvm = get_tvm_output(graph, MODEL_HOME, inputs)
    rmdir(MODEL_HOME)

    assert_shape(out_flow, out_tvm)
    tvm.testing.assert_allclose(out_flow, out_tvm, rtol=rtol, atol=atol)


# defs/nn
@tvm.testing.uses_gpu
def test_conv2d():
    class Conv2dModel(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = flow.nn.Conv2d(3, 64, 3)
        def forward(self, x):
            x = self.conv(x)
            return x

    if os.path.exists(MODEL_HOME):
        rmdir(MODEL_HOME)

    model = Conv2dModel()
    model.eval()
    model.to("cuda")
    varifly_conv(model)


test_conv2d()
rmdir("log")
