"""
Copyright (c) 2015-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import google.protobuf.text_format
import caffe.proto.caffe_pb2 as pb2
import torch2caffe.caffe_layers

import logging
log = logging.getLogger(__name__)

def to_caffe(layers, edges, opts):
    """ 1. prepare the caffe layers """
    caffe_layers = []
    for layer in layers:
        caffe_layer = torch2caffe.caffe_layers.convert(
            opts, layer.typename, layer.torch_layer)
        caffe_layer.name = layer.name
        caffe_layer.bottom.extend([edges[i].name for i in layer.bottom_edges])
        caffe_layer.top.extend([edges[i].name for i in layer.top_edges])
        caffe_layers.append(caffe_layer)

    """ 2. caffe input parameters """
    text_net = pb2.NetParameter()
    if os.environ.get("T2C_DEBUG"):
        text_net.debug_info = True
    for input_spec in opts["inputs"]:
        input_name = input_spec["name"]
        input_dims = input_spec["input_dims"]
        text_net.input.append(input_name)
        input_shape = pb2.BlobShape()
        input_shape.dim.extend([int(i) for i in input_dims])
        text_net.input_shape.extend([input_shape])

    """ 3. caffe layer parameters """
    binary_weights = pb2.NetParameter()
    binary_weights.CopyFrom(text_net)
    for caffe_layer in caffe_layers:
        binary_weights.layer.extend([caffe_layer])
        without_weights = pb2.LayerParameter()
        without_weights.CopyFrom(caffe_layer)
        del without_weights.blobs[:]
        text_net.layer.extend([without_weights])

    return text_net, binary_weights

def save(opts, text_net, binary_weights):
    log.info("Saving to %s (text)", opts["prototxt"])
    with open(opts["prototxt"], "w") as f:
        f.write(google.protobuf.text_format.MessageToString(text_net))

    log.info("Saving to %s (binary)", opts["caffemodel"])
    with open(opts["caffemodel"], "w") as f:
        f.write(binary_weights.SerializeToString())
    log.info("Finished saving model")
