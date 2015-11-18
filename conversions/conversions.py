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

import logging
import itertools
import os
import tempfile

import click
import numpy as np

import caffe
import caffe.proto.caffe_pb2 as pb2

import google.protobuf.text_format


log = logging.getLogger(__name__)

THRESHOLD = 1E-3

# TODO - refactor this in to a sequence of (prototxt, caffemodel) ->
# (prototxt, caffemodel) passes.


def flatmap(f, items):
    return itertools.chain.from_iterable(itertools.imap(f, items))


def load_prototxt(params_file):
    params = pb2.NetParameter()
    with open(params_file) as f:
        google.protobuf.text_format.Merge(f.read(), params)
    return params


def convert_fc_layer(net, fc_layer):
    conv_layer = pb2.LayerParameter()
    conv_layer.name = "{}_conv".format(fc_layer.name)
    conv_layer.type = "Convolution"
    conv_layer.bottom.extend(list(fc_layer.bottom))
    conv_layer.top.extend(list(fc_layer.top))
    # get input
    assert len(fc_layer.bottom) == 1
    bottom_name = fc_layer.bottom[0]
    bottom_shape = list(net.blobs[bottom_name].shape)
    if len(bottom_shape) == 2:
        bottom_shape.extend([1, 1])

    num_output = net.params[fc_layer.name][0].data.shape[0]
    assert bottom_shape[-1] == bottom_shape[-2], bottom_shape
    conv_layer.convolution_param.kernel_size = bottom_shape[-1]
    conv_layer.convolution_param.num_output = num_output
    return conv_layer


def convert_fc_prototxt(params_file):
    params = load_prototxt(params_file)

    def find_layer(name):
        (layer,) = [l for l in params.layer if l.name == name]
        return layer

    def f(layer):
        if layer.type == "Flatten":
            new_layer = pb2.LayerParameter()
            new_layer.CopyFrom(layer)
            new_layer.type = "Reshape"
            new_layer.reshape_param.shape.dim.extend([0, 0, 0, 0])
            return new_layer
        if layer.type == "InnerProduct":
            new_layer = pb2.LayerParameter()
            new_layer.CopyFrom(layer)
            new_layer.inner_product_param.axis = 1
            return new_layer
        return layer

    new_layers = [f(l) for l in params.layer]
    new_params = pb2.NetParameter()
    new_params.CopyFrom(params)
    del new_params.layer[:]
    new_params.layer.extend(new_layers)
    return new_params


def convert_spatial_prototxt(params_file):
    net = caffe.Net(str(params_file), caffe.TEST)
    params = load_prototxt(params_file)

    def find_layer(name):
        (layer,) = [l for l in params.layer if l.name == name]
        return layer

    def f(layer):
        if layer.type != "InnerProduct":
            return [layer]
        return [convert_fc_layer(net, layer)]

    new_layers = flatmap(f, params.layer)
    new_params = pb2.NetParameter()
    new_params.CopyFrom(params)
    del new_params.layer[:]
    new_params.layer.extend(new_layers)
    return new_params


def convert_spatial_net(spatial_params_file, spatial_weights_file,
                        conv_params_file):
    spatial_net = caffe.Net(
        str(spatial_params_file), str(spatial_weights_file), caffe.TEST)
    # Initialize from the SPATIAL layer
    conv_net = caffe.Net(
        str(conv_params_file), str(spatial_weights_file), caffe.TEST)
    spatial_params = load_prototxt(spatial_params_file)

    converted_layer_names = [
        (layer.name, convert_fc_layer(spatial_net, layer).name)
        for layer in spatial_params.layer
        if layer.type == "InnerProduct"
    ]

    for layer_pair in converted_layer_names:
        log.info("Converting layer pair: %s", layer_pair)
        (spatial_layer_name, conv_layer_name) = layer_pair
        spatial_params = spatial_net.params[spatial_layer_name]
        conv_params = conv_net.params[conv_layer_name]

        assert len(spatial_params) == len(conv_params)
        for spatial_param, conv_param in zip(spatial_params, conv_params):
            log.info("Spatial Layer: %s - %s, Conv Layer: %s - %s",
                     spatial_layer_name, spatial_param.data.shape,
                     conv_layer_name, conv_param.data.shape)
            assert(conv_param.data.size == spatial_param.data.size)
            conv_param.data.flat = spatial_param.data.flat
    return spatial_net, conv_net


def verify_equivalent(fc_net, conv_net):
    log.info("Verifying convnets")
    input_names = fc_net.inputs
    log.info("Running on inputs: %s", input_names)
    inputs = {
        input_name: np.random.random(
            size=tuple(list(fc_net.blobs[input_name].shape)))
        for input_name in input_names}

    fc_outputs = fc_net.forward(**inputs)
    conv_outputs = conv_net.forward(**inputs)
    # Verify convolutional model works
    for k, conv_output in conv_outputs.iteritems():
        log.info("%s: %s", k, conv_output.shape)
        fc_output = fc_outputs[k]
        delta = np.amax(np.abs(conv_output.flatten() - fc_output.flatten()))
        log.info("Maximum delta: %s", delta)
        if delta < THRESHOLD:
            log.info("Delta: %s < threshold: %s", delta, THRESHOLD)
            continue

        log.info("Conv output: %s", conv_output.flatten())
        log.info("FC output: %s", fc_output.flatten())
        for ((fcn, fcb), (cnn, cnb)) in zip(
                list(fc_net.blobs.iteritems()),
                list(conv_net.blobs.iteritems())):
            log.info("FCN: %s - %s, CNN: %s - %s",
                     fcn, fcb.data.shape, cnn, cnb.data.shape)
            log.info(np.amax(np.abs(fcb.data.flatten() - cnb.data.flatten())))
        raise Exception("Failed to precisely convert models")


@click.group()
def cli():
    pass


@cli.command()
@click.option("--conv-prototxt", type=str, required=True)
@click.option("--output-scanning-prototxt", type=str, required=True)
def scanning(conv_prototxt, output_scanning_prototxt):
    """
    Add a scanning layer on top of all softmax layers, so we max-pool
    the class probabilities over spatial locations.
    """
    conv_params = load_prototxt(conv_prototxt)

    def add_scanning(layer):
        if layer.type != "Softmax":
            return [layer]
        scanning_layer = pb2.LayerParameter()
        scanning_layer.name = "{}_scanning".format(layer.name)
        scanning_layer.bottom.extend(layer.top)
        scanning_layer.top.extend([scanning_layer.name])
        scanning_layer.type = "Pooling"
        scanning_layer.pooling_param.pool = pb2.PoolingParameter.MAX
        scanning_layer.pooling_param.global_pooling = True
        return [layer, scanning_layer]

    scanning_layers = flatmap(add_scanning, conv_params.layer)
    scanning_params = pb2.NetParameter()
    scanning_params.CopyFrom(conv_params)
    del scanning_params.layer[:]
    scanning_params.layer.extend(scanning_layers)
    scanning_prototxt = tempfile.NamedTemporaryFile(
        dir=os.path.dirname(output_scanning_prototxt),
        delete=False).name
    with open(scanning_prototxt, "w") as f:
        f.write(google.protobuf.text_format.MessageToString(scanning_params))
    # Verify the net loads with the scanning change.
    caffe.Net(str(scanning_prototxt), caffe.TEST)
    log.info("Moving: %s to %s", scanning_prototxt, output_scanning_prototxt)
    os.rename(scanning_prototxt, output_scanning_prototxt)


@cli.command()
@click.option("--fc-prototxt", type=str, required=True)
@click.option("--fc-caffemodel", type=str, required=True)
@click.option("--output-spatial-prototxt", type=str, required=True)
@click.option("--output-spatial-caffemodel", type=str, required=True)
def spatial(fc_prototxt, fc_caffemodel, output_spatial_prototxt,
            output_spatial_caffemodel):
    """
    Remove `Flatten` layers to preserve the spatial structure
    """
    logging.basicConfig(level=logging.INFO)

    spatial_net_params = convert_fc_prototxt(fc_prototxt)
    spatial_prototxt = tempfile.NamedTemporaryFile(
        dir=os.path.dirname(output_spatial_prototxt),
        suffix=".spatial_prototxt",
        delete=False).name
    with open(spatial_prototxt, "w") as f:
        f.write(google.protobuf.text_format.MessageToString(
            spatial_net_params))
    log.info("Spatial params: %s", spatial_prototxt)
    fc_net = caffe.Net(str(fc_prototxt), str(fc_caffemodel), caffe.TEST)
    spatial_net = caffe.Net(str(spatial_prototxt), str(fc_caffemodel),
                            caffe.TEST)
    verify_equivalent(fc_net, spatial_net)

    spatial_caffemodel = tempfile.NamedTemporaryFile(
        dir=os.path.dirname(output_spatial_caffemodel),
        suffix=".spatial_caffemodel",
        delete=False).name
    spatial_net.save(str(spatial_caffemodel))
    log.info("Moving: %s to %s", spatial_prototxt, output_spatial_prototxt)
    os.rename(spatial_prototxt, output_spatial_prototxt)
    log.info("Moving: %s to %s", spatial_caffemodel, output_spatial_caffemodel)
    os.rename(spatial_caffemodel, output_spatial_caffemodel)


@cli.command()
@click.option("--spatial-prototxt", type=str, required=True)
@click.option("--spatial-caffemodel", type=str, required=True)
@click.option("--output-conv-prototxt", type=str, required=True)
@click.option("--output-conv-caffemodel", type=str, required=True)
def convolutional(spatial_prototxt, spatial_caffemodel,
                  output_conv_prototxt, output_conv_caffemodel):
    """
    Convert all fully connected layers to convolutional layers.
    """
    logging.basicConfig(level=logging.INFO)

    conv_net_params = convert_spatial_prototxt(spatial_prototxt)
    conv_prototxt = tempfile.NamedTemporaryFile(
        dir=os.path.dirname(output_conv_prototxt),
        suffix=".conv_prototxt",
        delete=False).name
    with open(conv_prototxt, "w") as f:
        f.write(google.protobuf.text_format.MessageToString(conv_net_params))
    log.info("Conv params: %s", conv_prototxt)

    (spatial_net, conv_net) = convert_spatial_net(
        spatial_prototxt, spatial_caffemodel, conv_prototxt)
    verify_equivalent(spatial_net, conv_net)
    conv_caffemodel = tempfile.NamedTemporaryFile(
        dir=os.path.dirname(output_conv_caffemodel),
        suffix=".conv_caffemodel",
        delete=False).name
    conv_net.save(str(conv_caffemodel))

    log.info("Moving: %s to %s", conv_prototxt, output_conv_prototxt)
    os.rename(conv_prototxt, output_conv_prototxt)
    log.info("Moving: %s to %s", conv_caffemodel, output_conv_caffemodel)
    os.rename(conv_caffemodel, output_conv_caffemodel)


@cli.command()
@click.option("--conv-prototxt", type=str, required=True)
@click.option("--scale", type=float, multiple=True)
def scales(conv_prototxt, scale):
    """
    Examine the network output dimensions across a series of input scales.
    """
    logging.basicConfig(level=logging.INFO)

    net = caffe.Net(str(conv_prototxt), caffe.TEST)
    input_names = net.inputs
    input_shapes = {
        input_name: tuple(net.blobs[input_name].shape)
        for input_name in input_names}

    for scalar in scale:
        log.info("Running on scale: %s", scalar)

        def perturb(i, n):
            # only perturb HxW in NxCxHxW
            if i in (2, 3):
                return int(n * scalar)
            return n

        inputs = {
            input_name: np.random.random(
                size=tuple(
                    perturb(i, n)
                    for (i, n) in enumerate(shape)))
            for input_name, shape in input_shapes.iteritems()}

        for input_name, input in inputs.iteritems():
            log.info("Input: %s, shape: %s", input_name, input.shape)
            net.blobs[input_name].reshape(*input.shape)
        net.reshape()
        conv_outputs = net.forward(**inputs)
        for output_name, conv_output in conv_outputs.iteritems():
            log.info("%s: %s", output_name, conv_output.shape)


@cli.command()
@click.option("--prototxt", required=True)
@click.option("--caffemodel", required=True)
@click.option("--output-prototxt", required=True)
@click.option("--output-caffemodel", required=True)
@click.pass_context
def vision(ctx, prototxt, caffemodel, output_prototxt, output_caffemodel):
    spatial_prototxt = tempfile.NamedTemporaryFile(
        suffix=".spatial_prototxt", delete=False).name
    spatial_caffemodel = tempfile.NamedTemporaryFile(
        suffix=".spatial_caffemodel", delete=False).name
    ctx.invoke(spatial,
               fc_prototxt=prototxt,
               fc_caffemodel=caffemodel,
               output_spatial_prototxt=spatial_prototxt,
               output_spatial_caffemodel=spatial_caffemodel)
    conv_prototxt = tempfile.NamedTemporaryFile(
        suffix=".conv_prototxt", delete=False).name
    conv_caffemodel = tempfile.NamedTemporaryFile(
        dir=os.path.dirname(output_caffemodel),
        suffix=".conv_prototxt", delete=False).name
    ctx.invoke(convolutional,
               spatial_prototxt=spatial_prototxt,
               spatial_caffemodel=spatial_caffemodel,
               output_conv_prototxt=conv_prototxt,
               output_conv_caffemodel=conv_caffemodel)

    ctx.invoke(scanning,
               conv_prototxt=conv_prototxt,
               output_scanning_prototxt=output_prototxt)
    log.info("Moving: %s to %s", conv_caffemodel, output_caffemodel)
    os.rename(conv_caffemodel, output_caffemodel)

if __name__ == "__main__":
    cli()
