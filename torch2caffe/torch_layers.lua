--[[
Copyright (c) 2015-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
--]]
local M = {}

local py = require 'fb.python'
local logging = require 'fb.util.logging'
local pl = require('pl.import_into')()

-- Various libraries for loading Torch layers.
require 'torch'
require 'nn'

-- Don't automatically fail if CUDA isn't available.
pcall(function() require 'cudnn' end)
pcall(function() require 'cunn' end)
pcall(function() require 'fbcunn' end)

local function simple(options)
    assert(options.typename)
    local typename = options.typename
    local props = options.layer
    local inplace = options.inplace or false
    local num_bottoms = options.num_bottoms or 1
    local num_tops = options.num_tops or 1
    return function(net, layer, bottom_edges, top_edges)
        local actual = nil
        if type(props) == "nil" then
            actual = layer
        elseif type(props) == "function" then
            actual = props(layer)
        else
            actual = props
        end
        return py.eval(net.add_layer(
            typename, actual, bottom_edges, top_edges,
            num_bottoms, num_tops, inplace))
    end
end

M.CONVERTER = {
    ['nn.LogSoftMax'] = simple{typename="caffe.LogSoftmax", inplace=true},
    ['nn.SoftMax'] = simple{typename="caffe.Softmax", inplace=true},
    ['nn.LogSumExp'] = simple{typename="caffe.Reduction",
                              layer={operation="LOGSUMEXP", axis=-1}},
    ['nn.Mean'] = simple{typename="caffe.Reduction",
                         layer={operation="MEAN", axis=-1}},
    ['nn.Sum'] = simple{typename="caffe.Reduction",
                        layer={operation="SUM", axis=-1}},
    ['nn.Linear'] = simple{
        typename="caffe.InnerProduct",
        layer=function(layer)
            layer["num_output"] = layer.weight:size(1)
            return layer
        end},
    ['FixedLookupTable'] = simple{
        typename="caffe.Embed",
        layer=function(layer)
            layer["num_output"] = layer.weight:size(2)
            layer["input_dim"] = layer.weight:size(1)
            return layer
        end},
    ['nn.LookupTable'] = simple{
        typename="caffe.Embed",
        layer=function(layer)
            layer["num_output"] = layer.weight:size(2)
            layer["input_dim"] = layer.weight:size(1)
            return layer
        end},
    ['nn.Tanh'] = simple{typename='caffe.TanH', inplace=true},
    ['nn.ReLU'] = simple{typename='caffe.ReLU', inplace=true},
    ['nn.Threshold'] = simple{typename='caffe.FBThreshold', inplace=true},
    ['nn.Sequential'] = function(net, layer, bottom_edges, top_edges)
        for i = 1, #layer.modules do
            local tops = (i == #layer.modules) and top_edges or nil
            bottom_edges = M.add(net, layer.modules[i], bottom_edges, tops)
        end
        return bottom_edges
    end,
    ['nn.ConcatTable'] = function(net, layer, bottom_edges, top_edges)
        assert(not top_edges or #top_edges == #layer.modules)
        local actual_top_edges = {}
        for i = 1, #layer.modules do
            local top_edge = top_edges and {top_edges[i]} or nil
            top_edge = M.add(net, layer.modules[i], bottom_edges, top_edge)
            assert(#top_edge == 1)
            table.insert(actual_top_edges, top_edge[1])
        end
        return actual_top_edges
    end,
    ['nn.Concat$'] = function(net, layer, bottom_edges, top_edges)
        local mid_edges = {}
        for i = 1, #layer.modules do
            local mid_edge = M.add(net, layer.modules[i], bottom_edges, nil)
            assert(#mid_edge == 1)
            table.insert(mid_edges, mid_edge[1])
        end
        return py.eval(
            net.add_layer("caffe.Concat",
                          {axis=(layer.dimension - 1)},
                          mid_edges, -- bottom_edges
                          top_edges, -- top_edges
                          #layer.modules, -- num_bottoms
                          1, -- num_tops
                          false)) -- inplace
    end,
    ['nn.ParallelTable$'] = function(net, layer, bottom_edges, top_edges)
        assert(#bottom_edges == #layer.modules)
        local actual_top_edges = {}
        for i = 1, #layer.modules do
            local top_edge = top_edges and {top_edges[i]} or nil
            top_edge = M.add(net, layer.modules[i], {bottom_edges[i]}, top_edge)
            assert(#top_edge == 1)
            table.insert(actual_top_edges, top_edge[1])
        end
        return actual_top_edges
    end,
    ['nn.JoinTable'] = function(net, layer, bottom_edges, top_edges)
        return py.eval(
            net.add_layer(
                "caffe.Concat",
                {
                    -- TODO - infer this?
                    -- Assuming that in caffe we are pre-pending
                    -- one additional dimension to all blobs
                    -- (compensated by 1 vs. 0-indexing)
                    axis=layer.dimension
                },
                bottom_edges,
                top_edges,
                #bottom_edges, -- num_bottoms
                1, -- num_tops
                false)) -- inplace
    end,
    ['nn.Parallel$'] = function(net, layer, bottom_edges, top_edges)
        local split_edges = py.eval(
            net.add_layer("caffe.Slice",
                          {
                              axis=-layer.inputDimension,
                              num_slices=#layer.modules,
                          },
                          bottom_edges,
                          nil, -- top_edges
                          1, -- num_bottoms
                          #layer.modules, -- num_tops
                          false)) -- inplace
        local mid_edges = {}
        for i = 1, #layer.modules do
            local mid_edge = M.add(net, layer.modules[i], {split_edges[i]}, nil)
            assert(#mid_edge == 1)
            table.insert(mid_edges, mid_edge[1])
        end
        return py.eval(
            net.add_layer(
                "caffe.Concat",
                {
                    -- @oujin - more efficient this way for your model.
                    axis=-1
                    -- axis=-layer.outputDimension
                },
                mid_edges, -- bottom_edges
                top_edges, -- top_edges
                #layer.modules, -- num_bottoms
                1, -- num_tops
                false)) -- inplace
    end,
    ['nn.DataParallel'] = function(net, layer, bottom_edges, top_edges)
        -- Only choose the first branch of DataParallel, since
        -- in training, Caffe has global DataParallel
        return M.add(net, layer.modules[1], bottom_edges, top_edges)
    end,
    ['nn.TemporalConvolution'] = function(net, layer, bottom_edges, top_edges)
        local convolution_edges = py.eval(
            net.add_layer('caffe.TemporalConvolution',
                          layer,
                          bottom_edges,
                          nil, -- top_edges
                          1, -- num_bottoms
                          1, -- num_tops
                          false)) -- inplace
        -- Flatten out to squeeze the inner dimension, since
        -- TemporalConvolution leaves an nxmx1 batch in Caffe but not
        -- in Torch.
        return py.eval(
            net.add_layer("caffe.Reduction",
                          {operation="SUM", axis=-1},
                          convolution_edges, -- bottom_edges,
                          top_edges,
                          1, -- num_bottoms
                          1, -- num_tops
                          false)) -- inplace
    end,
    ['nn.SpatialConvolutionMM'] = simple{
        typename='caffe.SpatialConvolution',
        layer=function(layer)
            -- weight is (nOutput, nInputxkHxkW)
            local new_layer = layer:clone()
            new_layer.weight = torch.FloatTensor(
                layer.nOutputPlane, layer.nInputPlane, layer.kH, layer.kW):copy(
                layer.weight)
            if not new_layer.padH then
                new_layer.padH = layer.padding
            end
            if not new_layer.padW then
                new_layer.padW = layer.padding
            end
            return new_layer
        end},
    ['nn.SpatialConvolution$'] = simple{typename='caffe.SpatialConvolution'},
    ['nn.SpatialMaxPooling'] = simple{
        typename='caffe.Pooling',
        layer=function(layer)
            layer["operation"] = "MAX"
            return layer
        end},
    ['nn.SpatialAveragePooling'] = simple{
        typename='caffe.Pooling',
        layer=function(layer)
            layer["operation"] = "AVE"
            return layer
        end},
    ['nn.Dropout'] = simple{typename='caffe.Dropout', inplace=true},
    ['nn.View'] = simple{
        typename='caffe.Flatten',
        layer=function(layer)
            if layer.size:size() ~= 1 then
                logging.fatalf("Only handle nn.View(k) for now")
            end
            return {}
        end},
    ['nn.LSTM'] = function(net, layer, bottom_edges, top_edges)
        if #bottom_edges < 2 then
            local cont_edge = py.eval(net.edge_by_name('lstm_continuation'))
            assert(type(cont_edge) ~= 'nil')
            table.insert(bottom_edges, cont_edge)
        end
        return py.eval(
            net.add_layer("caffe.LSTM",
                          layer,
                          bottom_edges, -- bottom_edges
                          top_edges, -- top_edges
                          2, -- num_bottoms
                          1, -- num_tops
                          false)) -- inplace
    end,
    ['nn.Reshape'] = simple{
        typename='caffe.Flatten',
        layer=function(layer)
            if layer.size:size() ~= 1 then
                logging.fatalf("Only handle nn.Reshape(k) for now")
            end
            return {}
        end},
}

function M.add(net, layer, bottom_edges, top_edges)
    local layer_type = torch.type(layer)
    for layer_pattern, converter in pairs(M.CONVERTER) do
        if string.find(layer_type, layer_pattern) then
            return converter(net, layer, bottom_edges, top_edges)
        end
    end
    logging.fatalf("Unknown layer type: %s, known types: %s",
                   layer_type,
                   pl.stringx.join(", ", pl.tablex.keys(M.CONVERTER)))
end

return M
