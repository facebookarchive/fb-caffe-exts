--[[
Copyright (c) 2015-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
--]]
require 'torch'
local logging = require 'fb.util.logging'
local M = {}

local function fold_sequential_batch_norm_layer(model)
    local new_model = nn.Sequential()
    for i = 1, #model.modules do
        (function()
                local layer_type = torch.type(model.modules[i])
                local batch_norm_layer_type
                if os.getenv("HACK_SKIP_SPATIAL_BN") then
                    batch_norm_layer_type = 'nn.BatchNormalization'
                else
                    batch_norm_layer_type = 'nn..*BatchNormalization'
                end
                if string.find(layer_type, batch_norm_layer_type) then
                    logging.infof("Got bn layer: %s, skipping", layer_type)
                    return
                end

                if i + 1 > #model.modules then
                    new_model:add(
                        M.fold_batch_normalization_layers(model.modules[i]))
                    return
                end

                local next_layer_type = torch.type(model.modules[i+1])
                if not string.find(next_layer_type, 'nn..*BatchNormalization')
                then
                    new_model:add(
                        M.fold_batch_normalization_layers(model.modules[i]))
                    return
                end
                logging.infof("Current: %s, Next: %s", layer_type, next_layer_type)
                assert(string.find(layer_type, 'nn.SpatialConvolution')
                           or string.find(layer_type, 'nn.Linear'))
                local new_module = model.modules[i]:clone()
                local bn_layer = model.modules[i+1]
                assert(bn_layer.running_mean)
                assert(bn_layer.running_std)
                local mean = bn_layer.running_mean
                local std = bn_layer.running_std
                local a2 = bn_layer.weight
                local b2 = bn_layer.bias
                local sz = new_module.weight:size()
                local nc = sz[1]
                sz[1] = 1

                -- ((a * x + b) - m) * std => a * std * x + (b-m) * std
                new_module.bias:add(-1, mean)
                new_module.bias:cmul(std)
                local buf = torch.repeatTensor(std:view(nc, 1), sz)
                new_module.weight:cmul(buf)
                mean:zero()
                std:fill(1)

                if bn_layer.affine then
                    -- a2 * (a1 * x + b1) + b2 => a2 * a1 * x + a2 * b1 + b2
                    new_module.bias:cmul(a2)
                    new_module.bias:add(b2)
                    buf = torch.repeatTensor(a2:view(nc, 1), sz)
                    new_module.weight:cmul(buf)
                    a2:fill(1)
                    b2:zero()
                end
                new_model:add(new_module)
         end
        )()
    end
    return new_model
end


function M.fold_batch_normalization_layers(model)
    local model_type = torch.type(model)
    if model_type == 'nn.ConcatTable' then
        for i = 1, #model.modules do
            model.modules[i] =
                M.fold_batch_normalization_layers(model.modules[i])
        end
        return model
    elseif model_type == 'nn.Sequential' then
        return fold_sequential_batch_norm_layer(model)
    else
        return model
    end
end

local LSTM = torch.class('nn.LSTM', 'nn.Module')
function LSTM:__init(layers)
    self.layers = layers
end

-- Convert an nngraph to an LSTM.
-- Relies on nngraph annotations to infer the network structure -
function M.nngraph_lstm_to_sequential(graph)
    require 'torch'
    require 'nngraph'
    local nodes = graph.forwardnodes
    local m = nn.Sequential()
    local lstm_layers = {}
    local last_node = nil
    for i=1,#nodes do
        local anns = nodes[i].data.annotations
        local mod = nodes[i].data.module
        if anns and torch.type(anns.layer) == "number" and (
            torch.typename(mod) == "nn.Linear"
            or torch.typename(mod) == "nn.CMul") then
            if not lstm_layers[anns.layer] then
                lstm_layers[anns.layer] = {}
            end
            lstm_layers[anns.layer][
                string.format("%s_%s_%s",
                              anns.gate, anns.name, torch.typename(mod))] = mod
        end
        if anns and anns.layer == "last" then
            last_node = nodes[i]
        end
    end

    assert(last_node)
    assert(torch.typename(last_node.data.module) == "nn.Linear")
    -- Assumption - LSTM layers followed by "last" sequence
    for i=1,#lstm_layers do
        m:add(nn.LSTM(lstm_layers[i]))
    end
    m:add(nn.View(last_node.data.module.weight:size()[2]))
    while true do
        if not last_node or not last_node.data.module then
            break
        end
        m:add(last_node.data.module)
        if not last_node.children then
            break
        end
        assert(#last_node.children == 1)
        last_node = last_node.children[1]
    end
    return m
end

return M
