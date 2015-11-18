--[[
Copyright (c) 2015-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
--]]
local M = {}

require 'nn'

local pl = require 'pl.import_into'()
local py = require 'fb.python'
local torch = require 'torch'
require 'fbtorch'
local logging = require 'fb.util.logging'
local torch_layers = require 'torch2caffe.torch_layers'
local t2c = py.import('torch2caffe.lib_py')

function M.evaluate_caffe(caffe_net, inputs)
    local input_kwargs = {}
    for i=1,#inputs do
        local input_spec = inputs[i]
        input_kwargs[input_spec.name] = input_spec.tensor
    end
    local py_caffe_output = caffe_net.forward(py.kwargs, input_kwargs)
    local caffe_output_list = py.reval(t2c.format_output(py_caffe_output))
    local caffe_output_length = py.eval("len(a)", {a=caffe_output_list})
    local caffe_outputs = {}
    for i=0,caffe_output_length-1 do
        table.insert(caffe_outputs,
                     torch.FloatTensor(
                         torch.totable(py.eval(caffe_output_list[i]))))
    end
    return caffe_outputs
end

local function debug_nets(caffe_net, torch_net)
    py.reval(t2c.debug_net(caffe_net))
    torch_net:apply(
        function(m)
            if m.output then
                local sizes = {}
                local sums = {}
                if type(m.output) == 'table' then
                    for i=1,#m.output do
                        table.insert(sizes, m.output[i]:size())
                        table.insert(sums, torch.sum(m.output[i]))
                    end
                else
                    sizes = torch.totable(m.output:size())
                    sums = torch.sum(m.output)
                end
                logging.infof("Layer %s, %s, Sum: %s",
                              torch.typename(m),
                              sizes,
                              sums)
            end
        end
    )
end

function M.compare(opts, torch_net)
    torch_net:apply(function(m) m:evaluate() end)
    local inputs = {}
    for i=1,#opts.inputs do
        local input_spec = opts.inputs[i]
        local tensor
        if input_spec.tensor then
            tensor = input_spec.tensor
        else
            tensor = torch.rand(table.unpack(input_spec.input_dims)):float()
        end
        table.insert(inputs, {name=input_spec.name, tensor=tensor})
    end

    -- Legacy code
    if opts.input_tensor then
        assert(inputs[1].name == "data")
        inputs[1].tensor = opts.input_tensor
    end

    local caffe_net = t2c.load(opts)
    local caffe_outputs = M.evaluate_caffe(caffe_net, inputs)

    -- Torch multi-inputs take an ordered Table.
    local function inputs_to_torch_inputs(inputs, type)
        if #inputs == 1 then
            return inputs[1].tensor:type(type)
        end
        local tensors = {}
        for i=1,#inputs do
            table.insert(tensors, inputs[i].tensor:type(type))
        end
        return tensors
    end
    local torch_outputs
    -- Some networks only accept CUDA input.
    local ok, err = pcall(function()
            torch_net:float()
            local torch_inputs = inputs_to_torch_inputs(
                inputs, 'torch.FloatTensor')
            torch_outputs = torch_net:forward(torch_inputs)
    end)
    if not ok then
        logging.infof("Got error running forward: %s", err)
        torch_net:cuda()
        local torch_inputs = inputs_to_torch_inputs(
            inputs, 'torch.CudaTensor')
        torch_outputs = torch_net:forward(torch_inputs)
    end

    if type(torch_outputs) == "table" then
        for i=1,#torch_outputs do
            torch_outputs[i] = torch_outputs[i]:float()
        end
    else
        torch_outputs = {torch_outputs:float()}
    end

    if #caffe_outputs ~= #torch_outputs then
        logging.errorf("Inconsistent output blobs: Caffe: %s, Torch: %s",
                       #caffe_outputs, #torch_outputs)
        error("Inconsistent output blobs")
    end

    for i = 1,#caffe_outputs do
        local torch_output = torch_outputs[i]
        local caffe_output = caffe_outputs[i]
        logging.infof("Caffe norm: %s, Torch norm: %s",
                      torch.norm(caffe_output), torch.norm(torch_output))
        if not caffe_output:isSameSizeAs(torch_output) then
            logging.errorf("Inconsistent output size: Caffe: %s, Torch: %s",
                           caffe_output:size(), torch_output:size())
            error("Inconsistent output sizes")
        end

        local max_absolute_error = (caffe_output - torch_output):abs():max()
        logging.infof("Maximum difference between Caffe and Torch output: %s",
                      max_absolute_error)
        if (max_absolute_error > 0.001) then
            debug_nets(caffe_net, torch_net)
            if os.getenv('LUA_DEBUG_ON_ERROR') then
                require('fb.debugger').enter()
            end
            error("Error in conversion!")
        end
    end
    if os.getenv('LUA_DEBUG_ON_ERROR') then
        require('fb.debugger').enter()
    end
end

function M.convert(opts, torch_net)
    assert(opts)
    assert(torch_net)
    local net_builder = py.reval(t2c.initialize())
    local bottom_edges = py.eval(t2c.setup_inputs(opts, net_builder))
    local top_edges = py.eval(t2c.setup_outputs(opts, net_builder))
    torch_layers.add(net_builder, torch_net, bottom_edges, top_edges)
    t2c.finalize(opts, net_builder)
end

function M.run(opts, torch_net)
    print("Running with model: ")
    print(torch_net)
    M.convert(opts, torch_net)
    M.compare(opts, torch_net)
end

function M.main(opts)
    logging.infof("Opts: %s", pl.pretty.write(opts))
    if opts.input_tensor ~= "" then
        opts.input_tensor = torch.load(opts.input_tensor)
    else
        opts.input_tensor = nil
    end

    -- Initialize fbcunn, fbnn, random includes, additions to
    -- t2c.CONVERTER, etc

    local model
    if opts.format == "lua" then
        model = assert(torch.load(opts.input))
    elseif opts.format == "luathrift" then
        local f = assert(io.open(opts.input))
        local thrift = require 'fb.thrift'
        model = thrift.from_file(f)
    end

    if opts.preprocessing and opts.preprocessing ~= "" then
        paths.dofile(opts.preprocessing)
    end
    if g_t2c_preprocess then
        model = g_t2c_preprocess(model, opts)
    end

    if not opts.inputs then
        opts.inputs = {{name="data", input_dims=opts.input_dims}}
    end

    logging.infof("Parsed opts: %s", pl.pretty.write(opts))
    if opts.verify ~= "" then
        return M.compare(opts, model)
    else
        return M.run(opts, model)
    end

end

return M
