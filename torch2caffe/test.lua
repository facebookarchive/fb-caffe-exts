--[[
Copyright (c) 2015-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
--]]
require 'fb.luaunit'
local t2c = require 'torch2caffe.lib'
local logging=  require 'fb.util.logging'
Test = {}
g_SLOW = "" -- set to "" to run slow tests

local fwk
if pcall(function() require 'cudnn' end) then
    logging.info("Using `cudnn`")
    fwk = cudnn
else
    logging.info("Using `nn`")
    fwk = nn
end

local function check(module, input_dims)
    module:apply(function(m) m:evaluate() end)
    local opts = {
            prototxt=os.tmpname(),
            caffemodel=os.tmpname(),
            inputs={{name="data", input_dims=input_dims}},
    }
    t2c.run(opts, module)
    return opts
end

local function check_opts(module, opts)
    module:apply(function(m) m:evaluate() end)
    opts.prototxt=os.tmpname()
    opts.caffemodel=os.tmpname()
    t2c.run(opts, module)
end

function Test:testSequential()
    local m = nn.Sequential()
    m:add(nn.Linear(100, 40))
    m:add(nn.ReLU())
    m:add(nn.LogSoftMax())
    check(m, {400, 100})
end

function Test:testLinear()
    local m = nn.Linear(100, 4096)
    check(m, {2, 100})
end

function Test:testSpatialConvolutionMM()
    local m = nn.SpatialConvolutionMM(3,32,3,3,1,1,1,1)
    check(m, {1,3,100,100})
end


function Test:testFBThreshold()
    local m = nn.Threshold(0, 1e-6)
    check(m, {2, 100})
end

local function vggConv()
    local m = nn.Sequential()
    m:add(fwk.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
    m:add(fwk.ReLU())
    m:add(fwk.SpatialMaxPooling(2,2,2,2))
    m:add(fwk.SpatialConvolution(64,128, 3,3, 1,1, 1,1))
    m:add(fwk.ReLU())
    m:add(fwk.SpatialMaxPooling(2,2,2,2))
    m:add(fwk.SpatialConvolution(128,256, 3,3, 1,1, 1,1))
    m:add(fwk.ReLU())
    m:add(fwk.SpatialConvolution(256,256, 3,3, 1,1, 1,1))
    m:add(fwk.ReLU())
    m:add(fwk.SpatialMaxPooling(2,2,2,2))
    m:add(fwk.SpatialConvolution(256,512, 3,3, 1,1, 1,1))
    m:add(fwk.ReLU())
    m:add(fwk.SpatialConvolution(512,512, 3,3, 1,1, 1,1))
    m:add(fwk.ReLU())
    m:add(fwk.SpatialMaxPooling(2,2,2,2))
    m:add(fwk.SpatialConvolution(512,512, 3,3, 1,1, 1,1))
    m:add(fwk.ReLU())
    m:add(fwk.SpatialConvolution(512,512, 3,3, 1,1, 1,1))
    m:add(fwk.ReLU())
    m:add(fwk.SpatialMaxPooling(2,2,2,2))
    return m
end

function Test:testVggConv()
    local m = vggConv()
    check(m, {1,3,32,32})
end

local function vggLinear()
    local m = nn:Sequential()
    m:add(nn.View(512))
    m:add(nn.Linear(512, 4096))
    m:add(nn.Threshold(1e-6, 0))
    m:add(nn.Dropout())
    m:add(nn.Linear(4096, 2))
    m:add(nn.LogSoftMax())
    return m
end

function Test:testVggLinear()
    local m = vggLinear()
    check(m, {2,512})
end

function Test:testVggCombined()
    local mm = nn.Sequential()
    mm:add(vggConv())
    mm:add(vggLinear())
    check(mm, {2,3,32,32})
end

-- Doesn't work b/c of an transposition issue - in Caffe we transpose,
-- in Torch we don't.
-- We get a more efficient evaluation using the Caffe method.

-- function Test:testLogSumExp()
--     local m = nn.LogSumExp()
--     check(m, {5, 2})
-- end

-- function Test:testTemporalConvolution()
--     local m = nn.Sequential()
--     m:add(nn.TemporalConvolution(2, 3, 3, 1))
--     check(m, {1,1,8,2})
-- end

function Test:testConvolution()
    local m = nn.SpatialConvolution(3,64,11,11,4,4,2,2)
    check(m, {1, 3, 224, 224})
end

function Test:testReLU()
    local m = nn.ReLU()
    check(m, {10,20})
end

function Test:testSpatialMaxPooling()
    local m = nn.SpatialMaxPooling(3,3,2,2)
    check(m, {1, 3, 224, 224})
end

function Test:testDropout()
    local m = nn.Sequential()
    m:add(nn.SpatialMaxPooling(3,3,2,2))
    m:add(nn.Dropout())
    check(m, {1, 3, 224, 224})
end

function Test:testView()
    local m = nn.View(3 * 5 * 5)
    check(m, {2, 3, 5, 5})
end

function Test:testAlexnet()
   -- this is AlexNet that was presented in the One Weird Trick paper.
    -- http://arxiv.org/abs/1404.5997
    local features = nn.Sequential()
    features:add(nn.SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
    features:add(nn.Dropout())
    features:add(nn.ReLU(true))
    features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
    features:add(nn.SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
    features:add(nn.Dropout())
    features:add(nn.ReLU(true))
    features:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
    features:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
    features:add(nn.Dropout())
    features:add(nn.ReLU(true))
    features:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
    features:add(nn.Dropout())
    features:add(nn.ReLU(true))
    features:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
    features:add(nn.Dropout())
    features:add(nn.ReLU(true))
    features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

    local classifier = nn.Sequential()
    classifier:add(nn.View(256*6*6))

    local s = nn.Sequential()
    s:add(nn.Linear(256*6*6, 4))
    s:add(nn.Dropout())
    s:add(nn.ReLU())
    classifier:add(s)

    local s = nn.Sequential()
    s:add(nn.Linear(4, 4))
    s:add(nn.Dropout())
    s:add(nn.ReLU())

    classifier:add(s)
    classifier:add(nn.Linear(4, 20))
    classifier:add(nn.LogSoftMax())

    local model = nn.Sequential():add(features):add(classifier)
    check(model, {10, 3, 224, 224})
end

local function inception(input_size, config)
    local concat = nn.Concat(2)
    if config[1][1] ~= 0 then
        local conv1 = nn.Sequential()
        conv1:add(nn.SpatialConvolution(input_size, config[1][1],1,1,1,1))
        conv1:add(nn.Dropout())
        conv1:add(nn.ReLU(true))
        concat:add(conv1)
    end

    local conv3 = nn.Sequential()
    conv3:add(nn.SpatialConvolution(input_size, config[2][1],1,1,1,1))
    conv3:add(nn.Dropout())
    conv3:add(nn.ReLU(true))
    conv3:add(nn.SpatialConvolution(config[2][1], config[2][2],3,3,1,1,1,1))
    conv3:add(nn.Dropout())
    conv3:add(nn.ReLU(true))
    concat:add(conv3)

    local conv3xx = nn.Sequential()
    conv3xx:add(nn.SpatialConvolution(input_size, config[3][1],1,1,1,1))
    conv3xx:add(nn.Dropout())
    conv3xx:add(nn.ReLU())
    conv3xx:add(nn.SpatialConvolution(config[3][1], config[3][2],3,3,1,1,1,1))
    conv3xx:add(nn.Dropout())
    conv3xx:add(nn.ReLU())
    conv3xx:add(nn.SpatialConvolution(config[3][2], config[3][2],3,3,1,1,1,1))
    conv3xx:add(nn.Dropout())
    conv3xx:add(nn.ReLU())
    concat:add(conv3xx)

    local pool = nn.Sequential()
    if config[4][1] == 'max' then
        pool:add(nn.SpatialMaxPooling(3,3,1,1,1,1):ceil())
    elseif config[4][1] == 'avg' then
        pool:add(nn.SpatialAveragePooling(3,3,1,1,1,1):ceil())
    else
        error('Unknown pooling')
    end
    if config[4][2] ~= 0 then
        pool:add(nn.SpatialConvolution(input_size, config[4][2],1,1,1,1))
        pool:add(nn.Dropout())
        pool:add(nn.ReLU(true))
    end
    concat:add(pool)

    return concat
end

local function Label(module, label)
    module[label] = true
    return module
end

function Test:testInceptionComponents()
    local incept = inception(3, {{ 1},{ 1, 1},{ 1, 1},{'max', 1}})
    for i=1,#incept.modules do
        print("checking model")
        local m = incept.modules[i]
        print(m)
        check(m, {10, 3, 16, 16})
    end
end

function Test:testInception()
    local incept = inception(3, {{ 1},{ 1, 1},{ 1, 1},{'max', 1}})
    check(incept, {1, 3, 3, 3})
end

Test[g_SLOW .. 'testGoogLeNet'] = function(self)
    local features = nn.Sequential()
    features:add(nn.SpatialConvolution(3,64,7,7,2,2,3,3), 'isLayer1')
    features:add(nn.Dropout()):add(nn.ReLU(true))
    features:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
    features:add(nn.SpatialConvolution(64,64,1,1))
    features:add(nn.Dropout()):add(nn.ReLU(true))
    features:add(nn.SpatialConvolution(64,192,3,3,1,1,1,1))
    features:add(nn.Dropout()):add(nn.ReLU(true))
    features:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
    features:add(inception( 192, {{ 64},{ 64, 64},
                                { 64, 96},{'max', 32}})) -- 3(a)
    features:add(inception( 256, {{ 64},{ 64, 96},
                                { 64, 96},{'max', 64}})) -- 3(b)
    features:add(Label(inception( 320, {{  0},{128,160},
                                      { 64, 96},{'max',  0}}), 'is3c')) -- 3(c)
    features:add(nn.SpatialConvolution(576,576,2,2,2,2))
    features:add(inception( 576, {{224},{ 64, 96},
                                { 96,128},{'max',128}})) -- 4(a)
    features:add(inception( 576, {{192},{ 96,128},
                                { 96,128},{'max',128}})) -- 4(b)
    features:add(inception( 576, {{160},{128,160},
                                {128,160},{'max', 96}})) -- 4(c)
    features:add(Label(inception( 576, {{ 96},{128,192},
                                      {160,192},{'max', 96}}), 'is4d')) -- 4(d)

    local main_branch = nn.Sequential()
    main_branch:add(inception( 576, {{  0},{128,192},
                                   {192,256},{'max',  0}})) -- 4(e)
    main_branch:add(nn.SpatialConvolution(1024,1024,2,2,2,2))
    main_branch:add(nn.Dropout())
    main_branch:add(inception(1024, {{352},{192,320},
                                  {160,224},{'max',128}})) -- 5(a)
    main_branch:add(Label(inception(1024, {{352},{192,320},
                                   {192,224},{'max',128}}), 'is5b')) -- 5(b)
    main_branch:add(nn.SpatialMaxPooling(7,7,1,1))
    main_branch:add(Label(nn.View(1024):setNumInputDims(3), 'isLastFeatures'))
    main_branch:add(Label(nn.Linear(1024,1000), 'isLastLinear'))

    local model = nn.Sequential():add(features):add(main_branch)
    check(model, {1, 3, 224, 224})
end

function Test:testParallelModel()
    local m = nn.Sequential()
    local pt = nn.ParallelTable()
    pt:add(nn.LookupTable(5, 10))
    pt:add(nn.LookupTable(8, 10))
    pt:add(nn.LookupTable(12, 10))
    m:add(pt)
    m:add(nn.JoinTable(1, 2))
    m:add(nn.Reshape(10 * (18 + 18 + 12)))
    m:add(nn.ReLU())
    m:add(nn.Linear(10 * (18 + 18 + 12), 2))
    check_opts(m, {inputs={
                       {name="parallel_1", input_dims={1,18},
                        tensor=torch.FloatTensor(1,18):fill(1)},
                       {name="parallel_2", input_dims={1,18},
                        tensor=torch.FloatTensor(1,18):fill(1)},
                       {name="parallel_3", input_dims={1,12},
                        tensor=torch.FloatTensor(1,12):fill(1)},
    }})
end

LuaUnit:main()
