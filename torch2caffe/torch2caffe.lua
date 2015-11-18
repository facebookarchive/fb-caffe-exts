--[[
Copyright (c) 2015-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
--]]
local t2c = require 'torch2caffe.lib'
local pl = require('pl.import_into')()

local opt = pl.lapp[[
   --input (default "") Input model file
   --preprocessing (default "") Preprocess the model
   --prototxt (default "") Output prototxt model file
   --caffemodel (default "") Output model weights file
   --format (default "lua") Format: lua | luathrift
   --input-tensor (default "") (Optional) Predefined input tensor
   --verify (default "") (Optional) Verify existing
   <input_dims...> (default 0)  Input dimensions (e.g. 10N x 3C x 227H x 227W)
]]

t2c.main(opt)
