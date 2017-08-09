--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'
require 'data_augmenter'
require 'hzproc'
local c = require 'trepl.colorize'
local json = require 'json'

tableFromJSON = json.load(opt.classMapping)

paths.dofile('dataset_fast.lua')
paths.dofile('util.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, 'trainCache.t7')
local testCache = paths.concat(opt.cache, 'testCache.t7')
local meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {3, opt.imageSize, opt.imageSize}
local sampleSize = {3, opt.cropSize, opt.cropSize}

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
   if input:size(3) < input:size(2) then
      input = image.scale(input, loadSize[2], loadSize[3] * input:size(2) / input:size(3))
   else
      input = image.scale(input, loadSize[2] * input:size(3) / input:size(2), loadSize[3])
   end
   return input
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]

-- data augmenter
local augmenter = DataAugmenter{nGpu = opt.nGPU}

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)

   local input = (loadImage(path)):cuda()
   -- crop 
   input = augmenter:Crop(input)

   -- do data augmentation with probability opt.PaugTrain
   if torch.uniform() < opt.PaugTrain then 
      input = augmenter:Augment(input)
   end

   assert(input:size(3) == opt.cropSize, 'image size and opt.cropSize dismatch')
   assert(input:size(2) == opt.cropSize, 'image size and opt.cropSize dismatch')

   -- mean/std
   input = augmenter:Normalize(input)
   return input
end

if paths.filep(trainCache) then
   print(c.blue 'Loading train metadata from cache')
   --print('TrainCache: ', trainCache)
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   --assert(trainLoader.paths[1] == paths.concat(opt.data, 'train'),
     --     'cached files dont have the same path as opt.data. Remove your cached files at: '
      --       .. trainCache .. ' and rerun the program')
else
   print('Creating train metadata')
   if opt.classMapping=='imagenet.json' then
     trainLoader = dataLoader{
        paths = {opt.data.."train/"},
        loadSize = loadSize,
        sampleSize = sampleSize,
        split = 100,
        forceClasses = tableFromJSON,
        verbose = true
     }
   else 
     trainLoader = dataLoader{
        paths = {opt.data},
        loadSize = loadSize,
        sampleSize = sampleSize,
        split = 98,
        forceClasses = tableFromJSON,
        verbose = true
     }
   end
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
   print('Train loader classes and nb of classes:')
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   print(nClasses)
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")

end

-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

-- function to load the image
testHook = function(self, path, pTest) 
   local input = (loadImage(path)):cuda()

   -- crop
   input = augmenter:Crop(input)

   -- do data augmentation with probability opt.PaugTest
   if torch.uniform() < pTest then 
     input = augmenter:Augment(input)
   end

   assert(input:size(3) == opt.cropSize, 'image size and opt.cropSize dismatch')
   assert(input:size(2) == opt.cropSize, 'image size and opt.cropSize dismatch')

   -- mean/std
   input = augmenter:Normalize(input)
   collectgarbage()
   return input
end

if paths.filep(testCache) then
   print('Loading test metadata from cache')
   testLoader = torch.load(testCache)
   testLoader.sampleHookTest = testHook
   --assert(testLoader.paths[1] == paths.concat(opt.data, 'val'),
     --     'cached files dont have the same path as opt.data. Remove your cached files at: '
       --      .. testCache .. ' and rerun the program')
else
   print('Creating test metadata')
   if opt.classMapping=='imagenet.json' then
     testLoader = dataLoader{
        paths = {opt.data.."val/"},
        loadSize = loadSize,
        sampleSize = sampleSize,
        split = 100,
        forceClasses = tableFromJSON,
        verbose = true
     }
   else 
     testLoader = dataLoader{
        paths = {opt.data},
        loadSize = loadSize,
        sampleSize = sampleSize,
        split = 98,
        forceClasses = tableFromJSON,
        verbose = true
     }
   end
   torch.save(testCache, testLoader)
   testLoader.sampleHookTest = testHook
end
collectgarbage()

-- End of test loader section

-- Estimate the per-channel mean/std (so that the loaders can normalize appropriately)
if paths.filep(meanstdCache) then
   local meanstd = torch.load(meanstdCache)
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
else
   local tm = torch.Timer()
   local nSamples = 10000
   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local meanEstimate = {0,0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)[1]
      for j=1,3 do
         meanEstimate[j] = meanEstimate[j] + img[j]:mean()
      end
   end
   for j=1,3 do
      meanEstimate[j] = meanEstimate[j] / nSamples
   end
   mean = meanEstimate

   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local stdEstimate = {0,0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)[1]
      for j=1,3 do
         stdEstimate[j] = stdEstimate[j] + img[j]:std()
      end
   end
   for j=1,3 do
      stdEstimate[j] = stdEstimate[j] / nSamples
   end
   std = stdEstimate

   local cache = {}
   cache.mean = mean
   cache.std = std
   torch.save(meanstdCache, cache)
   print('Time to estimate:', tm:time().real)
end
