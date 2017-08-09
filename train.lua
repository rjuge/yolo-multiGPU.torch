--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
require 'util'
--[[
   1. Setup  optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

netMap = torch.load(opt.netMapping, 'b64')
meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')
meanstd = torch.load(meanstdCache)

print("OptimState Created: ")
print(optimState.learningRate)

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsConservative(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
       {  1,     18,     1e-2,   5e-4, },
       { 19,     29,     5e-3,   5e-4  },
       { 30,     43,     1e-3,   0 },
       { 44,     52,     5e-4,   0 },
       { 53,     1e8,   1e-4,   0 },
       }
 

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

local lr, wd
local function paramsLinear(epoch)
--print("inParamsForEpoch")
--print(epoch)
--print(opt.LR)
if opt.LR ~= 0.0 and epoch == 1 then -- if manually specified	
	lr = opt.LR
	return { }
elseif epoch == 1 then
	lr = 0.1
	return { learningRate = lr, weightDecay=1e-4 }, true
elseif epoch > 13 then
	lr = lr * math.pow( 0.95, epoch - 13) 
	wd = 0 
	return { learningRate = lr, weightDecay=wd }, true
end
end

-- 2. Create loggers.
local batchNumber
local top1_epoch, top5_epoch, loss_epoch

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   if opt.regime == 'conservative' then
      local params, newRegime = paramsConservative(epoch)
      if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
      end
   elseif opt.regime == 'linear' then
      local params, newRegime = paramsLinear(epoch)
      if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
      end
   else
      assert(false, 'Regime not supported!')
   end
   --print("paramsForEpoch: ")
   --print("LR: ".. optimState.learningRate)
   --io.read()   
   --print(optimState)
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   top1_epoch = 0
   top5_epoch = 0
   loss_epoch = 0
   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels = trainLoader:sample(opt.batchSize)
	    return inputs, labels
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   top5_epoch = top5_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize
   
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t'
			  .. 'accuracy(%%):\t top-5 %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1_epoch, top5_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   local cp = {}
   local modelToSave = saveDataParallel(model)
   modelToSave = deepCopy(modelToSave):float():clearState()
   cudnn.convert(modelToSave,nn)
   modelToSave = modelToSave:float()
   cp.convnet = modelToSave
   cp.classids = netMap
   cp.img_mean = meanstd.mean
   cp.img_std = meanstd.std
   torch.save(paths.concat(opt.save, 'checkpoint_' .. epoch .. '.t7'), cp)
      
   modelToSave = nil

   collectgarbage()

   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)

   return loss_epoch, top1_epoch

end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()	
   local dataLoadingTime = dataTimer:time().real
   timer:reset()
   
   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)
   
   local err, outputs
   if opt.FT ~= 0 then 
      local inputsFT = torch.CudaTensor()
      inputsFT = base_model:forward(inputs)
      feval = function(x)
	  	model:zeroGradParameters()
	 	outputs = model:forward(inputsFT)
	 	err = criterion:forward(outputs, labels)
	 	local gradOutputs = criterion:backward(outputs, labels)
	 	model:backward(inputsFT, gradOutputs)
	 	return err, gradParameters
      end
   else
      feval = function(x)
	  	model:zeroGradParameters()
	  	outputs = model:forward(inputs)
	 	err = criterion:forward(outputs, labels)
	 	local gradOutputs = criterion:backward(outputs, labels)
	 	model:backward(inputs, gradOutputs)
	 	return err, gradParameters
      end
   end

   if(opt.optimizer == 'sgd') then
--     print(optimState)
--     io.read()
 	 optim.sgd(feval, parameters, optimState)
elseif(opt.optimizer == 'adam') then
     optim.adam(feval, parameters, optimState)
elseif (opt.optimizer == 'adagrad') then
     optim.adagrad(feval, parameters, optimState)
elseif (opt.optimizer == 'nesterov') then
     optim.nag(feval, parameters, optimState)
elseif (opt.optimizer == 'rmsprop') then
     optim.rmsprop(feval, parameters, optimState)
else
	error ("Optimizer: " .. opt.optimizer .. " not supported!")
end


   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   -- top-1 top-5 errors
   local top1 = 0
   local top5 = 0
   do
      local _,prediction_sorted = outputs:float():sort(2, true) -- descending
      for i=1,opt.batchSize do
	 if prediction_sorted[i][1] == labelsCPU[i] then
	    top1_epoch = top1_epoch + 1;
	    top1 = top1 + 1
	 end
	 if isTop5(prediction_sorted[i], labelsCPU[i]) == true then
	    top5_epoch = top5_epoch + 1;
	    top5 = top5 + 1
	 end
      end
      top1 = top1 * 100 / opt.batchSize; 
      top5 = top5 * 100 / opt.batchSize;
   end
   -- Calculate top-1 error, and print information
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, err, top1,
          optimState.learningRate, dataLoadingTime))

   dataTimer:reset()
end

function isTop5(X_hat, X)
   bool = false
   for i=1,5 do
      if X_hat[i] == X then
	 bool=true
      end
   end
   return bool
end

