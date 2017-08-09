--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local batchNumber
local top1_center, top5_center, loss
local timer = torch.Timer()

--local features = {}
--local l = {}

local function tableConcat(t1, t2)
   for i=1, t2:size(1) do
      t1[#t1 + 1]=t2[i]
   end
   return t1
end

function test(pTest)
   print(string.format('==> doing epoch on %.2f (%%) augmented validation data:', pTest))
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   if opt.FT ~= 0 then
      base_model:evaluate()
   end
   model:evaluate()

   top1_center = 0
   top5_center = 0
   loss = 0
   for i = 1, math.ceil(nTest/opt.batchSize) do -- nTest is set in data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = math.min(nTest, indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = testLoader:get(indexStart, indexEnd, pTest)
            return inputs, labels
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_center = top1_center * 100 / nTest
   top5_center = top5_center * 100 / nTest
   loss = loss / nTest -- because loss is calculated per batch

   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t '
			  .. 'accuracy [Center](%%):\t top-5 %.2f\t ',
                       epoch, timer:time().real, loss, top1_center, top5_center))

   print('\n')

   return loss, top1_center
end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch(inputsCPU, labelsCPU)
   collectgarbage()
   batchNumber = batchNumber + opt.batchSize

   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   if opt.FT ~= 0 then
      local inputsFT = torch.CudaTensor()
      inputsFT = base_model:forward(inputs)
      local outputs = model:forward(inputsFT)
      local err = criterion:forward(outputs, labels)
      cutorch.synchronize()
      local pred = outputs:float()
      
      loss = loss + err * outputs:size(1)
      
      local _, pred_sorted = pred:sort(2, true)
      for i=1,pred:size(1) do
	 local g = labelsCPU[i]
	 if pred_sorted[i][1] == g then top1_center = top1_center + 1 end
	 if isTop5(prediction_sorted[i], g) == true then top5_center = top5_center + 1 end
      end
      if batchNumber % 1024 == 0 then
	 print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, nTest))
      end
   else
      local outputs = model:forward(inputs)
      --tableConcat(l, labelsCPU)
      --tableConcat(features, model:get(1):get(41).output:float())
      --if #l == nTest then
	-- torch.save('labels.t7', l)
	 --torch.save('features.t7', features)
      --end
      local err = criterion:forward(outputs, labels)
      cutorch.synchronize()
      local pred = outputs:float()
      
      loss = loss + err * outputs:size(1)
      
      local _, pred_sorted = pred:sort(2, true)
      for i=1,pred:size(1) do
	 local g = labelsCPU[i]
	 if pred_sorted[i][1] == g then top1_center = top1_center + 1 end 
	 if isTop5(pred_sorted[i], g) == true then top5_center = top5_center + 1 end
      end
      if batchNumber % 1024 == 0 then
	 print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, nTest))
      end
   end
end
