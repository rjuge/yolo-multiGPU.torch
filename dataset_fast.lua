-- Source: https://github.com/soumith/imagenet-multiGPU.torch/blob/master/dataset.lua
-- Modified by Brandon Amos in Sept 2015 for OpenFace by adding
-- `samplePeople` and `sampleTriplet`.

require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'
tds = require 'tds'
local c = require 'trepl.colorize'
local string = require 'string'
require 'sort'

local dataTrain = torch.load('annotations/instances_train2014.tds.t7')
local dataVal = torch.load('annotations/instances_val2014.tds.t7')

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
   pack=true,
   help=[[
     A dataset class for images in a flat folder structure (folder-name is class-name).
     Optimized for extremely large datasets (upwards of 14 million images).
     Tested only on Linux (as it uses command-line linux utilities to scale up)
]],
   {check=function(paths)
       local out = true;
       for _,v in ipairs(paths) do
          if type(v) ~= 'string' then
             print('paths can only be of string input');
             out = false
          end
       end
       return out
   end,
    name="paths",
    type="table",
    help="Multiple paths of directories with images"},

   {name="sampleSize",
    type="table",
    help="a consistent sample size to resize the images"},

   {name="samplingMode",
    type="string",
    help="Sampling mode: random | balanced ",
    default = "balanced"},

   {name="verbose",
    type="boolean",
    help="Verbose mode during initialization",
    default = false},

   {name="loadSize",
    type="table",
    help="a size to load the images to, initially",
    opt = true},

   {name="forceClasses",
    type="table",
    help="If you want this loader to map certain classes to certain indices, "
       .. "pass a classes table that has {classname : classindex} pairs."
       .. " For example: {3 : 'dog', 5 : 'cat'}"
       .. "This function is very useful when you want two loaders to have the same "
    .. "class indices (trainLoader/testLoader for example)",
    default = tableFromJSON,
    opt = true},

   {name="sampleHookTrain",
    type="function",
    help="applied to sample during training(ex: for lighting jitter). "
       .. "It takes the image path as input",
    opt = true},

   {name="sampleHookTest",
    type="function",
    help="applied to sample during testing",
    opt = true},
}

function dataset:__init(...)

   -- argcheck
   local args =  initcheck(...)
   print(args)
   for k,v in pairs(args) do self[k] = v end

   if not self.loadSize then self.loadSize = self.sampleSize; end

   if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
   if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end
   
   -- find class names
   print('finding class names')
   self.classes = tds.Hash()
   self.classes_cnt = 0
   for _,v in pairs(dataTrain['categories']) do
     self.classes[v['id']] = v['name']
	   self.classes_cnt = self.classes_cnt + 1 
   end

   print(self.classes_cnt .. ' class names found')
   self.classIndices = tds.Hash()
   for k,v in pairs(self.classes) do
      self.classIndices[v] = k
   end

   -- find the image path names
   print('Finding path for each image')
   self.imagePathTrain = torch.CharTensor()  -- path to each image in dataset
   self.imageClassTrain = torch.LongTensor() -- class index of each image (class index in self.classes)
   self.classList = tds.Hash()          -- index of imageList to each image of a particular class
   self.classListSample = self.classList -- the main list used when sampling data
   local countsTrain = tds.Hash()
   local maxPathLengthTrain = 0

   print('Calculating maximum class name length and counting files')
   local lengthTrain = 0
   local fullPathsTrain = tds.Hash()
   local fullPathsSizeTrain = 0
   
   for k,v in pairs(dataTrain['images']) do
     local count = 0
     local folderPath = opt.data .. 'train2014/'
     local fullPath = folderPath .. v['file_name']
     maxPathLengthTrain = math.max(fullPath:len(), maxPathLengthTrain)
     count = count + 1
     lengthTrain = lengthTrain + 1
     fullPathsTrain[fullPathsSizeTrain + 1] = fullPath
	   fullPathsSizeTrain = fullPathsSizeTrain + 1
     countsTrain[v] = count
   end

   assert(lengthTrain > 0, "Could not find any image file in the given input paths")
   assert(maxPathLengthTrain > 0, "paths of files are length 0?")
   maxPathLengthTrain = maxPathLengthTrain + 1
   self.imagePathTrain:resize(lengthTrain, maxPathLengthTrain):fill(0)
   local s_data = self.imagePathTrain:data()
   local count = 0
   for i,line in pairs(fullPathsTrain) do
     ffi.copy(s_data, line)
     s_data = s_data + maxPathLengthTrain
     if self.verbose and count % 10000 == 0 then
        xlua.progress(count, lengthTrain)
     end
     count = count + 1
   end
   self.numSamplesTrain = self.imagePathTrain:size(1)
   if self.verbose then print(self.numSamplesTrain ..  ' training samples found.') end

   self.idToClassTrain = tds.Hash()
   self.idToBboxTrain = tds.Hash()
   for _,v in pairs(dataTrain['annotations']) do
     self.idToClassTrain[v['image_id']] = v['category_id']
     self.idToBboxTrain[v['image_id']] = v['bbox']
   end

   self.idToSizeTrain = tds.Hash()
   self.idToPathTrain = tds.Hash()
   for _,v in pairs(dataTrain['images']) do
     local folderPath = opt.data .. 'train2014/'
     self.idToSizeTrain[v['id']] = {v['width'],v['height']}
     self.idToPathTrain[v['id']] = folderPath .. v['file_name']
   end

   self.classToPathTrain = tds.Hash()
   self.classToIdTrain = tds.Hash()
   local classIdsTableTrain = {}
   for k,v in pairs(self.idToClassTrain) do
     self.classToIdTrain[v] = k
   end

   local 
   for k,v in pairs(self.classToIdTrain) do
     self.classToPathTrain[k] = self.idToPathTrain[v]
   end




   --==========================================================================
   print('Updating classList and imageClass appropriately')
   self.imageClassTrain:resize(self.numSamplesTrain)
   local runningIndex = 0
   for i=1, self.classes_cnt do
	  if self.verbose then xlua.progress(i, self.classes_cnt) end
      local clsLength = countsTrain[classPaths[self.classes[i]]]
      if clsLength == 0 then
         error('Class has zero samples: ' .. self.classes[i])
      else
         self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + clsLength, clsLength):long()
         self.imageClass[{{runningIndex + 1, runningIndex + clsLength}}]:fill(i)
      end
      runningIndex = runningIndex + clsLength
   end

   --==========================================================================
   if opt.classMapping == 'imagenet.json' then
      self.classListTrain = {}
      self.classListSample = self.classListTrain
      local totalTrainSamples = 0
      for i=1,self.classes_cnt do
         local list = self.classList[i]
         count = self.classList[i]:size(1)
         local perm = torch.randperm(count)
         self.classListTrain[i] = torch.LongTensor(count)
         for j=1,count do
            self.classListTrain[i][j] = list[perm[j]]
         end
         totalTrainSamples = totalTrainSamples + self.classListTrain[i]:size(1)
      end
      if self.paths[1] == opt.data.."val/" then
         local totalTestSamples = totalTrainSamples
         self.classListTest = self.classListTrain
         self.testIndices = torch.LongTensor(totalTestSamples)
         self.testIndicesSize = totalTestSamples
         local tdata = self.testIndices:data()
         local tidx = 0
         for i=1,self.classes_cnt do
            local list = self.classListTest[i]
            if list:dim() ~= 0 then
               local ldata = list:data()
               for j=0,list:size(1)-1 do
                  tdata[tidx] = ldata[j]
                  tidx = tidx + 1
               end
            end
         end
      else
         self.testIndicesSize = 0
      end
   else
   -- for train and val in unique folder
   if self.split == 100 then
      self.testIndicesSize = 0
   else
      print('Splitting training and test sets to a ratio of '
               .. self.split .. '/' .. (100-self.split))
      self.classListTrain = {}
      self.classListTest  = {}
      self.classListSample = self.classListTrain
      local totalTestSamples = 0
      -- split the classList into classListTrain and classListTest
      for i=1,self.classes_cnt do
         local list = self.classList[i]
         count = self.classList[i]:size(1)
         local splitidx = math.floor((count * self.split / 100) + 0.5) -- +round
         local perm = torch.randperm(count)
         self.classListTrain[i] = torch.LongTensor(splitidx)
         for j=1,splitidx do
            self.classListTrain[i][j] = list[perm[j]]
         end
         if splitidx == count then -- all samples were allocated to train set
            self.classListTest[i]  = torch.LongTensor()
         else
            self.classListTest[i]  = torch.LongTensor(count-splitidx)
            totalTestSamples = totalTestSamples + self.classListTest[i]:size(1)
            local idx = 1
            for j=splitidx+1,count do
               self.classListTest[i][idx] = list[perm[j]]
               idx = idx + 1
            end
         end
      end
      -- Now combine classListTest into a single tensor
      self.testIndices = torch.LongTensor(totalTestSamples)
      self.testIndicesSize = totalTestSamples
      local tdata = self.testIndices:data()
      local tidx = 0
      for i=1,self.classes_cnt do
         local list = self.classListTest[i]
         if list:dim() ~= 0 then
            local ldata = list:data()
            for j=0,list:size(1)-1 do
               tdata[tidx] = ldata[j]
               tidx = tidx + 1
            end
         end
      end
   end
end
end

-- size(), size(class)
function dataset:size(class, list)
   list = list or self.classList
   if not class then
      return self.numSamples
   elseif type(class) == 'string' then
      return list[self.classIndices[class]]:size(1)
   elseif type(class) == 'number' then
      return list[class]:size(1)
   end
end

-- size(), size(class)
function dataset:sizeTrain(class)
   if class then
      return self:size(class, self.classListTrain)
   else
      return self.numSamples - self.testIndicesSize
   end
end

-- size(), size(class)
function dataset:sizeTest(class)
   if class then
      return self:size(class, self.classListTest)
   else
      return self.testIndicesSize
   end
end

-- by default, just load the image and return it
function dataset:defaultSampleHook(imgpath)
   local out = image.load(imgpath, 3, 'float')
   out = image.scale(out, self.sampleSize[3], self.sampleSize[2])
   return out
end

-- getByClass
function dataset:getByClass(classId)
   local index = math.ceil(torch.uniform() * self.classToPaths[classId]:nElement())
   local imgpath = ffi.string(torch.data(self.imagePath[self.classToPaths[classId][index]]))
   return self:sampleHookTrain(imgpath)
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, scalarTable, dataTableSize, scalarTableSize)
   local data, scalarLabels
   local quantity = scalarTableSize
   local samplesPerDraw
   if dataTable[1]:dim() == 3 then samplesPerDraw = 1
   else samplesPerDraw = dataTable[1]:size(1) end
   if quantity == 1 and samplesPerDraw == 1 then
      data = dataTable[1]
      scalarLabels = scalarTable[1]
   else
      data = torch.Tensor(quantity * samplesPerDraw,
                          self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
      scalarLabels = torch.LongTensor(quantity * samplesPerDraw)
      for i=1,dataTableSize do
         local idx = (i-1)*samplesPerDraw
	 data[{{idx+1,idx+samplesPerDraw}}]:copy(dataTable[i])
         scalarLabels[{{idx+1,idx+samplesPerDraw}}]:fill(scalarTable[i])
      end
   end
   return data, scalarLabels
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
   if self.split == 0 then
      error('No training mode when split is set to 0')
   end
   quantity = quantity or 1
   local dataTable = {}
   local scalarTable = {}
   local dataTableSize = 0
   local scalarTableSize = 0
   for _=1,quantity do
      local classId = torch.random(1, self.classes_cnt)
      local out = self:getByClass(classId)
      dataTable[dataTableSize + 1] = out
      scalarTable[scalarTableSize + 1] = classId
      dataTableSize = dataTableSize + 1
      scalarTableSize = scalarTableSize + 1
   end
   local data, scalarLabels = tableToOutput(self, dataTable, scalarTable, dataTableSize, scalarTableSize) 
   return data, scalarLabels
end

function dataset:get(i1, i2, pVal)
   local indices, quantity
   if type(i1) == 'number' then
      if type(i2) == 'number' then -- range of indices
         indices = torch.range(i1, i2);
         quantity = i2 - i1 + 1;
      else -- single index
         indices = {i1}; quantity = 1
      end
   elseif type(i1) == 'table' then
      indices = i1; quantity = #i1;         -- table
   elseif (type(i1) == 'userdata' and i1:nDimension() == 1) then
      indices = i1; quantity = (#i1)[1];    -- tensor
   else
      error('Unsupported input types: ' .. type(i1) .. ' ' .. type(i2))
   end
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local dataTable = {}
   local scalarTable = {}
   local dataTableSize = 0
   local scalarTableSize = 0
   for i=1,quantity do
      -- load the sample
      local idx = self.testIndices[indices[i]]
      local imgpath = ffi.string(torch.data(self.imagePath[idx]))
      local out = self:sampleHookTest(imgpath, pVal)
      table.insert(dataTable, out)
      table.insert(scalarTable, self.imageClass[idx])
      dataTableSize = dataTableSize + 1
      scalarTableSize = scalarTableSize + 1
   end
   local data, scalarLabels = tableToOutput(self, dataTable, scalarTable, dataTableSize, scalarTableSize) 
   collectgarbage()
   return data, scalarLabels
end

function dataset:test(quantity)
   if self.split == 100 then
      error('No test mode when you are not splitting the data')
   end
   local i = 1
   local n = self.testIndicesSize
   local qty = quantity or 1
   return function ()
      if i+qty-1 <= n then
         local data, scalarLabelss = self:get(i, i+qty-1)
         i = i + qty
         return data, scalarLabelss
      end
   end
end

return dataset
