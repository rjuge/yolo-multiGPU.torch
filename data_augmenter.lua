local Augmentations = require 'data_augmentations'
require 'hzproc'

local DataAugmenter = torch.class('DataAugmenter')

--------------------------------------------------
----- Initialization
--------------------------------------------------
function DataAugmenter:__init(opt)
  
  self.pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

  self.meanstd = 
  {
    mean = { 0.485, 0.456, 0.406 },
    std = { 0.229, 0.224, 0.225 },
  }

  self.maxblurkernelwidth = 7
  --self.gaussianblurlayers = {}
  --self.validblurs = {}

  if(opt.nGpu > 0) then
    self.pca.eigval = self.pca.eigval:cuda()
    self.pca.eigvec = self.pca.eigvec:cuda()
  end
  
  Augmentations.InitGaussianKernels(7, opt.nGpu)
  
  self.augmentationPipeline = Augmentations.Compose
  {
    Augmentations.RandomLightning(0.80, self.pca),
    Augmentations.RandomHueJitter(0.5),
    Augmentations.RandomTinge(0.5),
    Augmentations.RandomBlurAndNoise(0.50, 0.75),
    --Augmentations.RandomHorizontalFlip(0.6),
    -Augmentations.RandomAffine(1.0),
  }
  collectgarbage()
end

--------------------------------------------------
----- External Interface
--------------------------------------------------
function DataAugmenter:Augment(input)
  --print("Augmenting")
  input = self.augmentationPipeline(input)
  return input
  
end

function DataAugmenter:Crop(input)

   local iW = input:size(3)
   local iH = input:size(2)

   local oW = 224
   local oH = 224

   -- crop
   x1, y1 = torch.random(0, iW - oW), torch.random(0, iH - oH)
   input = hzproc.Crop.Fast(input, oW, oH, x1, y1, x1+oW, y1+oH)
   collectgarbage()
   return input
end

function DataAugmenter:Normalize(input)

   --imagenet mean and std
   local mean = self.meanstd['mean']
   local std = self.meanstd['std']
   
   for i=1,3 do -- channels
      input[{{i},{},{}}]:add(-mean[i])
      input[{{i},{},{}}]:div(std[i]) 
   end
   collectgarbage()
   return input
end
