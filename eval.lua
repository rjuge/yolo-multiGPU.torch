require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
display = require 'display'

local pl = require('pl.import_into')()

local topN = 5

local args = pl.lapp([[
  -m,--model    (default 'model.t7')
  -p,--probe    (default 'probe.jpg')
]])

function imgPreProcess(im)
   im = image.scale(im, 224,224)
   for i=1,3 do -- channels
      if model.img_mean then im[i]:add(-model.img_mean[i]) end
      if model.img_std then im[i]:div(model.img_std[i]) end
   end
   im_t = im:view(1, im:size(1), im:size(2), im:size(3))
   return im_t
end

model = torch.load(args.model)
print '==> Loading Model'
model.convnet:add(nn.SoftMax())
model.convnet:cuda()
cudnn.convert(model.convnet, cudnn)
model.convnet:evaluate()

print(model.convnet)

print '==> Loading and Preprocessing Input Image...'
local probe = image.load(args.probe, 3)
img = imgPreProcess(probe)

print '==> Attempting Forward Pass...'
outputs = model.convnet:forward(img:cuda())
outputs = outputs:float()

local confidences, classnums = outputs:view(-1):sort(true)

topNClasses = {}
topNConfidences = {}

for i=1, topN do
   topNClasses[i] = model.classids[classnums[i]+model.labelOffset]
   topNConfidences[i] = confidences[i]
end

print(topNClasses, topNConfidences)
