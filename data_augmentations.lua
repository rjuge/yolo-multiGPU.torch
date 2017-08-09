require 'hzproc'
require 'nn'

local A = {}
local gaussianblurlayers = {}
local validblurs = {}

------------------------------------------------
---- Composition
------------------------------------------------
function A.Compose(transforms)
   return function(input)
      for _, transform in ipairs(transforms) do
         input = transform(input)
      end
      return input
   end
end

function A.RandomOrder(ts)
   return function(input)
      local img = input.img or input
      local order = torch.randperm(#ts)
      for i=1,#ts do
         img = ts[order[i]](img)
      end
      return img
   end
end

---------------------------------------------------
---- Randomization
---------------------------------------------------
function A.RandomHorizontalFlip(prob)
  return function(input)
    if torch.uniform() < prob then
       transform = A.HorizontalFlip(true)
       input = transform(input)
    end
    return input
  end
end

function A.RandomAffine(prob)
  return function(input)
    if torch.uniform() < prob then
       local deg = torch.uniform(-13,13)
       local xshear = torch.uniform(-0.05,0.05)
       local yshear = torch.uniform(-0.05,0.05)
       local scale = torch.uniform(1.0,1.2)
       transform = A.Affine(deg, xshear, yshear, scale)
       input = transform(input)
    end
    return input
  end
end
--[[
function A.RandomCrop(prob)
  return function(input)
    
    if torch.uniform() < prob then
      
      local whichTransform = torch.uniform()
      
      if whichTransform < 0.20 then
   print("Center Crop")
   transform = A.CenterCrop(20)
      elseif 0.20 < whichTransform and whichTransform < 0.40 then
   print("Random Crop Pad")
         transform = A.RandomCropPad(20, 5)
      elseif 0.40 <  whichTransform and whichTransform < 0.60 then
   print("Ten Crop")
   transform = A.TenCrop(20) 
      elseif 0.60 <  whichTransform and whichTransform < 0.80 then
   print("Random Sized Crop")
         transform =  A.RandomSizedCrop(20)
      end
      
      input = transform(input)
      
    end
    return input
  end
end
--]]

function A.RandomHueJitter(prob)
  return function(input)
    if torch.uniform() < prob then
      transform = A.HueJitter(18)
      input = transform(input)
    end
    return input
  end
end

function A.RandomTinge(prob)
  return function(input)
    if torch.uniform() < prob then
      local whichtransform = torch.uniform()
      if whichtransform <= 0.5 then
        transform = A.GreenTinge(0.10)
      else
        transform = A.MagentaTinge(0.05)
      end
      input = transform(input)
    end
    return input
  end
end

function A.RandomAWGN(prob)  
  return function(input)
    if torch.uniform() < prob then
      local whichtransform = torch.uniform()
      
      if whichtransform <= 0.5 then
        --print("Color AWGN")
        transform = A.ColorAWGN(0.07)
      else
        --print("Intensity AWGN")
        transform = A.IntensityAWGN(0.1)
      end
      
      input = transform(input)
    end
    return input
  end
end

function A.RandomBlurAndNoise(pblur, pnoise)
  local transforms = {A.RandomGaussianBlur(pblur), A.RandomAWGN(pnoise)}
  return A.RandomOrder(transforms)
end
  
function A.RandomGaussianBlur(prob)
  return function(input)
    if torch.uniform() < prob then
      
      local idx = torch.random(1, table.getn(validblurs))
      transform = A.GaussianBlur(validblurs[idx])
      
      input = transform(input)
    end
    return input
  end
end

function A.RandomLightning(prob, pca)
  return function(input)
    
    if torch.uniform() < prob then
      
      local whichTransform = torch.uniform()
      
      if whichTransform < 0.20 then
        --print("ColorJitter")
        transform = A.ColorJitter({
           contrast = 0.30,
           brightness = 0.50,
           saturation = 0.50,
         })
       elseif whichTransform < 0.40 then
         --print("Contrast")
         transform = A.Contrast(0.30)
       elseif whichTransform < 0.60 then
         --print("Brightness")
         transform =  A.Brightness(0.50)
       elseif whichTransform < 0.80 then
         --print("Saturation")
         transform =  A.Saturation(0.50)
       else
         --print("Lighting PCA")
         transform = A.Lighting(0.5, pca.eigval, pca.eigvec)
      end
      
      input = transform(input)
    
    end
    return input
  end
end

------------------------------------------------
---- Standardization
------------------------------------------------
function A.ColorNormalize(meanstd)
   return function(img)
      --img = img:clone()
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
         img[i]:div(meanstd.std[i])
      end
      return img
   end
end

------------------------------------------------
---- Scaling & Crops
------------------------------------------------

-- Scales the smaller edge to size
function A.Scale(size, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(input)
      local w, h = input:size(3), input:size(2)
      if (w <= h and w == size) or (h <= w and h == size) then
         return input
      end
      if w < h then
         return image.scale(input, size, h/w * size, interpolation)
      else
         return image.scale(input, w/h * size, size, interpolation)
      end
   end
end

-- Crop to centered rectangle
function A.CenterCrop(size)
   return function(input)
      local w1 = math.ceil((input:size(3) - size)/2)
      local h1 = math.ceil((input:size(2) - size)/2)
      return image.crop(input, w1, h1, w1 + size, h1 + size) -- center patch
   end
end

-- Random crop form larger image with optional zero padding
function A.RandomCropPad(size, padding)
   padding = padding or 0

   return function(input)
      if padding > 0 then
         local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
         temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
         input = temp
      end

      local w, h = input:size(3), input:size(2)
      if w == size and h == size then
         return input
      end

      local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
      local output = image.crop(input, x1, y1, x1 + size, y1 + size)
      assert(output:size(2) == size and output:size(3) == size, 'wrong crop size')
      return output
   end
end

-- Four corner patches and center crop from image and its horizontal reflection
function A.TenCrop(size)
   local centerCrop = A.CenterCrop(size)

   return function(input)
      local w, h = input:size(3), input:size(2)

      local output = {}
      for _, img in ipairs{input, image.hflip(input)} do
         table.insert(output, centerCrop(img))
         table.insert(output, image.crop(img, 0, 0, size, size))
         table.insert(output, image.crop(img, w-size, 0, w, size))
         table.insert(output, image.crop(img, 0, h-size, size, h))
         table.insert(output, image.crop(img, w-size, h-size, w, h))
      end

      -- View as mini-batch
      for i, img in ipairs(output) do
         output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
      end

      return input.cat(output, 1)
   end
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function A.RandomScale(minSize, maxSize)
   return function(input)
      local w, h = input:size(3), input:size(2)

      local targetSz = torch.random(minSize, maxSize)
      local targetW, targetH = targetSz, targetSz
      if w < h then
         targetH = torch.round(h / w * targetW)
      else
         targetW = torch.round(w / h * targetH)
      end

      return image.scale(input, targetW, targetH, 'bicubic')
   end
end

-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
function A.RandomSizedCrop(size)
   local scale = A.Scale(size)
   local crop = A.CenterCrop(size)

   return function(input)
      local attempt = 0
      repeat
         local area = input:size(2) * input:size(3)
         local targetArea = torch.uniform(0.08, 1.0) * area

         local aspectRatio = torch.uniform(3/4, 4/3)
         local w = torch.round(math.sqrt(targetArea * aspectRatio))
         local h = torch.round(math.sqrt(targetArea / aspectRatio))

         if torch.uniform() < 0.5 then
            w, h = h, w
         end

         if h <= input:size(2) and w <= input:size(3) then
            local y1 = torch.random(0, input:size(2) - h)
            local x1 = torch.random(0, input:size(3) - w)

            local out = image.crop(input, x1, y1, x1 + w, y1 + h)
            assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

            return image.scale(out, size, size, 'bicubic')
         end
         attempt = attempt + 1
      until attempt >= 10

      -- fallback
      return crop(scale(input))
   end
end


------------------------------------------------
---- Flip, Rotate
------------------------------------------------
function A.HorizontalFlip(bool)
   return function(input)
      if bool == true then 
   input = hzproc.Flip.Horizon(input)
      end
      collectgarbage()
      return input
   end
end

function A.Affine(deg, xshear, yshear, scale)

   return function(input)
      if deg ~= 0 then
   local affine = hzproc.Affine.RotateArround(deg * math.pi/180, input:size(3)/2, input:size(2)/2)
   torch.mm(affine, affine, hzproc.Affine.ScaleArround(scale, scale, input:size(3)/2, input:size(2)/2))
   torch.mm(affine,affine, hzproc.Affine.ShearArround(xshear, yshear, input:size(3)/2, input:size(2)/2))
   -- affine mapping
   input = hzproc.Transform.Fast(input, affine); 
      end  
      collectgarbage()
      return input
   end
end

------------------------------------------------
---- Lighting
------------------------------------------------


local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function A.Saturation(var)
   local gs
   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function A.Brightness(var)
   local gs
   return function(input)
      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function A.Contrast(var)
   local gs
   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)
      gs:fill(gs[1]:mean())
      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
   
end

function A.ColorJitter(opt)
   local brightness = opt.brightness or 0
   local contrast = opt.contrast or 0
   local saturation = opt.saturation or 0

   local ts = {}
   if brightness ~= 0 then
      table.insert(ts, A.Brightness(brightness))
   end
   if contrast ~= 0 then
      table.insert(ts, A.Contrast(contrast))
   end
   if saturation ~= 0 then
      table.insert(ts, A.Saturation(saturation))
   end

   if #ts == 0 then
      return function(input) return input end
   end

   return A.RandomOrder(ts)
end

-- Lighting noise (AlexNet-style PCA-based noise)
function A.Lighting(alphastd, eigval, eigvec)
   return function(input)
      if alphastd == 0 then
         return input
      end

      local alpha = torch.Tensor(3):type(torch.type(input)):normal(0, alphastd)
      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()
         
         --print(torch.type(alpha))
         --print(torch.type(eigval))
        --print(torch.type(eigvec))
         --print(torch.type(rgb))
         --io.read()

      input = input:clone()
      for i=1,3 do
         input[i]:add(rgb[i])
      end
      return input
   end
end

--------------------------------------------------
---- AWGN
--------------------------------------------------
function A.ColorAWGN(var)

  return function(input)
    noisetensor = noisetensor or torch.Tensor(input:size()):type(torch.type(input))
    noisetensor:normal(0, torch.uniform(0,var))
    --randomkit.normal(noisetensor, 0, torch.uniform(0,var))
    --print(noisetensor)
    input:add(noisetensor)
    return input
  end
  
end

function A.IntensityAWGN(var)
  
  return function(input)
  --print("Intensity AVGN")
   noiseplane = noiseplane or torch.Tensor(input:size(2), input:size(3)):type(torch.type(input)) 
   noiseplane:normal(0,torch.uniform(0,var))
   noisetensor = noisetensor or torch.Tensor(input:size()):type(torch.type(input))
   noisetensor:zero()

   
   --finalnoisetensor = torch.Tensor(input:size()):type(torch.type(input))
   noisetensor[1]:add(noiseplane):mul(0.299)
   noisetensor[2]:add(noiseplane):mul(0.587)
   noisetensor[3]:add(noiseplane):mul(0.114)
   input:add(noisetensor)
   return input
  end
end

--------------------------------------------------
---- COLOR
--------------------------------------------------
function A.HueJitter(var)
  --https://beesbuzz.biz/code/hsv_color_transforms.php
  return function(input)
    
    local angle = torch.random(-var,var) -- shift in hue space (deg)
    local VSU = math.cos(angle*math.pi/180)
    local VSW = math.sin(angle*math.pi/180)
    huejittered = huejittered or torch.Tensor(input:size()):type(torch.type(input))

        
    huejittered[1] = torch.mul(input[1], (.299 + .701*VSU + .168*VSW)):add( torch.mul(input[2], (.587 - .587*VSU + .330*VSW))):add(torch.mul(input[3], (.114 - .114*VSU - .497*VSW)))
    huejittered[2] = torch.mul(input[1], (.299 - .299*VSU - .328*VSW)):add( torch.mul(input[2], (.587 + .413*VSU + .035*VSW))):add( torch.mul(input[3], (.114 - .114*VSU + .292*VSW)))
    huejittered[3] = torch.mul(input[1], (.299 - .3*VSU + 1.25*VSW)):add(torch.mul(input[2], (.587 - .588*VSU - 1.05*VSW))):add( torch.mul(input[3], (.114 + .886*VSU - .203*VSW)))
    return huejittered
    
  end
end

function A.MagentaTinge(var)
  return function(input)
    --output = torch.Tensor(input:size()):copy(input)
    local greenstrength = torch.uniform(0,var)
    local bluestrength = 2*greenstrength
    input[2]:add(-greenstrength)
    input[3]:add(bluestrength)
    return input
  end
end


function A.GreenTinge(var)
  return function(input)
    --output = torch.Tensor(input:size()):copy(input)
    local strength = torch.uniform(0,var)
    input[1]:add(-strength)
    input[2]:add(strength)
    return input
  end
end

--------------------------------------------------
---- Blur
--------------------------------------------------

function A.InitGaussianKernels(maxkernelwidth, nGpu)
  for kw=3, maxkernelwidth do
    if kw % 2 == 1 then
      local stride = 1
      local kernel = image.gaussian(kw, 0.25, 1, true)
      local pad_w = math.floor((kw - stride ) / 2)
      local pad_h = math.floor((kw - stride ) / 2)
      local convlayer = nn.SpatialConvolution(3,3, kw, kw, stride, stride, pad_w, pad_h)
      convlayer.weight:zero()
      convlayer.weight[1][1] = kernel
      convlayer.weight[2][2] = kernel
      convlayer.weight[3][3] = kernel
      convlayer.bias:zero()
      gaussianblurlayers[kw] =  convlayer
      table.insert(validblurs,kw)
    end
  end
  if(nGpu > 0) then
    for k,v in pairs(gaussianblurlayers) do
      v = v:cuda()
    end
  end
end

function A.GaussianBlur(kw)
  return function(input)
    --kernelidx = torch.random(1,table.getn(gaussianblurlayers))
    --print(kernelidx)
    --print(gaussianblurlayers)
    blurred = blurred or torch.Tensor(input:size()):type(torch.type(input))
    convlayer = gaussianblurlayers[kw]
    blurred = convlayer:forward(input)

    --output = torch.conv2(input, kernel 'F')
    return blurred
  end
end

return A
