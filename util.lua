require 'cunni'
require 'torch'
local ffi=require 'ffi'

function makeDataParallel(model, nGPU)	

	if nGPU > 1 then
	print('converting module to nn.DataParallelTable')
	assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
	if opt.backend == 'cudnn' and opt.cudnnAutotune == 1 then
		local gpu_table = torch.range(1, nGPU):totable()
		local dpt = nn.DataParallelTable(1, true):add(model, gpu_table):threads(function() require 'cudnn'
					cudnn.benchmark = true	end)
		dpt.gradInput = nil
		model = dpt:cuda()
	else
		local model_single = model
		model = nn.DataParallelTable(1)
		for i=1, nGPU do
		cutorch.setDevice(i)
		model:add(model_single:clone():cuda(), i)
		end
		cutorch.setDevice(opt.GPU)
	end
	else
	if (opt.backend == 'cudnn' and opt.cudnnAutotune == 1) then
		require 'cudnn'
		cudnn.benchmark = true
	end
	end

	return model
end

local function cleanDPT(module)
	-- This assumes this DPT was created by the function above: all the
	-- module.modules are clones of the same network on different GPUs
	-- hence we only need to keep one when saving the model to the disk.
	--local newDPT = nn.DataParallelTable(1)
	--cutorch.setDevice(opt.GPU)
	--newDPT:add(module:get(1), opt.GPU)
	return module:get(1)
	--return newDPT
end

function saveDataParallel(model)
	if torch.type(model) == 'nn.DataParallelTable' then
	local temp_model = cleanDPT(model)
	return temp_model
	elseif torch.type(model) == 'nn.Sequential' then
	local temp_model = nn.Sequential()
	for i, module in ipairs(model.modules) do
		if torch.type(module) == 'nn.DataParallelTable' then
		temp_model:add(cleanDPT(module))
		else
		temp_model:add(module)
		end
	end
	return temp_model
	else
	error('This saving function only works with Sequential or DataParallelTable modules.')
	end
end

function loadDataParallel(filename, nGPU)
	if opt.backend == 'cudnn' then
	require 'cudnn'
	end
	local model = torch.load(filename)
	if torch.type(model) == 'nn.DataParallelTable' then
	return makeDataParallel(model:get(1):float(), nGPU)
	elseif torch.type(model) == 'nn.Sequential' then
	for i,module in ipairs(model.modules) do
		if torch.type(module) == 'nn.DataParallelTable' then
		model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
		elseif torch.type(module) == 'nn.Sequential' then
		model.modules[i] = makeDataParallel(module:float(), nGPU)
	end
	end
	return model
	else
	error('The loaded model is not a Sequential or DataParallelTable module.')
	end
end

function deepCopy(tbl)
  -- creates a copy of a network with new modules and the same tensors
	local copy = {}
	for k, v in pairs(tbl) do
    if type(v) == 'table' then
      copy[k] = deepCopy(v)
	 else
		copy[k] = v
	 end
	end
	if torch.typename(tbl) then
	 torch.setmetatable(copy, torch.typename(tbl))
	end
	return copy
end

local function countModules(model)
  if torch.type(model) == 'nn.Sequential' then
	 ft_model = nn.Sequential()
	 local containers = #model
	 local mod = 0
	 local mod_cnt = 0
	 for i=1,containers do
	   if torch.type(model:get(i))=='nn.Sequential' and #model:get(i):listModules() ~= 1 then
	     mod_cnt = mod_cnt + #model:get(i):listModules() - 1
      elseif torch.type(model:get(i))=='nn.Sequential' and #model:get(i):listModules() == 1 then
	     mod_cnt = mod_cnt + 1
	   else
	     mod_cnt = mod_cnt + #model:get(i):listModules()
	   end
    end
	 mod = mod_cnt
	 return mod
  else
	 error'Unsupported model type'
	end
end

function bb_iou(bb1, bb2)
	mx = torch.min(bb1[1]-bb1[3]/2.0, bb2[1]-bb2[3]/2.0)
	Mx = torch.max(bb1[1]+bb1[3]/2.0, bb2[1]+bb2[3]/2.0)
	my = torch.min(bb1[2]-bb1[4]/2.0, bb2[2]-bb2[4]/2.0)
	My = torch.max(bb1[2]+bb1[4]/2.0, bb2[2]+bb2[4]/2.0)
	w1 = bb1[3]
	h1 = bb1[4]
	w2 = bb2[3]
	h2 = bb2[4]

	uw = Mx - mx
	uh = My - my
	cw = w1 + w2 - uw
	ch = h1 + h2 - uh
	carea = 0

	if cw <= 0 or ch <= 0 then
		return 0.0
	end

	area1 = w1 * h1
	area2 = w2 * h2
	carea = cw * ch
	uarea = area1 + area2 - carea
	return carea/uarea
end

	
function bb_ious(bb1, bb2)
	mx = torch.min(bb1[1]-bb1[3]/2.0, bb2[1]-bb2[3]/2.0)
	Mx = torch.max(bb1[1]+bb1[3]/2.0, bb2[1]+bb2[3]/2.0)
	my = torch.min(bb1[2]-bb1[4]/2.0, bb2[2]-bb2[4]/2.0)
	My = torch.max(bb1[2]+bb1[4]/2.0, bb2[2]+bb2[4]/2.0)
	w1 = bb1[3]
	h1 = bb1[4]
	w2 = bb2[3]
	h2 = bb2[4]

	uw = Mx - mx
	uh = My - my
	cw = w1 + w2 - uw
	ch = h1 + h2 - uh
	mask = ((cw <= 0) + (ch <= 0) > 0)
	area1 = w1 * h1
	area2 = w2 * h2
	carea = cw * ch
	carea[mask] = 0
	uarea = area1 + area2 - carea
	return carea/uarea
end

function nms(boxes, nms_thresh)
	if boxes:size(1) == 0 then
		return boxes
	end

	det_confs = torch.zeros(boxes:size(1))
	for i=1,boxes:size(1) do
		det_confs[i] = 1-boxes[i][4] 
	end
	
	_,sortIds = torch.sort(det_confs)
	out_boxes = {}

	for i=1,boxes:size(1) do
		box_i = boxes[sortIds[i]]
		if box_i[4] > 0 then
			table.insert(out_boxes, box_i)
			for j=i+1,boxes:size(1) do
			box_j = boxes[sortIds[j]]
			if bbox_iou(box_i, box_j) > nms_thresh then
			 print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
				box_j[4] = 0
		  end
	   end
		end
	end
	return out_boxes
end

function get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False)

	anchor_step = anchors:size(1)/num_anchors
	if output:dim() == 3 then
		output = output.unsqueeze(0)
	end

	batch = output:size(1)
	assert(output:size(2) == (5+num_classes)*num_anchors)
	h = output:size(3)
	w = output:size(4)

	timer = torch.Timer()
	t0 = timer:time()
	all_boxes = {}
	output = output:view(batch*num_anchors, 5+num_classes, h*w):transpose(1,2):contiguous():view(5+num_classes, batch*num_anchors*h*w)

	
	grid_x = torch.linspace(0, w-1, w)
	grid_x = torch.repeatTensor(grid_x,h,1)
	grid_x = torch.repeatTensor(grid_x,batch*num_anchors, 1, 1):view(batch*num_anchors*h*w):cuda()
	grid_y = torch.linspace(0, h-1, h)
	grid_y = torch.repeatTensor(grid_y,w,1):transpose(1,2)
	grid_y = torch.repeatTensor(grid_y,batch*num_anchors, 1, 1):view(batch*num_anchors*h*w):cuda()
	xs = torch.sigmoid(output[1]) + grid_x
	ys = torch.sigmoid(output[2]) + grid_y

	anchor_w = torch.Tensor(anchors):view(num_anchors, anchor_step):select(2, torch.LongTensor({1}))
	anchor_h = torch.Tensor(anchors):view(num_anchors, anchor_step):select(2, torch.LongTensor({2}))
	anchor_w = torch.repeatTensor(anchor_w,batch, 1)
	anchor_w = torch.repeatTensor(anchor_w,1, 1, h*w):view(batch*num_anchors*h*w):cuda()

	anchor_h = torch.repeatTensor(anchor_h,batch, 1)
	anchor_w = torch.repeatTensor(anchor_h,1, 1, h*w):view(batch*num_anchors*h*w):cuda()
	
	ws = torch.exp(output[3]) * anchor_w
	hs = torch.exp(output[4]) * anchor_h

	det_confs = torch.sigmoid(output[5])

	cls_confs = torch.nn.Softmax(output[{{6,6+num_classes-1}}].transpose(1,2))
	cls_max_confs, cls_max_ids = torch.max(cls_confs, 2)
	cls_max_confs = cls_max_confs:view(-1)
	cls_max_ids = cls_max_ids:view(-1)
	t1 = timer:time()

	sz_hw = h*w
	sz_hwa = sz_hw*num_anchors
	det_confs = convert2cpu(det_confs)
	cls_max_confs = convert2cpu(cls_max_confs)
	cls_max_ids = convert2cpu_long(cls_max_ids)
	xs = convert2cpu(xs)
	ys = convert2cpu(ys)
	ws = convert2cpu(ws)
	hs = convert2cpu(hs)

	if validation then
		cls_confs = convert2cpu(cls_confs:view(-1, num_classes))
	end
	t2 = timer:time()
	
	for b=1,batch do
		boxes = {}
		for cy=1,h do
			for cx=1,w do
			for i=1,num_anchors do
		    ind = b*sz_hwa + i*sz_hw + cy*w + cx
				det_conf =	det_confs[ind]
		    if only_objectness then
				conf =	det_confs[ind]
				else
				conf = det_confs[ind] * cls_max_confs[ind]
		    end
		    if conf > conf_thresh then
				bcx = xs[ind]
				bcy = ys[ind]
				bw = ws[ind]
				bh = hs[ind]
				cls_max_conf = cls_max_confs[ind]
				cls_max_id = cls_max_ids[ind]
				box = {bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id}
		      if (not only_objectness) and validation then
				  for c=1,num_classes do
					 tmp_conf = cls_confs[ind][c]
					 if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh then
						table.insert(box,tmp_conf)
						table.insert(box,c)
			       end
		        end
		      end
		    table.insert(boxes,box)
		    end
		  end
	   end
		end
		table.insert(all_boxes,boxes)
	end
	t3 = time.time()
	if false then
		print('---------------------------------')
		print('matrix computation : ', (t1-t0))
		print('		gpu to cpu : ',(t2-t1))
		print('	boxes filter : ',(t3-t2))
		print('---------------------------------')
	end
	return all_boxes
end

function convert2cpu(gpu_matrix)
	return torch.FloatTensor(gpu_matrix:size()):copy(gpu_matrix)
end

function convert2cpu_long(gpu_matrix):
	return torch.LongTensor(gpu_matrix:size()):copy(gpu_matrix)
end


























