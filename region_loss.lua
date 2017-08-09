require 'util'
require 'math'

function build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen)

	nB = target:size(1)
	nA = num_anchors
	nC = num_classes
	anchor_step = anchors:size(1)/num_anchors
	conf_mask	= torch.ones(nB, nA, nH, nW) * noobject_scale
	coord_mask = torch.zeros(nB, nA, nH, nW)
	cls_mask	 = torch.zeros(nB, nA, nH, nW)
	tx				 = torch.zeros(nB, nA, nH, nW)
	ty				 = torch.zeros(nB, nA, nH, nW)
	tw				 = torch.zeros(nB, nA, nH, nW)
	th				 = torch.zeros(nB, nA, nH, nW)
	tconf			= torch.zeros(nB, nA, nH, nW)
	tcls			 = torch.zeros(nB, nA, nH, nW)

	nAnchors = nA*nH*nW
  nPixels	= nH*nW

  for b=1,nB do
    cur_pred_boxes = pred_boxes[{{b*nAnchors,(b+1)*nAnchors-1}}]:transpose(1,2)
    cur_ious = torch.zeros(nAnchors)
    for t=1,50 do
      if target[b][t*5+1] == 0 then
        break
      end
      gx = target[b][t*5+1]*nW
      gy = target[b][t*5+2]*nH
      gw = target[b][t*5+3]*nW
      gh = target[b][t*5+4]*nH

      cur_gt_boxes = torch.FloatTensor({gx,gy,gw,gh})
      cur_gt_boxes = torch.repeatTensor(cur_gt_boxes,nAnchors,1):transpose(1,2)
      cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes))

    end
    conf_mask[b][cur_ious>sil_thresh] = 0
  end
  if seen<12800 then
    if anchor_step == 4 then
      tx = torch.FloatTensor(anchors):view(nA, anchor_step):select(2, torch.LongTensor({3}))
      tx = tx:view(1,nA,1,1)
      tx = torch.repeatTensor(tx,nB,1,nH,nW)
    else
      tx:fill(0.5)
      ty:fill(0.5)
    end
    tx:zero()
    ty:zero()
    coord_mask:fill(1)
  end

  nGT = 0
  nCorrect = 0
  for b=1,nB do
    for t=1,50 do
      if target[b][t*5+1] == 0 then
        break
      end

      nGT = nGT +1
      best_iou = 0.0
      best_n = -1
      min_dist = 10000
      gx = target[b][t*5+1] * nW
      gy = target[b][t*5+2] * nH
      gi = math.floor(gx)
      gj = math.floor(gy)
      gw = target[b][t*5+3]*nW
      gh = target[b][t*5+4]*nH
      gt_box = {0, 0, gw, gh}
      for n=1,nA do
        aw = anchors[anchor_step*n]
        ah = anchors[anchor_step*n+1]
        anchor_box = {0, 0, aw, ah}
        iou  = bbox_iou(anchor_box, gt_box)
        if anchor_step == 4 then
          ax = anchors[anchor_step*n+2]
          ay = anchors[anchor_step*n+3]
          dist = torch.pow(((gi+ax) - gx), 2) + torch.pow(((gj+ay) - gy), 2)
        end
        if iou > best_iou then
          best_iou = iou
          best_n = n
        elseif anchor_step==4 and iou == best_iou and dist < min_dist then
          best_iou = iou
          best_n = n
          min_dist = dist
        end
      end

      gt_box = {gx, gy, gw, gh}
      pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]

      coord_mask[b][best_n][gj][gi] = 1
      cls_mask[b][best_n][gj][gi] = 1
      conf_mask[b][best_n][gj][gi] = object_scale
      tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
      ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
      tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
      th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])
      iou = bbox_iou(gt_box, pred_box)
      tconf[b][best_n][gj][gi] = iou
      tcls[b][best_n][gj][gi] = target[b][t*5]
      if iou > 0.5 then
        nCorrect = nCorrect + 1
      end

  return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

end

local RegionLoss, Parent = torch.class('nn.RegionLoss', 'nn.Criterion')

function RegionLoss:__init(num_classes=0, anchors={}, num_anchors=1)
  Parent.__init(self)
  self.num_classes = num_classes
  self.anchors = anchors
  self.num_anchors = num_anchors
  self.anchor_step = len(anchors)/num_anchors
  self.coord_scale = 1
  self.noobject_scale = 1
  self.object_scale = 5
  self.class_scale = 1
  self.thresh = 0.6
  self.seen = 0
  
  self.criterion_x = nn.MSECriterion():cuda()
  self.criterion_x.sizeAverage = false
                                                  
  self.criterion_y = nn.MSECriterion():cuda()
  self.criterion_y.sizeAverage = false
                                                  
  self.criterion_w = nn.MSECriterion():cuda()
  self.criterion_w.sizeAverage = false
                                                  
  self.criterion_h = nn.MSECriterion():cuda()
  self.criterion_h.sizeAverage = false
                                                  
  self.criterion_conf = nn.MSECriterion():cuda()
  self.criterion_conf.sizeAverage = false
                                                  
  self.criterion_cls = nn.CrossEntropyCriterion():cuda()
  self.criterion_cls.sizeAverage = false
end

function RegionLoss:updateOutput(output, target)
  nB = output.data:size(1)
  nA = self.num_anchors
  nC = self.num_classes
  nH = output.data:size(3)
  nW = output.data:size(4)

  output = output:view(nB, nA, (5+nC), nH, nW)
  x = torch.sigmoid(output:select(3, torch.LongTensor({1}):cuda()).view(nB, nA, nH, nW))
  y = torch.sigmoid(output:select(3, torch.LongTensor({2}):cuda()):view(nB, nA, nH, nW))
  w = output:select(3, torch.LongTensor({3}):cuda()):view(nB, nA, nH, nW)
  h = output:select(3, torch.LongTensor({4}):cuda()):view(nB, nA, nH, nW)
  conf = torch.sigmoid(output:select(3, torch.LongTensor({5}):cuda()):view(nB, nA, nH, nW))
  cls = output:select(3, torch.linspace(5,5+nC-1,nC):long():cuda())
  cls  = cls:view(nB*nA, nC, nH*nW):transpose(2,3):contiguous():view(nB*nA*nH*nW, nC)

  pred_boxes = torch.FloatTensor(4, nB*nA*nH*nW):cuda()
  grid_x = torch.linspace(0, nW-1, nW)
  grid_x = torch.repeatTensor(grid_x,nH,1)
  grid_x = torch.repeatTensor(grid_x,nB*nA, 1, 1):view(nB*nA*nH*nW):cuda()
  grid_y = torch.linspace(0, nH-1, nH)
  grid_y = torch.repeatTensor(grid_y,nW,1):transpose(1,2)
  grid_y = torch.repeatTensor(grid_y,nB*nA, 1, 1):view(nB*nA*nH*nW):cuda()
  anchor_w = torch.Tensor(self.anchors):view(nA, self.anchor_step):select(2, torch.LongTensor({1})):cuda()
  anchor_h = torch.Tensor(self.anchors):view(nA, self.anchor_step):select(2, torch.LongTensor({2})):cuda()
  anchor_w = torch.repeatTensor(anchor_w,nB, 1)
  anchor_w = torch.repeatTensor(anchor_w,1, 1, nH*nW):view(nB*nA*nH*nW)
  anchor_h = torch.repeatTensor(anchor_h,nB, 1)
  anchor_h = torch.repeatTensor(anchor_h,1, 1, nH*nW):view(nB*nA*nH*nW)
  pred_boxes[1] = x.data + grid_x
  pred_boxes[2] = y.data + grid_y
  pred_boxes[3] = torch.exp(w.data) * anchor_w
  pred_boxes[4] = torch.exp(h.data) * anchor_h
  pred_boxes = convert2cpu(pred_boxes:transpose(1,2):contiguous():view(-1,4))

  nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)

  cls_mask = (cls_mask == 1)
  nProposals = int((conf > 0.25):sum().data[0]) --TO MODIFY

  tx = tx:cuda()
  ty = ty:cuda()
  tw = tw:cuda()
  th = th:cuda()
  tconf = tconf:cuda()
  tcls = tcls:view(-1)[cls_mask]:long():cuda()

  coord_mask = coord_mask:cuda()
  conf_mask = conf_mask:cuda():sqrt()
  cls_mask = cls_mask:view(-1, 1)
  cls_mask = torch.repeatTensor(cls_mast,1,nC):cuda()
  cls = cls[cls_mask]:view(-1, nC)

  loss_x = self.coord_scale * self.criterion_x:updateOutput(x*coord_mask, tx*coord_mask)/2.0
  loss_y = self.coord_scale * self.criterion_y:updateOutput(y*coord_mask, ty*coord_mask)/2.0
  loss_w = self.coord_scale * self.criterion_w:updateOutput(w*coord_mask, tw*coord_mask)/2.0
  loss_h = self.coord_scale * self.criterion_h:updateOutput(h*coord_mask, th*coord_mask)/2.0
  loss_conf = self.criterion_conf:updateOutput(conf*conf_mask, tconf*conf_mask)/2.0
  loss_cls = self.class_scale * self.criterion_cls:updateOutput(cls, tcls)

  loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

  return loss
end


function RegionLoss:updateGradInput(input, gradOutput)

end
