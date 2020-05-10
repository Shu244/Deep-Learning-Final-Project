import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.core import (delta2bbox, multiclass_nms, bbox_target,bbox_overlaps,
                        weighted_cross_entropy, weighted_smoothl1, accuracy)
from ..registry import HEADS


@HEADS.register_module
class TrackHead(nn.Module):
    """Tracking head, predict tracking features and match with reference objects
       Use dynamic option to deal with different number of objects in different
       images. A non-match entry is added to the reference objects with all-zero 
       features. Object matched with the non-match entry is considered as a new
       object.
    """

    def __init__(self,
                 with_avg_pool=False,
                 num_fcs = 2,
                 in_channels=256,
                 roi_feat_size=7,
                 fc_out_channels=1024,
                 match_coeff=None,
                 bbox_dummy_iou=0,
                 dynamic=True
                 ):
        super(TrackHead, self).__init__()
        self.in_channels = in_channels
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = roi_feat_size
        self.match_coeff = match_coeff
        self.bbox_dummy_iou = bbox_dummy_iou
        self.num_fcs = num_fcs
        self.prev_shift=None
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= (self.roi_feat_size * self.roi_feat_size) 
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):

            in_channels = (in_channels
                          if i == 0 else fc_out_channels)
            fc =nn.Linear(in_channels, fc_out_channels)
            self.fcs.append(fc)

        fc = nn.Linear(1, 2)
        self.fcs.append(fc)
        
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        self.dynamic=dynamic

    def init_weights(self):
        for fc in self.fcs[:-1]:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)
        fc = self.fcs[-1]
        nn.init.constant_(fc.weight[0], 3)
        nn.init.constant_(fc.weight[1], -8)
        
        nn.init.constant_(fc.bias, 0)

    def compute_comp_scores(self, match_ll, det_bboxes, prev_bboxes, label_delta, add_bbox_dummy=False):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if self.match_coeff is None:
            return match_ll
        else:
            ious = []
            matches= []
            bboxes_ious=[]
            mean_scores=[]
            shifts=[]
            bbox_scores =  det_bboxes[:, 4].view(-1)
            bboxes=det_bboxes[:, :4]
            _,score_order=torch.sort(bbox_scores,descending=True)
            iii = 0
            while len(mean_scores)==0 or iii<len(score_order):
                base_match = score_order[iii]
                pmatch = match_ll[0 ,base_match, :]
                mval, midx = torch.sort(pmatch,descending=True)
                #midx=midx[mval>0.7]
                for idx in midx:
                    # match coeff needs to be length of 3
                    assert(len(self.match_coeff) == 3)
                    ref_bbox = prev_bboxes[idx-1,:4]
                    ref_center = torch.tensor([(ref_bbox[0]+ref_bbox[2])/2.0,(ref_bbox[1]+ref_bbox[3])/2.0],
                                             device=torch.cuda.current_device())
                    base_bbox = bboxes[base_match,:]
                    base_center = torch.tensor([(base_bbox[0]+base_bbox[2])/2.0,(base_bbox[1]+base_bbox[3])/2.0],device=torch.cuda.current_device())
                    shift = ref_center - base_center
                    if self.prev_shift is not None:
                        if torch.dot(self.prev_shift,shift)<=0:
                            continue
                    bboxes_shifted= bboxes + torch.cat([shift,shift])
                    bbox_ious = bbox_overlaps(bboxes_shifted, prev_bboxes[:, :4])
                    if add_bbox_dummy:
                        bbox_iou_dummy = torch.ones(bbox_ious.size(0), 1,
                                                     device=torch.cuda.current_device()) * self.bbox_dummy_iou
                        bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
                        label_dummy = torch.ones(bbox_ious.size(0), 1,
                                                 device=torch.cuda.current_device())
                        label_delta = torch.cat((label_dummy, label_delta), dim=1)
                    best_ious, best_matches = torch.max(bbox_ious,dim=1)
                    best_matches[best_ious<0.4]=0
                    multi_ids = [a for i,a in enumerate(best_matches) if a in best_matches[:i]]
                    multi_ids = torch.tensor(multi_ids,device=torch.cuda.current_device()).unique()
                    multi_ids = multi_ids[multi_ids >0]

                    while len(multi_ids) > 0:
                        for multi_id in multi_ids:
                            bidx = (best_matches==multi_id).nonzero()
                            comp_score = self.match_coeff[0]*torch.log(bbox_scores[bidx].view(-1))+ self.match_coeff[1] * bbox_ious[bidx,multi_id].view(-1)
                            bidx = bidx[comp_score < max(comp_score)]
                            bbox_ious[bidx,multi_id]=0
                            best_ious, best_matches = torch.max(bbox_ious,dim=1)
                            best_matches[best_ious<0.05]=0
                            multi_ids = [a for i,a in enumerate(best_matches) if a in best_matches[:i]]
                            multi_ids = torch.tensor(multi_ids,device=torch.cuda.current_device()).unique()
                            multi_ids = multi_ids[multi_ids >0]

                    bboxes_ious.append(bbox_ious)
                    ious.append(best_ious)
                    matches.append(best_matches)
                    mean_scores.append(torch.sum(best_ious)
                                       /torch.tensor(len(best_matches.nonzero())).float())
                    shifts.append(shift)
                iii += 1 
            
            shift = shifts[mean_scores==max(mean_scores)]
            print(shift)
            self.prev_shift=shift
            bbox_ious=bboxes_ious[mean_scores==max(mean_scores)]
            best_ious = ious[mean_scores==max(mean_scores)]
            best_matches = matches[mean_scores==max(mean_scores)] 
            return best_matches

    
    def forward(self, bboxes, ref_bboxes, x, ref_x, x_n, ref_x_n, delta_frame=1):
        # x and ref_x are the grouped bbox features of current and reference frame
        # x_n are the numbers of proposals in the current images in the mini-batch, 
        # ref_x_n are the numbers of ground truth bboxes in the reference images.
        # here we compute a correlation matrix of x and ref_x
        # we also add a all 0 column denote no matching
        assert len(x_n) == len(ref_x_n)
        if self.with_avg_pool:
            x = self.avg_pool(x)
            ref_x = self.avg_pool(ref_x)
        x = x.view(x.size(0), -1)
        ref_x = ref_x.view(ref_x.size(0), -1)
        
        for idx, fc in enumerate(self.fcs[:-1]):
            x = fc(x)
            ref_x = fc(ref_x)
            if idx < len(self.fcs) - 2:
                x = self.relu(x)
                ref_x = self.relu(ref_x)
        #bboxes=torch.stack([bboxes],dim=0)
        #ref_bboxes=torch.stack([ref_bboxes],dim=0)
        #fc=self.fcs[-1]
        #shift=fc(torch.ones(1,1,device=torch.cuda.current_device())*delta_frame)
        #bbox_shifted=bboxes+torch.cat([shift,shift], dim = 1)
        n = len(x_n)
        x_split = torch.split(x, x_n, dim=0)
        ref_x_split = torch.split(ref_x, ref_x_n, dim=0)
        prods = []
        #bboxes_split = torch.split(bboxes, x_n, dim=0)
        #ref_bboxes_split = torch.split(ref_bboxes, x_n, dim=0)
        if len(bboxes.shape)<3:
            bboxes=torch.stack([bboxes],dim=0)
            ref_bboxes=torch.stack([ref_bboxes],dim=0)
        IoUs2 = torch.zeros(bboxes[0].size(0), ref_bboxes[0].size(0)+1, device=torch.cuda.current_device())
        distances_xy = torch.zeros(2, bboxes[0].size(0), ref_bboxes[0].size(0)+1, device=torch.cuda.current_device())
        distances=[]
        for i in range(n):
            # shift
           
            centers = torch.stack([(bboxes[i][:,0]+bboxes[i][:,2])/2.0,(bboxes[i][:,1]+bboxes[i][:,3])/2.0], dim=1)
            ref_centers=torch.stack([(ref_bboxes[i][:,0]+ref_bboxes[i][:,2])/2.0, (ref_bboxes[i][:,1]+ref_bboxes[i][:,3])/2.0], dim=1)
            distances_split=torch.zeros(len(centers), 2, device=torch.cuda.current_device())
            for idx, ref_center in enumerate(ref_centers):
                distance = (ref_center-centers)
                distances_xy[0, :, idx+1] = distance[:,0]
                distances_xy[1, :, idx+1] = distance[:,1]
                distances_split=torch.cat([distances_split,distance],dim=1)
                distance = torch.cat([distance,distance],dim=1)
                bbox_shift1=bboxes[i]+distance
                IoUs2[:,idx+1] = bbox_overlaps(bbox_shift1 , ref_bboxes[i])[:,idx]
            distances_split=torch.stack([distances_split],dim=0)  
            distances.append(distances_split)
            # feature map similarity
            prod = torch.mm(x_split[i], torch.transpose(ref_x_split[i], 0, 1))
            prods.append(prod)
        if self.dynamic:
            match_score = []
            for prod in prods:
                m = prod.size(0)
                dummy = torch.zeros( m, 1, device=torch.cuda.current_device())
                prod_ext = torch.cat([dummy, prod], dim=1)
                match_score.append(prod_ext)
        else:
            dummy = torch.zeros(n, m, device=torch.cuda.current_device())
            prods_all = torch.cat(prods, dim=0)
            match_score = torch.cat([dummy,prods_all], dim=2)
        IoUs = torch.zeros(bboxes[0].size(0), ref_bboxes[0].size(0)+1, device=torch.cuda.current_device())
        #for j,ref_bbox in enumerate(ref_bboxes):
        #    IoUs[:,1:] = bbox_overlaps(bbox_shifted[0,:,:],ref_bbox)
        shift=0
        return match_score, shift, IoUs2, distances_xy


    def loss(self,
             match_score,
             shift,
             IoUs2,
             distances_xy,
             ids,
             id_weights,
             reduce=True):
        losses = dict()
        if self.dynamic:
            n = len(match_score)
            x_n = [s.size(0) for s in match_score]
            ids = torch.split(ids, x_n, dim=0)
            loss_match = 0.
            match_acc = 0.
            n_total = 0
            batch_size = len(ids)
            for score, cur_ids, cur_weights in zip(match_score, ids, id_weights):
                valid_idx = torch.nonzero(cur_weights).squeeze()
                valid_cls = torch.index_select(cur_ids,0, valid_idx)
                nonzero_ids=valid_cls.clone()
                nonzero_ids [ nonzero_ids > 0 ] = 1    
                #loss_iou = torch.sum((1-IoUs[valid_idx , valid_cls])*nonzero_ids)
                #loss_iou = loss_iou/(torch.sum(nonzero_ids)+1e-7)
                #print(valid_idx)
                valid_iou = IoUs2[valid_idx , valid_cls]
                distance_mean = torch.zeros(2, device=torch.cuda.current_device())
                for match_cls in valid_cls.unique():
                    if match_cls == 0:
                        continue
                    valid_iou[(valid_cls==match_cls) & (valid_iou != valid_iou[valid_cls==match_cls][0])]=0
                valid_iou[valid_iou<0.7]=0
                valid_xy = distances_xy[:, valid_idx , valid_cls]
                #dif_x =  valid_xy[0]-shift[0][0]
                #dif_y =  valid_xy[1]-shift[0][1]
                #distance_mean = torch.sum(torch.sqrt(dif_x*dif_x + dif_y*dif_y) * valid_iou / (torch.sum(valid_iou)+1e-7))
                if len(valid_idx.size()) == 0: continue
                n_valid = valid_idx.size(0)
                n_total += n_valid
                loss_match += weighted_cross_entropy(
                    score, cur_ids, cur_weights, reduce=reduce)
                match_acc += accuracy(torch.index_select(score, 0, valid_idx), 
                                      valid_cls) * n_valid
            losses['loss_match'] = loss_match / n
            if n_total > 0:
                losses['match_acc'] = match_acc / n_total
        else:
            print('not dynamic')
            if match_score is not None:
                valid_idx = torch.nonzero(cur_weights).squeeze()
                losses['loss_match'] = weighted_cross_entropy(
                    match_score, ids, id_weights, reduce=reduce)
                losses['match_acc'] = accuracy(torch.index_select(match_score, 0, valid_idx), 
                                               torch.index_select(ids, 0, valid_idx))
        return losses

