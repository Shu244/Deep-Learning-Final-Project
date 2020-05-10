import numpy as np
import random
import os.path as osp
from .custom import CustomDataset
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from .extra_aug import ExtraAugmentation
from libtiff import TIFF, TIFFfile
import xlrd
from mmcv.parallel import DataContainer as DC
import os
import cv2
import mmcv
from .utils import to_tensor

class CellcountDataset(CustomDataset):
    
    CLASSES = (['RBC'])
    
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=False,
                 with_label=True,
                 with_track=False,
                 extra_aug=None,
                 aug_ref_bbox_param=None,
                 resize_keep_ratio=True,
                 test_mode=False):
        
        # prefix of images path
        self.img_prefix = img_prefix
        self.ann_file=ann_file
        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # load annotations
        self.vid_infos = self.load_annotations(ann_file)
        img_ids = []
        for v, vid_info in enumerate(self.vid_infos):
            for f,ann in enumerate(vid_info['ann_info']):
                img_ids.append([v,f])
        self.img_ids = img_ids
        
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = [i for i, (v, f) in enumerate(self.img_ids)
                if len(self.get_ann_info(v, f)['bboxes'])]
            self.img_ids = [self.img_ids[i] for i in valid_inds]

        # normalization configs
        self.img_norm_cfg = img_norm_cfg
        
        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor
        
        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        self.with_track = with_track
        # params for augmenting bbox in the reference frame
        self.aug_ref_bbox_param = aug_ref_bbox_param
        # in test mode or not
        self.test_mode = test_mode
    
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.numpy2tensor = Numpy2Tensor()
        
        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None
    
        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(self.img_ids[idx])
        data = self.prepare_train_img(self.img_ids[idx])
        return data
    
    def load_annotations(self, ann_file):
        vid_infos = []
        datas=xlrd.open_workbook(ann_file)
        datas=datas.sheets()
        for i ,data in enumerate(datas):
            Image_dir=os.path.join(self.img_prefix,data.col_values(0)[1])
            frames=np.int8(data.col_values(1)[1:])
            cell_ids=data.col_values(3)[1:]
            mask_ids=data.col_values(4)[1:]
            #class_id=data.col_values(2)[1:]
            class_id=1
            ann=[]
            for j in range(1,max(frames)+1):
                nums= [nums for nums,frame in enumerate(frames) if frame==j]
                ann_perframe=[]
                for _, num in enumerate(nums):
                    ann_single={
                        'frame_id': j,
                        'cell_id':int(cell_ids[num]),
                        'mask_id':mask_ids[num],
                        'class_id':class_id
                        }
                    ann_perframe.append(ann_single)
                ann.append(ann_perframe)
            vid_info={
                'filename': Image_dir,
                'vid_id':i,
                'frames': j,
                'ann_info':ann,
                'width':self.img_scales[0][1],
                'height':self.img_scales[0][0]
            }
            vid_infos.append(vid_info)
        return vid_infos
    
    def get_ann_info(self, idx, frame_id):
        ann_info=self.vid_infos[idx]['ann_info']
        return self._parse_ann_info(ann_info, frame_id)
    
    
    def extract_bboxes(self, mask):
        mask=np.stack(mask, axis=-1)
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([x1, y1, x2, y2])
        return boxes.astype(np.int32)
    
    def bbox_aug(self, bbox, img_size):
        assert self.aug_ref_bbox_param is not None
        center_off = self.aug_ref_bbox_param[0]
        size_perturb = self.aug_ref_bbox_param[1]
        
        n_bb = bbox.shape[0]
        # bbox center offset
        center_offs = (2*np.random.rand(n_bb, 2) - 1) * center_off
        # bbox resize ratios
        resize_ratios = (2*np.random.rand(n_bb, 2) - 1) * size_perturb + 1
        # bbox: x1, y1, x2, y2
        centers = (bbox[:,:2]+ bbox[:,2:])/2.
        sizes = bbox[:,2:] - bbox[:,:2]
        new_centers = centers + center_offs * sizes
        new_sizes = sizes * resize_ratios
        new_x1y1 = new_centers - new_sizes/2.
        new_x2y2 = new_centers + new_sizes/2.
        c_min = [0,0]
        c_max = [img_size[1], img_size[0]]
        new_x1y1 = np.clip(new_x1y1, c_min, c_max)
        new_x2y2 = np.clip(new_x2y2, c_min, c_max)
        bbox = np.hstack((new_x1y1,new_x2y2)).astype(np.float32)
        return bbox
    
    def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
        """
        Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
        Returns:
            dict: A dict containing the following keys: bboxes, labels, masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        ann_info=ann_info[frame_id]
        # 1. mask: a binary map of the same size of the image.
        if with_mask:
            gt_masks = []
            for i, ann in enumerate(ann_info):
                # each ann is a list of masks
                # ann:
                # bbox: list of bboxes
                # category_id
                mask_id=ann['mask_id']
                mask_dir=self.ann_file.replace('cellLabel.xlsx','{}.tif'.format(mask_id))
                mask = cv2.imread(mask_dir,0).astype(np.bool)
                gt_masks.append(mask)
                gt_ids.append(ann['cell_id'])
                gt_labels.append(ann['class_id'])
            #gt_masks=np.stack(gt_masks, axis=-1)
            gt_bboxes=self.extract_bboxes(gt_masks)

        if gt_bboxes.any():
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            #gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            #gt_labels = np.array([], dtype=np.int64)

        ann = dict(
            bboxes=gt_bboxes, cell_ids=gt_ids, labels=gt_labels)

        if with_mask:
            ann['masks'] = gt_masks
        return ann
   
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            vid_id, _ = self.img_ids[i]
            vid_info = self.vid_infos[vid_id]
            if vid_info['width'] / vid_info['height'] > 1:
                self.flag[i] = 1
                       
    def Image_gen(self, FileName):
        self.frame=0
        self.ImgeFile=FileName
        tif=TIFF.open(FileName)
        for Image_gened in tif.iter_images():
        #Image = imread(FileName,1)
            self.frame+=1
            yield Image_gened  
    
    def Image_getFrame(self, Image_dir, Frame):
        #if self.ImgeFile!=Image_dir or self.frame > Frame:
        Image_gen=self.Image_gen(Image_dir)
        self.frame=0
        im_getFrame=[]
        while self.frame<Frame:
            im_getFrame= next(Image_gen)
        im_getFrame=np.array(im_getFrame)
        if np.size(im_getFrame) > 0:
            std=np.std(im_getFrame)
            im_getFrame=(im_getFrame-np.mean(im_getFrame))/std
            im_getFrame=im_getFrame[:,:,np.newaxis]
        return im_getFrame
    
    def sample_ref(self, vid,frame_id):
        # sample another frame in the same sequence as reference
        vid_info = self.vid_infos[vid]
        #sample_range = range(vid_info['frames'])
        sample_range =range(frame_id-3,frame_id+3)
        valid_samples = []
        for i in sample_range:
          # check if the frame id is valid
            ref_idx = [vid, i]
            if i != frame_id and ref_idx in self.img_ids:
                valid_samples.append(ref_idx)
        assert len(valid_samples) > 0
        return random.choice(valid_samples)
        
    def prepare_train_img(self, idx,for_check=False):
        # prepare a pair of image in a sequence
        vid,frame_id = idx
        vid_info = self.vid_infos[vid]
        frame_id=frame_id%vid_info['frames']
        # load image
        img = self.Image_getFrame(vid_info['filename'],frame_id+1)
        _, ref_frame_id = self.sample_ref(vid,frame_id)
        ref_img = self.Image_getFrame(vid_info['filename'],ref_frame_id+1)

        ann = self.get_ann_info(vid, frame_id)
        ref_ann = self.get_ann_info(vid, ref_frame_id)
        gt_bboxes = ann['bboxes']
        gt_labels=ann['labels']
        ref_bboxes = ref_ann['bboxes']
        # obj ids attribute does not exist in current annotation
        # need to add it
        ref_ids = ref_ann['cell_ids']
        gt_ids = ann['cell_ids']
        # compute matching of reference frame with current frame
        # 0 denote there is no matching
        gt_pids = [ref_ids.index(i)+1 if i in ref_ids else 0 for i in gt_ids]

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = self.img_scales[0]
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        
        ref_img, ref_img_shape, _, ref_scale_factor = self.img_transform(
            ref_img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        ref_img = ref_img.copy()
        
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,flip)
        ref_bboxes = self.bbox_transform(ref_bboxes, ref_img_shape, ref_scale_factor,flip)
        
        if self.aug_ref_bbox_param is not None:
            ref_bboxes = self.bbox_aug(ref_bboxes, ref_img_shape)
        
        if self.with_mask:
            gt_masks = self.mask_transform(np.float64(ann['masks']), pad_shape, scale_factor, flip)

        ori_shape = (vid_info['height'], vid_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip,
            delta_frame=ref_frame_id-frame_id)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            ref_img=DC(to_tensor(ref_img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)),
            ref_bboxes = DC(to_tensor(ref_bboxes)),
            gt_bboxes_ignore=DC(to_tensor([])),
        )

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_track:
            data['gt_pids'] = DC(to_tensor(gt_pids))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        
        if  for_check:
            return gt_masks
        
        return data
    
    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        frame_id=frame_id%vid_info['frames']
        # load image
        img = self.Image_getFrame(vid_info['filename'],frame_id+1)
        proposal = None

        def prepare_single(img, frame_id, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(vid_info['height'], vid_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                is_first=(frame_id == 0),
                video_id=vid,
                frame_id =frame_id,
                scale_factor=scale_factor,
                flip=flip)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, frame_id, scale, False, proposal)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, proposal)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        return data
