import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def batch_iou(bbox1, bbox2):  
    # pairwise jaccard botween boxes a and boxes b
    # box: [left, top, right, bottom]
    lt = torch.max(bbox1[:, :2].unsqueeze(1), bbox2[:, :2])
    rb = torch.min(bbox1[:, 2:].unsqueeze(1), bbox2[:, 2:])
    
    inter = torch.clamp(rb - lt, 0, None)

    area_i = torch.prod(inter, dim=2)
    area_a = torch.prod(bbox1[:, 2:] - bbox1[:, :2], dim=1)
    area_b = torch.prod(bbox2[:, 2:] - bbox2[:, :2], dim=1)

    area_u = area_a.unsqueeze(1) + area_b - area_i
    return area_i.float() / torch.clamp(area_u.float(), 1e-7, None)  # shape: (len(a) x len(b))


def cnms(boxes, scores, max_output_size, iou_threshold=0.5, score_threshold=1e+10 ):
        
    num = scores.shape[0]    
    Keep = torch.zeros( num ).byte()  
    max_output_size = min(max_output_size, num)    
    
    if num == 0:
        return Keep

    idx = scores.topk( max_output_size )[1]       
    if idx.shape[0] == 0:
        return Keep
    
    boxes = boxes[idx]
    scores = scores[idx]
    iou = batch_iou(boxes, boxes)
   
    keep = torch.zeros( scores.shape[0] ).byte()
    keep[ scores.argmax() ] = 1
    for i in scores.sort(descending=True)[1]:
        if (iou[i, keep==1 ] < iou_threshold).all():
            keep[i] = 1
                                 
    Keep[idx] = keep
    return Keep



def filter_detections(
    boxes,
    classification,
    class_specific_filter = True,
    bnms                  = True,
    score_threshold       = 0.05,
    max_detections        = 300,
    nms_threshold         = 0.5
):
    """ Filter detections using the boxes and classification values.

    Args
        boxes                 : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification        : Tensor of shape (num_boxes, num_classes) containing the classification scores.
        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
        bnms                  : Flag to enable/disable non maximum suppression.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    def _filter_detections(scores, labels):
        
        
        # threshold based on score
        indices = scores > score_threshold 
        indices = torch.nonzero(indices).view(-1)       

        if bnms:
            filtered_boxes  =  boxes[indices,...]
            filtered_scores =  scores[indices]

            # perform NMS
            #nms_indices = nms( torch.cat( [filtered_boxes, filtered_scores], dim=1 ), 0.5 )            
            nms_indices = cnms( filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold, score_threshold = score_threshold)
            nms_indices = torch.nonzero(nms_indices).view(-1)

            # filter indices based on NMS
            indices = indices[nms_indices]

        # add indices to list of all indices
        labels = labels[indices]  

        if indices.shape[0] != 0:
            indices = torch.stack([indices, labels ], dim=1)
        return indices

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * torch.ones( scores.shape[0] ).long()
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        indices = torch.cat(all_indices, dim=0)

    else:
        scores, labels  = torch.max(classification, dim = 1)
        indices = _filter_detections(scores, labels)


    if indices.shape[0] != 0:        

        # select top k
        #scores              = torch.index_select( classification, 0,  indices.long() )
        scores              = classification.repeat( classification.shape[1], 1 )[indices[:,0], indices[:,1] ]
        labels              = indices[:, 1]
        scores, top_indices = torch.topk(scores, k=min( max_detections, scores.shape[0] ) ) 
    
        # filter input using the final set of indices
        indices             = indices[:, 0][top_indices]
        boxes               = boxes[indices, ...]
        labels              = labels[top_indices]

        
    else:
        
        box_size = min(scores.shape[0], max_detections)
        boxes  = boxes[:box_size, ...]  * 0 - 1
        scores = scores[:box_size, ...] * 0 - 1
        labels = scores[:box_size, ...] * 0 - 1
        labels = labels.int()


    # zero pad the outputs
    pad_size = max(0, max_detections - scores.shape[0])

    #torch.nn.functional.pad(input, pad, mode='constant', value=0)
    boxes    = F.pad(boxes,  ( 0, 0, 0, pad_size ), mode='constant', value=-1)
    scores   = F.pad(scores, ( 0, pad_size), mode='constant', value=-1)
    labels   = F.pad(labels, ( 0, pad_size), mode='constant', value=-1)
    labels   = labels.int()  

    # set shapes, since we know what they are
    boxes.view(max_detections, 4 )
    scores.view(max_detections,1)
    labels.view(max_detections,1)


    return boxes, scores, labels



class FilterDetections(nn.Module):
    def __init__(
        self,
        nms                   = True,
        class_specific_filter = True,
        nms_threshold         = 0.5,
        score_threshold       = 0.05,
        max_detections        = 300,
        **kwargs
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            nms                   : Flag to enable/disable NMS.
            class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold       : Threshold used to prefilter the boxes with.
            max_detections        : Maximum number of detections to keep.
        """
        self.nms                   = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold         = nms_threshold
        self.score_threshold       = score_threshold
        self.max_detections        = max_detections
        super(FilterDetections, self).__init__(**kwargs)



    def forward(self, inputs):

        boxes          = inputs[0]
        classification = inputs[1]

        Boxes  = []
        Scores = []
        Labels = []
        for box, cla in zip(boxes, classification ): 
            
            boxes, scores, labels = filter_detections(
                box,
                cla,
                class_specific_filter = self.class_specific_filter,
                bnms                  = self.nms,
                score_threshold       = self.score_threshold,
                max_detections        = self.max_detections,
                nms_threshold         = self.nms_threshold 
            )
            Boxes.append( boxes )
            Scores.append( scores )
            Labels.append( labels )

        Boxes  = torch.stack( Boxes,  dim=0 )
        Scores = torch.stack( Scores, dim=0 )
        Labels = torch.stack( Labels, dim=0 )
        
        return Boxes, Scores, Labels



