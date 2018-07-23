
import torch

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels.long()]     # [N,D]








def test_one_hot_embedding():
  labels = torch.Tensor([2,0,1,0,2])
  num_classes = 3
  labels_oh_ep = torch.Tensor([[0,0,1], [1,0,0], [0,1,0], [1,0,0], [0,0,1] ])
  labels_oh = one_hot_embedding(labels, num_classes)
  print(labels_oh_ep)
  print(labels_oh)
  assert( (labels_oh_ep - labels_oh).sum() == 0  )

# test_one_hot_embedding()