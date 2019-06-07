import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from dl4cv.classifiers.segmentation_nn import SegmentationNN
from dl4cv.data_utils import SegmentationData, label_img_to_rgb
from dl4cv.solver import Solver
import torch.nn.functional as F

#torch.set_default_tensor_type('torch.FloatTensor')

#%matplotlib inline

#Load data
train_data = SegmentationData(image_paths_file='datasets/segmentation_data/train.txt')
val_data = SegmentationData(image_paths_file='datasets/segmentation_data/val.txt')




########################################################################
#                             YOUR CODE                                #
########################################################################
#model = SegmentationNN()
model=torch.load("models/segmentation_nn5.model")
solver = Solver()

train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=3,
                                          shuffle=False,
                                          num_workers=4)
val_loader = torch.utils.data.DataLoader(val_data,
                                          batch_size=3,
                                          shuffle=False,
                                          num_workers=1)
solver.train(model, train_loader, val_loader, log_nth=4, num_epochs=3)


#### TEST
test_data = SegmentationData(image_paths_file='datasets/segmentation_data_test/test.txt')
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=1)

test_scores = []
model.eval()
for inputs, targets in test_loader:
    inputs, targets = Variable(inputs), Variable(targets)
    if model.is_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    
    outputs = model.forward(inputs)
    _, preds = torch.max(outputs, 1)
    targets_mask = targets >= 0
    test_scores.append(np.mean((preds == targets)[targets_mask].data.cpu().numpy()))
    
model.train()
val=np.mean(test_scores)
print("Final score n test: %d", val)
model.save("models/segmentation_nn6.model")