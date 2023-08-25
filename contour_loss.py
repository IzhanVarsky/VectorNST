from torch import nn
import torch.nn.functional as F
from torchvision import transforms


class ContourLoss(nn.Module):
    def __init__(self, target_contour):
        super(ContourLoss, self).__init__()
        self.target_contour = target_contour.detach()
        dim1 = int(self.target_contour.size()[2] * 0.25)
        dim2 = int(self.target_contour.size()[3] * 0.25)
        self.transform = transforms.RandomCrop(size=(dim1, dim2))

    def forward(self, input_contour):
        img1 = self.transform(self.target_contour)
        img2 = self.transform(input_contour)
        self.loss = F.l1_loss(img1, img2)
        # self.loss = F.mse_loss(img1,img2)
        # self.loss = torch.mean((img1**0.5-img2**0.5)**2)
        return self.loss
