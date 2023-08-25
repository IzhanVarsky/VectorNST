import numpy as np
import torch
from torchvision import models


class LPIPS(torch.nn.Module):
    def __init__(self, device, pretrained=True, normalize=False, pre_relu=True):
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(LPIPS, self).__init__()
        # VGG using perceptually-learned weights (LPIPS metric)
        self.normalize = normalize
        self.pretrained = pretrained

        self.feature_extractor = LPIPS._FeatureExtractor(pretrained, pre_relu, device)

    def _l2_normalize_features(self, x, eps=1e-10):
        nrm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        return x / (nrm + eps)

    def sample_xform(self):
        color_scale = torch.rand(size=(3,)) * 2
        channel_perm = np.random.permutation(3)
        return dict(color_scale=color_scale,
                    channel_perm=channel_perm)

    def xform(self, im, params):
        color_scale = params["color_scale"].view(1, 3, 1, 1).to(im.device)
        im = im * color_scale

        return im

    def forward(self, pred, target):
        """Compare VGG features of two inputs."""

        p = self.sample_xform()
        pred = self.xform(pred, p)
        target = self.xform(target, p)

        # Get VGG features
        pred = self.feature_extractor(pred)
        target = self.feature_extractor(target)

        # L2 normalize features
        # if self.normalize:
        pred = [self._l2_normalize_features(f) for f in pred]
        target = [self._l2_normalize_features(f) for f in target]

        # TODO(mgharbi) Apply Richard's linear weights?

        if self.normalize:
            diffs = [torch.sum((p - t) ** 2, 1) for (p, t) in zip(pred, target)]
        else:
            # mean instead of sum to avoid super high range
            diffs = [torch.mean((p - t) ** 2, 1) for (p, t) in zip(pred, target)]

        # Spatial average
        diffs = [diff.mean([1, 2]) for diff in diffs]

        return sum(diffs).mean()

    class _FeatureExtractor(torch.nn.Module):
        def __init__(self, pretrained, pre_relu, device):
            super(LPIPS._FeatureExtractor, self).__init__()
            vgg_pretrained = models.vgg19(pretrained=pretrained).features.eval()
            vgg_pretrained.to(device)

            self.breakpoints = [0, 4, 9, 16, 23, 30, 36]
            self.weights = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]

            if pre_relu:
                for i, _ in enumerate(self.breakpoints[1:]):
                    self.breakpoints[i + 1] -= 1

            for i, b in enumerate(self.breakpoints[:-1]):
                ops = torch.nn.Sequential()
                for idx in range(b, self.breakpoints[i + 1]):
                    op = vgg_pretrained[idx].requires_grad_(False)
                    ops.add_module(str(idx), op)

                self.add_module("group{}".format(i), ops)

            self.register_buffer("shift", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("scale", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            feats = []
            x = (x - self.shift) / self.scale
            for idx in range(len(self.breakpoints) - 1):
                m = getattr(self, "group{}".format(idx))
                x = m(x) * self.weights[idx]
                feats.append(x)
            return feats


class ELPIPS(torch.nn.Module):
    def __init__(self, device, pretrained=True, normalize=False, pre_relu=True, max_shift=16,
                 nsamples=3):
        """Ensemble of LPIPS."""
        super(ELPIPS, self).__init__()
        self.max_shift = max_shift
        self.ploss = LPIPS(device, pretrained=pretrained, normalize=normalize, pre_relu=pre_relu)
        self.nsamples = nsamples

    def forward(self, a, b):
        losses = []
        for smp in range(self.nsamples):
            losses.append(self.ploss(a, b))
        losses = torch.stack(losses)
        return losses.mean()
