import numpy as np
import torch
import torch.nn as nn

floattype = torch.cuda.DoubleTensor

def bilinear_interpolate(img, x, y):
    """Return bilinear interpolation of 4 nearest pts w.r.t to x,y from img
    Taken from https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e

    Args:
        img (torch.Tensor): Tensor of size wxh. Usually one channel of feature layer
        x (torch.Tensor): Float dtype, x axis location for sampling
        y (torch.Tensor): Float dtype, y axis location for sampling

    Returns:
        torch.Tensor: interpolated value
    """
    
    x = x.type(floattype)
    y = y.type(floattype)
    x0 = torch.floor(x).type(torch.cuda.LongTensor)
    x1 = x0 + 1

    y0 = torch.floor(y).type(torch.cuda.LongTensor)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, img.shape[1]-1)
    x1 = torch.clamp(x1, 0, img.shape[1]-1)
    y0 = torch.clamp(y0, 0, img.shape[0]-1)
    y1 = torch.clamp(y1, 0, img.shape[0]-1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    norm_const = 1/((x1.type(floattype) - x0.type(floattype))*(y1.type(floattype) - y0.type(floattype)))
    # print(norm_const.type(), x.type(), y.type(), x1.type(), y1.type())
    wa = (x1.type(floattype) - x) * (y1.type(floattype) - y) * norm_const
    wb = (x1.type(floattype) - x) * (y-y0.type(floattype)) * norm_const
    wc = (x-x0.type(floattype)) * (y1.type(floattype) - y) * norm_const
    wd = (x-x0.type(floattype)) * (y - y0.type(floattype)) * norm_const

    # print("Ia, wa, Ib, wb, Ic, wc, Id, wd", Ia, wa, Ib, wb, Ic, wc, Id, wd, Ia.type(), wa.type(), Ib.type(), wb.type(), Ic.type(), wc.type(), Id.type(), wd.type())
    # print(Ia*wa + Ib*wb + Ic*wc + Id*wd)
    return Ia*wa + Ib*wb + Ic*wc + Id*wd
    # return torch.t(torch.t(Ia)*wa) + torch.t(torch.t(Ib)*wb) + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)

class TorchROIAlign(object):

    def __init__(self, output_size, scaling_factor):
        print("output_size, scaling_factor", output_size, scaling_factor)
        self.output_size = output_size
        self.scaling_factor = scaling_factor

    def _roi_align(self, features, scaled_proposal):
        """Given feature layers and scaled proposals return bilinear interpolated
        points in feature layer

        Args:
            features (torch.Tensor): Tensor of shape channels x width x height
            scaled_proposal (list of torch.Tensor): Each tensor is a bbox by which we
            will extract features from features Tensor
        """

        _, num_channels, h, w = features.shape

        xp0, yp0, xp1, yp1 = scaled_proposal
        p_width = xp1 - xp0
        p_height = yp1 - yp0

        w_stride = p_width/self.output_size
        h_stride = p_height/self.output_size

        interp_features = torch.zeros((num_channels, self.output_size, self.output_size))

        for i in range(self.output_size):
            for j in range(self.output_size):
                x_bin_strt = i*w_stride + xp0
                y_bin_strt = j*h_stride + yp0

                # generate 4 points for interpolation
                # notice no rounding
                x1 = torch.Tensor([x_bin_strt + 0.25*w_stride])
                x2 = torch.Tensor([x_bin_strt + 0.75*w_stride])
                y1 = torch.Tensor([y_bin_strt + 0.25*h_stride])
                y2 = torch.Tensor([y_bin_strt + 0.75*h_stride])

                for c in range(num_channels):
                    img = features[0, c]
                    v1 = bilinear_interpolate(img, x1, y1)
                    v2 = bilinear_interpolate(img, x1, y2)
                    v3 = bilinear_interpolate(img, x2, y1)
                    v4 = bilinear_interpolate(img, x2, y2)

                    interp_features[c, j, i] = (v1+v2+v3+v4)/4

        return interp_features

    def __call__(self, feature_layer, proposals):
        """Given feature layers and a list of proposals, it returns aligned
        representations of the proposals. Proposals are scaled by scaling factor
        before pooling.

        Args:
            feature_layer (torch.Tensor): Feature layer of size (num_channels, width,
            height)
            proposals (list of torch.Tensor): Each element of the list represents a
            bounding box as (w,y,w,h)

        Returns:
            torch.Tensor: Shape len(proposals), channels, self.output_size,
            self.output_size
        """

        _, num_channels, _, _ = feature_layer.shape

        # first scale proposals down by self.scaling factor
        scaled_proposals = torch.zeros_like(proposals)

        # notice no ceil or floor functions
        scaled_proposals[:, 0] = proposals[:, 0] * self.scaling_factor
        scaled_proposals[:, 1] = proposals[:, 1] * self.scaling_factor
        scaled_proposals[:, 2] = proposals[:, 2] * self.scaling_factor
        scaled_proposals[:, 3] = proposals[:, 3] * self.scaling_factor

        res = torch.zeros((len(proposals), num_channels, self.output_size,
                        self.output_size))
        for idx in range(len(scaled_proposals)):
            proposal = scaled_proposals[idx]
            res[idx] = self._roi_align(feature_layer, proposal)

        return res

class TorchROIPool(object):

    def __init__(self, output_size, scaling_factor):
        """ROI max pooling works by dividing the hxw RoI window into an HxW grid of
           approximately size h/H x w/W and then max-pooling the values in each
           sub-window. Pooling is applied independently to each feature map channel.
        """
        self.output_size = output_size
        self.scaling_factor = scaling_factor

    def _roi_pool(self, features):
        """Given scaled and extracted features, do channel wise pooling
        to return features of fixed size self.output_size, self.output_size

        Args:
            features (np.Array): scaled and extracted features of shape
            num_channels, proposal_width, proposal_height
        """

        num_channels, h, w = features.shape

        w_stride = w/self.output_size
        h_stride = h/self.output_size

        res = torch.zeros((num_channels, self.output_size, self.output_size))
        res_idx = torch.zeros((num_channels, self.output_size, self.output_size))
        for i in range(self.output_size):
            for j in range(self.output_size):

                # important to round the start and end, and then conver to int
                w_start = int(np.floor(j*w_stride))
                w_end = int(np.ceil((j+1)*w_stride))
                h_start = int(np.floor(i*h_stride))
                h_end = int(np.ceil((i+1)*h_stride))

                # limiting start and end based on feature limits
                w_start = min(max(w_start, 0), w)
                w_end = min(max(w_end, 0), w)
                h_start = min(max(h_start, 0), h)
                h_end = min(max(h_end, 0), h)

                patch = features[:, h_start: h_end, w_start: w_end]
                max_val, max_idx = torch.max(patch.reshape(num_channels, -1), dim=1)
                res[:, i, j] = max_val
                res_idx[:, i, j] = max_idx

        return res, res_idx

    def __call__(self, feature_layer, proposals):
        """Given feature layers and a list of proposals, it returns pooled
        respresentations of the proposals. Proposals are scaled by scaling factor
        before pooling.

        Args:
            feature_layer (np.Array): Feature layer of size (num_channels, width,
            height)
            proposals (list of np.Array): Each element of the list represents a bounding
            box as (w,y,w,h)

        Returns:
            np.Array: Shape len(proposals), channels, self.output_size, self.output_size
        """

        batch_size, num_channels, _, _ = feature_layer.shape

        # first scale proposals based on self.scaling factor
        scaled_proposals = torch.zeros_like(proposals)

        # the rounding by torch.ceil is important for ROI pool
        scaled_proposals[:, 0] = torch.ceil(proposals[:, 0] * self.scaling_factor)
        scaled_proposals[:, 1] = torch.ceil(proposals[:, 1] * self.scaling_factor)
        scaled_proposals[:, 2] = torch.ceil(proposals[:, 2] * self.scaling_factor)
        scaled_proposals[:, 3] = torch.ceil(proposals[:, 3] * self.scaling_factor)

        res = torch.zeros((len(proposals), num_channels, self.output_size,
                        self.output_size))
        res_idx = torch.zeros((len(proposals), num_channels, self.output_size,
                        self.output_size))
        for idx in range(len(proposals)):
            proposal = scaled_proposals[idx]
            # adding 1 to include the end indices from proposal
            extracted_feat = feature_layer[0, :, proposal[1].to(dtype=torch.int8):proposal[3].to(dtype=torch.int8)+1, proposal[0].to(dtype=torch.int8):proposal[2].to(dtype=torch.int8)+1]
            res[idx], res_idx[idx] = self._roi_pool(extracted_feat)

        return res

if __name__ == "__main__":
    device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    # create feature layer, proposals and targets
    num_proposals = 10
    feat_layer = torch.randn(1, 64, 32, 32)
    print(np.shape(feat_layer))
    proposals = torch.zeros((num_proposals, 4))
    proposals[:, 0] = torch.randint(0, 16, (num_proposals,))
    proposals[:, 1] = torch.randint(0, 16, (num_proposals,))
    proposals[:, 2] = torch.randint(16, 32, (num_proposals,))
    proposals[:, 3] = torch.randint(16, 32, (num_proposals,))
    print(proposals, np.shape(proposals))

    my_roi_pool_obj = TorchROIPool(7, 2**-1)
    roi_pool1 = my_roi_pool_obj(feat_layer, proposals)
    print(np.shape(roi_pool1))
    # my_roi_align_obj = TorchROIAlign(7, 2**-1)
    # roi_align1 = my_roi_align_obj(feat_layer, proposals)
    # roi_align1 = roi_align1.cpu().numpy()
    # print(np.shape(roi_align1))
