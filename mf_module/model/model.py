import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np


def create_coordinate_map(B, H, W, device='cpu', dtype=torch.float32):
    """ Create coordinate grid for a single image (H, W) """
    y_coords = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)  # (H, W)
    x_coords = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)  # (H, W)
    coord_map = torch.stack((x_coords, y_coords), dim=0)                            # (2, H, W)
    coord_map = coord_map.unsqueeze(0).expand(B, -1, -1, -1)                        # (B, 2, H, W)
    return coord_map


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, norm="bn", **kwargs):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if norm == "bn":
            norm_1 = nn.BatchNorm2d(mid_channels)
            norm_2 = nn.BatchNorm2d(mid_channels) 

        elif norm == "gn":
            norm_1 = nn.GroupNorm(kwargs.get("gn_group", 8), mid_channels)
            norm_2 = nn.GroupNorm(kwargs.get("gn_group", 8), mid_channels)

        elif norm == 'none':
            norm_1 = nn.Identity()
            norm_2 = nn.Identity()

        disable_dropout = kwargs.get("disable_dropout", False)

        if kwargs.get("dropout", False):
            if disable_dropout:
                dropout_val = 0.0
            else:
                dropout_val = kwargs.get("dropout_val", 0.1)

            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                norm_1,
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_val),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                norm_2,
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_val)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                norm_1,
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                norm_2,
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm="bn", **kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm=norm, **kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm="bn", **kwargs):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm, disable_dropout=True, **kwargs)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm, disable_dropout=True, **kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DualheadUNet(nn.Module):
    def __init__(
        self, 
        n_channels, 
        n_classes_a, 
        n_classes_b, 
        bilinear=False, 
        n_base_channel=64, 
        norm="gn", 
        **kwargs
    ):
        """ Dual-head UNet for motion estimation """
        super(DualheadUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes_a = n_classes_a
        self.n_classes_b = n_classes_b
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, n_base_channel, norm=norm, **kwargs))
        self.down1 = (Down(n_base_channel, n_base_channel*2, norm=norm, **kwargs))
        self.down2 = (Down(n_base_channel*2, n_base_channel*4, norm=norm, **kwargs))
        self.down3 = (Down(n_base_channel*4, n_base_channel*8, norm=norm, **kwargs))
        factor = 2 if bilinear else 1

        self.down4 = (Down(n_base_channel*8, n_base_channel*16 // factor, norm=norm, **kwargs))
        self.up1 = (Up(n_base_channel*16, n_base_channel*8 // factor, bilinear, norm=norm, **kwargs))
        self.up2 = (Up(n_base_channel*8, n_base_channel*4 // factor, bilinear, norm=norm, **kwargs))
        self.up3a = (Up(n_base_channel*4, n_base_channel*2 // factor, bilinear, norm=norm, **kwargs))
        self.up4a = (Up(n_base_channel*2, n_base_channel, bilinear, norm=norm))
        self.up3b = (Up(n_base_channel*4, n_base_channel*2 // factor, bilinear, norm=norm, **kwargs))
        self.up4b = (Up(n_base_channel*2, n_base_channel, bilinear, norm=norm))
        self.outca = (OutConv(n_base_channel, n_classes_a))
        self.outcb = (OutConv(n_base_channel, n_classes_b))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        xa = self.up3a(x, x2)
        xa = self.up4a(xa, x1)
        xb = self.up3b(x, x2)
        xb = self.up4b(xb, x1)

        outa = self.outca(xa)
        outb = self.outcb(xb)

        return torch.cat([outa, outb], dim=1)


class MotionDetectorModel(nn.Module):
    def __init__(
        self, 
        n_base_channel=64, 
        dropout=True, 
        dropout_val=0.0,
        n_channels=10, 
        f_mean=300.0,
        f_scale=1000.0,
        **kwargs
    ):
        """ Wrapper """
        super().__init__()
        self.net = DualheadUNet(
            n_channels=n_channels, 
            n_classes_a=1, 
            n_classes_b=3, 
            n_base_channel=n_base_channel, 
            dropout=dropout, 
            dropout_val=dropout_val,
            **kwargs
        )

        self.f_mean = f_mean
        self.f_scale = f_scale
        return 

    def forward(self, x, camera_intrinsics, camera_transform=None):
        '''
            camera_intrinsics: [B, 4] torch.Tensor
            camera_transform:  [B, 9] (not used by this release)
        '''
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        fxy = camera_intrinsics[:, :2]
        cxy = camera_intrinsics[:, 2:]

        f_map = torch.ones(B, 2, H, W).to(x.device)
        inv_fxy = ((1 / fxy - 1 / self.f_mean) * self.f_scale)[:, :, None, None]
        f_map = f_map * inv_fxy 

        fc_map = create_coordinate_map(B, H, W, device=x.device)
        fc_map = (fc_map - cxy[:, :, None, None]) / fxy[:, :, None, None]

        x = torch.cat((x, f_map, fc_map), dim=1)
        prediction = self.net(x)
        
        depth = prediction[:, 0:1, ...]
        flow = prediction[:, 1:4, ...]
        return {"depth": depth, "flow": flow, "all": prediction}

    def get_motion_field(self, x, camera_intrinsics, mask, camera_transform=None):
        return {"motion": self.forward(x, camera_intrinsics, camera_transform=None)['all'].detach().cpu().numpy()}

    def init_result_data(self, args):
        shape = args["image_shape"]
        return {
            "motion": np.zeros([shape[0], shape[1], 4]), 
            "visual": np.zeros([shape[0], shape[1], 4])
        }

    def add_data_to_result(self, obj_id, out, result, input_data, data_parser):
        result["motion"] = result["motion"] + np.moveaxis(out["motion"][0], 0, -1)
        result["visual"] = result["visual"] + np.moveaxis(out["motion"][0], 0, -1)
        return result

    def convert_result_to_save_format(self, result):
        '''
            The result is the value of key "motion above.
        '''
        return {"motion": result}

