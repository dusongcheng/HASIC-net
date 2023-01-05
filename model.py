import torch
import torch.nn as nn


class CBAMLayer3DSpectral(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAMLayer3DSpectral, self).__init__()
        # spectral Gate
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # spatial Gate
        kernel_size = 7
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2),
                                     nn.Sigmoid())

    def forward(self, x):
        out = self.spectralGate(x)
        # out = self.spatialGate(out)
        return out

    def spectralGate(self, x):
        b, c, d, h, w = x.size()
        y_avg = self.avg_pool(x.view(b, c * d, h, w)).view(b, c * d)
        y_max = self.max_pool(x.view(b, c * d, h, w)).view(b, c * d)
        y = self.fc(y_avg+y_max).view(b, c, d, 1, 1)
        return x * y.expand_as(x)

    def spatialGate(self, x):
        b, c, d, h, w = x.size()
        x_compress = torch.cat((torch.max(x.view(b, c * d, h, w), 1)[0].unsqueeze(1), torch.mean(x.view(b, c * d, h, w), 1).unsqueeze(1)), dim=1)
        y = self.spatial(x_compress).view(b, 1, d, h, w)
        return x * y.expand_as(x)


class ResBlock3DSpatial(nn.Module):
    def __init__(self, inplanes=256, planes=256, shortcut=None, bias=True):
        super(ResBlock3DSpatial, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=bias)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=bias)
        self.prelu2 = nn.PReLU()
        self.shortcut = shortcut
        self.se = CBAMLayer3DSpectral(256, 16)

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.conv1(x)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.se(out)
        out += residual
        out = self.prelu2(out)
        return out


class ResBlock3DSpetral(nn.Module):
    def __init__(self, inplanes=8, planes=8, bias=True):
        super(ResBlock3DSpetral, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=bias)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=bias)
        self.prelu2 = nn.PReLU()
        self.se = CBAMLayer3DSpectral(256, 16)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.se(out)
        out += residual
        out = self.prelu2(out)
        return out


class DoubleFlow(nn.Module):
    def __init__(self, inplanes=3, planes=31, blockSpatial=ResBlock3DSpatial, blockSpectral=ResBlock3DSpetral, layers=[3, 3, 3, 3], inspatialblock=259, outspatialblock=256):
        super(DoubleFlow, self).__init__()
        self.spatial_conv = nn.Conv2d(inplanes, 256, 3, 1, 1)
        self.spatial_res1 = self.make_layer_spatial(blockSpatial, layers[0], inspatialblock, outspatialblock)
        self.spatial_res2 = self.make_layer_spatial(blockSpatial, layers[1], inspatialblock, outspatialblock)
        self.spatial_res3 = self.make_layer_spatial(blockSpatial, layers[2], inspatialblock, outspatialblock)
        self.spatial_res4 = self.make_layer_spatial(blockSpatial, layers[3], inspatialblock, outspatialblock)
        self.spatial_outconv = nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.spatial_outprelu = nn.PReLU()

        self.spectral_conv = nn.Conv2d(inplanes, 256, 3, 1, 1)
        self.spectral_res1 = self.make_layer_spectral(blockSpectral, layers[0])
        self.spectral_res2 = self.make_layer_spectral(blockSpectral, layers[1])
        self.spectral_res3 = self.make_layer_spectral(blockSpectral, layers[2])
        self.spectral_res4 = self.make_layer_spectral(blockSpectral, layers[3])
        self.spectral_outpconv = nn.Conv3d(8, 8, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.spectral_outprelu = nn.PReLU()

        self.output_conv1 = nn.Conv3d(8, 1, 3, 1, 1)
        self.output_conv2 = nn.Conv2d(32, planes, 3, 1, 1)

    def make_layer_spatial(self, block, num_layers, inspatialblock, outspatialblock):
        shortcut = nn.Conv3d(inspatialblock, outspatialblock, 1)
        layers = []
        layers.append(block(inspatialblock, outspatialblock, shortcut))
        for i in range(1, num_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def make_layer_spectral(self, block, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        lateral = self.spatialPath(x)
        spectral = self.spectralPath(x, lateral)
        out = self.output_conv2(self.output_conv1(spectral).squeeze(1))
        return out

    def spatialPath(self, x):
        rgb = self.count_struct_tensor(x)
        b = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]
        lateral = []
        x = self.spatial_conv(x).unsqueeze(2)  # 32,256,1,64,64
        residual = x
        res1 = self.spatial_res1(torch.cat((x, rgb.unsqueeze(2)), 1))  # 32,256,1,64,64
        lateral.append(res1.view(b, 8, 32, h, w))
        res2 = self.spatial_res2(torch.cat((res1, rgb.unsqueeze(2)), 1))  # 32,256,1,64,64
        lateral.append(res2.view(b, 8, 32, h, w))
        res3 = self.spatial_res3(torch.cat((res2, rgb.unsqueeze(2)), 1))  # 32,256,1,64,64
        lateral.append(res3.view(b, 8, 32, h, w))
        res4 = self.spatial_res4(torch.cat((res3, rgb.unsqueeze(2)), 1))  # 32,256,1,64,64
        res4 = self.spatial_outconv(res4)
        res4 += residual
        res4 = self.spatial_outprelu(res4)
        lateral.append(res4.view(b, 8, 32, h, w))
        return lateral

    def spectralPath(self, x, lateral):
        b = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]
        x = self.spectral_conv(x).unsqueeze(1).view(b, 8, 32, h, w)
        residual = x
        x = self.spectral_res1(torch.add(x, lateral[0]))
        x = self.spectral_res2(torch.add(x, lateral[1]))
        x = self.spectral_res3(torch.add(x, lateral[2]))
        x = self.spectral_res4(torch.add(x, lateral[3]))
        x = self.spectral_outpconv(x)
        x += residual
        x = self.spectral_outprelu(x)
        return x

    def count_struct_tensor(self, outputs):
        b, c, h, w, = outputs.shape
        outputs = outputs.view(b * c, h, w).unsqueeze(0).unsqueeze(0)
        gx_kernel = torch.Tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).cuda()
        gy_kernel = torch.Tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).cuda()
        gradx = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                          bias=False).cuda()
        gradx.weight.data = gx_kernel.view(1, 1, 1, 3, 3)
        grady = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                          bias=False).cuda()
        grady.weight.data = gy_kernel.view(1, 1, 1, 3, 3)

        with torch.no_grad():
            imx = gradx(outputs)
            imy = grady(outputs)
        M00, M01, M11 = imx * imx, imx * imy, imy * imy
        outputs_e1 = (M00 + M11) / 2 + torch.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
        # outputs_e1 = torch.exp(outputs_e1)
        outputs_e1 = outputs_e1 / torch.max(outputs_e1)
        return outputs_e1.view(b, c, h, w)


if __name__ == "__main__":
    input_tensor = torch.rand(1, 3, 64, 64).cuda()
    model = DoubleFlow().cuda()
    output = model(input_tensor)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print(output.size())
    print(torch.__version__)


