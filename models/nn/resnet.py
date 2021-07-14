import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# CustomResNet adapted from source code of torchvision.models's ResNet
# https://pytorch.org/docs/0.3.0/_modules/torchvision/models/resnet.html
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def convtrans3x3(in_planes, out_planes, stride=1):
    "3x3 transposed convolution with padding"
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3,
            stride=stride, padding=1, output_padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, convtrans=False):
        super(BasicBlock, self).__init__()
        if convtrans and stride != 1:
            self.conv1 = convtrans3x3(inplanes, planes, stride)
        else:
            # Use normal conv if stride == 1, otherwise ConvTranspose2d will
            # complain about output_padding not being less than stride
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CustomResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], inplanes=64,
            planes=[64, 128, 256, 512], num_classes=1000):
        self.inplanes = inplanes
        super(CustomResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes[0], layers[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(planes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class CustomResNetDecoder(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], inplanes=512,
            planes=[512, 256, 128, 64], num_classes=1000):
        self.inplanes = inplanes
        super(CustomResNetDecoder, self).__init__()
        self.layer1 = self._make_layer(block, planes[1], layers[1], stride=2)
        self.layer2 = self._make_layer(block, planes[2], layers[2], stride=2)
        self.layer3 = self._make_layer(block, planes[3], layers[3], stride=2)
        self.layer4 = self._make_layer(block, planes[3], layers[3])

        # output_padding=1 so output is 224, not 223
        self.conv1 = nn.ConvTranspose2d(planes[3], 3, kernel_size=7, stride=2,
                padding=3, output_padding=1, bias=False)
        # https://github.com/Alvinhech/resnet-autoencoder/blob/master/autoencoder1.py#L139
        # TODO: do we need bn1, relu after conv1?
        #self.bn1 = nn.BatchNorm2d(inplanes)
        #self.relu = nn.ReLU(inplace=True)


        # TODO make sure this init works for ConvTranspose2d, and make sure the parameters are the same in decoder
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        # TODO: add option for no residual connections
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, output_padding=1,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        print('layer1/block1 layer 1', self.inplanes, planes, stride)
        layers.append(block(self.inplanes, planes, stride, upsample,
            convtrans=True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            print('layer1/block1 layer 2', self.inplanes, planes)
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        print('before interpolate', x.shape)
        # nn.Upsample calls F.interpolate, so it's the same thing
        x = F.interpolate(x, scale_factor=7)
        print('after interpolate', x.shape)
        x = self.layer1(x)
        print('after layer1', x.shape)
        x = self.layer2(x)
        print('after layer2', x.shape)
        x = self.layer3(x)
        print('after layer3', x.shape)
        x = self.layer4(x)
        print('after layer4', x.shape)
        # TODO: should we do anything else for average/max pooling inverses?
        x = F.interpolate(x, scale_factor=2)
        print('after interpolate (max pool)', x.shape)
        x = self.conv1(x)
        print('after conv1', x.shape)
        #x = self.bn1(x)
        #x = self.relu(x)
        #x = self.maxpool(x)

        return x


class Resnet18(nn.Module):
    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True,
            pretrained=True, layers=[2, 2, 2, 2], inplanes=64,
            planes=[64, 128, 256, 512], decoder=False,
            device=torch.device('cuda')):
        super(Resnet18, self).__init__()
        if pretrained and not decoder:
            self.model = models.resnet18(pretrained=pretrained)
        elif decoder:
            self.model = CustomResNetDecoder(layers=layers, inplanes=inplanes,
                    planes=planes)
        else:
            self.model = CustomResNet(layers=layers, inplanes=inplanes,
                    planes=planes)
        self.device=device

        if args.gpu:
            try:
                self.model = self.model.to(self.device)
            except:
                self.model = self.model.to(self.device)

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()

        if use_conv_feat and not decoder:
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        elif not decoder:
            # We still trim off the last 1000-way fc layer because those are
            # the ImageNet class activations and it's unlikely we'll ever need
            # those
            self.model = nn.Sequential(*list(self.model.children())[:-1])

    def extract(self, x):
        return self.model(x)

    def forward(self, x):
        return self.extract(x)


class MaskRCNN(nn.Module):
    '''
    pretrained MaskRCNN from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, min_size=224,
            pretrained=True, device=torch.device('cuda')):
        super(MaskRCNN, self).__init__()
        self.model = models.detection.maskrcnn_resnet50_fpn(
                pretrained=pretrained, min_size=min_size)
        self.model = self.model.backbone.body
        self.feat_layer = 3
        self.device = device

        if args.gpu:
            try:
                self.model = self.model.to(self.device)
            except:
                self.model = self.model.to(self.device)

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()


    def extract(self, x):
        features = self.model(x)
        return features[self.feat_layer]

    def forward(self, x):
        return self.extract(x)


class Resnet(nn.Module):

    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True,
            pretrained=True, frozen=True, layers=[2, 2, 2, 2], inplanes=64,
            planes=[64, 128, 256, 512]):
        super(Resnet, self).__init__()
        self.model_type = args.visual_model
        self.gpu = args.gpu
        if self.gpu:
            if hasattr(args, 'gpu_index'):
                self.device = torch.device('cuda:' + str(args.gpu_index))
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # choose model type
        if self.model_type == "maskrcnn":
            # MaskRCNN does not support custom layers/architecture
            self.resnet_model = MaskRCNN(args, eval, share_memory,
                    pretrained=pretrained, device=self.device)
            self.model = self.resnet_model
            self.output_size = 2048 * 7 * 7 # Specific to MaskRCNN
        elif self.model_type == 'resnet':
            self.resnet_model = Resnet18(args, eval, share_memory,
                    pretrained=pretrained, use_conv_feat=use_conv_feat,
                    layers=layers, inplanes=inplanes, planes=planes,
                    device=self.device)
            self.model = self.resnet_model
            if use_conv_feat:
                self.output_size = planes[3] * 7 * 7 # Specific to Resnet18
            else:
                self.output_size = planes[3]
        elif self.model_type == 'resnet_ae':
            self.resnet_model = Resnet18(args, eval, share_memory,
                    pretrained=pretrained, use_conv_feat=use_conv_feat,
                    layers=layers, inplanes=inplanes, planes=planes,
                    device=self.device)
            # TODO: separate decoder architecture arguments from
            # model/model/args.py?
            self.decoder = Resnet18(args, eval, share_memory,
                    use_conv_feat=use_conv_feat, layers=layers[::-1],
                    inplanes=planes[-1], planes=planes[::-1],
                    decoder=True, device=self.device)
            self.model = nn.Sequential(self.resnet_model, self.decoder)

        self.frozen = frozen
        if self.frozen:
            # This only freezes the encoder, if resnet_ae is set
            for param in self.resnet_model.model.parameters():
                param.requires_grad = False

        # normalization transform
        self.transform = self.get_default_transform()


    @staticmethod
    def get_default_transform():
        # Unfortunately torchvision 0.7.0 does not support a torch.Tensor input
        # for transforms.resize in models/nn/resnet.py, only PIL.Image
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def featurize(self, images, batch=32):
        images_normalized = torch.stack([self.transform(i) for i in images], dim=0)
        if self.gpu:
            images_normalized = images_normalized.to(self.device)

        out = []
        with torch.set_grad_enabled(not self.frozen):
            for i in range(0, images_normalized.size(0), batch):
                b = images_normalized[i:i+batch]
                out.append(self.model(b))
        return torch.cat(out, dim=0)

    def forward(self, images, batch=32):
        return self.featurize(images, batch)
