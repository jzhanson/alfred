import torch
import torch.nn as nn
from torchvision import models, transforms


class Resnet18(nn.Module):
    '''
    pretrained Resnet18 from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True,
            pretrained=True, device=torch.device('cuda')):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
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

        if use_conv_feat:
            self.model = nn.Sequential(*list(self.model.children())[:-2])

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
            pretrained=True, frozen=True):
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
            self.resnet_model = MaskRCNN(args, eval, share_memory,
                    pretrained=pretrained, device=self.device)
            self.output_size = 2048 * 7 * 7 # Specific to MaskRCNN
        else:
            self.resnet_model = Resnet18(args, eval, share_memory,
                    pretrained=pretrained, use_conv_feat=use_conv_feat,
                    device=self.device)
            if use_conv_feat:
                self.output_size = 512 * 7 * 7 # Specific to Resnet18
            else:
                self.output_size = 1000

        self.frozen = frozen
        if self.frozen:
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
                out.append(self.resnet_model(b))
        return torch.cat(out, dim=0)

    def forward(self, images, batch=32):
        return self.featurize(images, batch)
