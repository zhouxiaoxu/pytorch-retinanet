import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    '''
        输入 c3,c4, c5分别对应，layer2,layer3，layer4的输出
        c5 通过NIN转换为256x7x7, 然后通过3x3的卷积提取特征， 输出P5
        c4 通过NIN转化为256x14x14，然后和P5_upsample融合获得新的特征，最后通过3x3提取特征，输出P4
        C3 通过NIN转化为256x28x28，然后和P4_upsample融合获得新的特征，最后通过3x3提取特征，输出P3

        C5 通过3x3 pading =2 的卷积直接提取特征获取P6
        P6 通过3x3 pading =2 的卷积直接提取特征获取P7特征

    '''
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):    # c3 128  C4_size 256  C5_size 256
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper  对layer4的输出 512x7x7 进行处理 转化为256x14x14
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)    # 512 256   
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)   

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)    # 对layer3的输出C4做1x1卷积
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)    # 对C5(即layer4的输出512x7x7)做1x1卷积， 输出256x7x7
        P5_upsampled_x = self.P5_upsampled(P5_x)    # 上采样，输入256x14x14
        P5_x = self.P5_2(P5_x)     # 对256x7x7 提取特征

        P4_x = self.P4_1(C4)    # 对C4(即layer3的输出256x14x14)做1x1卷积，输出256x14x14
        P4_x = P5_upsampled_x + P4_x    # 融合P4和P5_upsampled，输出256x14x14
        P4_upsampled_x = self.P4_upsampled(P4_x)    # 对融合后P4_x上采样，输出256x28x28(方便和P3融合)
        P4_x = self.P4_2(P4_x)  # 对256x14x14提取特征

        P3_x = self.P3_1(C3)    # 对c3(即layer2输出的128x28x28)做1x1卷积，输出256x28x28
        P3_x = P3_x + P4_upsampled_x    #融合P3和P4_upsampled,输出256x28x28
        P3_x = self.P3_2(P3_x)  # 对256x28x28提取特征

        P6_x = self.P6(C5)  # 对C5(即layer4的输出512x7x7)做3x3卷积提取特征

        P7_x = self.P7_1(P6_x)  # 对P6做3x3卷积提取特征
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]   # 特征金字塔有5层特征看，


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors  通过5个卷积层，channel的值变成4*num_anchors，使用channel表示box的坐标
        out = out.permute(0, 2, 3, 1)
        # view只能用在contiguous的内存中，如果之前使用permute或transpose，需要使用contiguous返回一个连续的副本在执行view操作
        return out.contiguous().view(out.shape[0], -1, 4)   


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes * n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        '''
            输入的图像的尺寸为224x224
            block: block的类型，可选的包括 retinanet.utils.BasicBlock和Bottleneck
            layers: list，len为4

        '''
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)   # 创建首个卷积层 conv-bn-relu-maxpool
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 构建FPN模块的输出通道数
        if block == BasicBlock:     # 获取layer2,、layer3、layer4的输出通道数 128,256,512
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels] #  resnet18时，该值为[128, 256,512]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])    # 创建特征金字塔PyramidFeatures

        self.regressionModel = RegressionModel(256) # 回归模型
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)    # 分类模型

        self.anchors = Anchors()    # anchors

        self.regressBoxes = BBoxTransform() # boxes精修

        self.clipBoxes = ClipBoxes()    

        self.focalLoss = losses.FocalLoss() # focal loss

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        # 初始化参数
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):    
        '''
        功能： 创建多个block块
        参数：
            block：指定模块类型BasicBlock 或者 Bottleneck
            planes： layer输出的channel数量
            blocks： block块的重复次数

        例如：
        # self.layer4 = self._make_layer(block, 512, 2, stride=2)  
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:    #  只有输出特征map变化，或者通道数翻倍时，才会进行下采样。这样block的输入和输出才能融合
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,  # 1x1卷积， stride=2，将通道数翻倍，特征map压缩为原来的1/2
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)] # 创建第一个block
        self.inplanes = planes * block.expansion    # 每次创建后，都会更新inplanes值
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes)) # 附加后续block

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])
        # 对特征金字塔的每一层输入到回归模型,预测出box框，返回值为batchsize x box数量 x box的4个坐标
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        # 对特征金字塔的每一层输入到分类模型，预测出box的分类信息，返回值为 batch size  x box数量 x 各分类的得分
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)   # 根据输入图片的尺寸，计算所有anchors


        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)    # 使用anchors数据和regression数据，精修box坐标
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)    #

            scores = torch.max(classification, dim=2, keepdim=True)[0]
            scores_over_thresh = (scores > 0.05)[:, :, 0]

    
            val = []
            for i in range(scores.shape[0]):
                one_image_scores_over_thresh = scores_over_thresh[i,:]

                if one_image_scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just return
                    #val.append([torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)])
                    val.append([torch.zeros([1]).cuda(0), torch.zeros([1]).cuda(0), torch.zeros([1, 4]).cuda(0)])
                    continue

                one_image_classification = classification[i, one_image_scores_over_thresh, :]
                one_image_transformed_anchors = transformed_anchors[i, one_image_scores_over_thresh, :]
                one_image_scores = scores[i, one_image_scores_over_thresh, :].reshape(-1)   # 不能使用sequeeze

                anchors_nms_idx = nms(one_image_transformed_anchors, one_image_scores, 0.5)   

                nms_scores, nms_class = one_image_classification[anchors_nms_idx, :].max(dim=1)    

                val.append([nms_scores, nms_class, one_image_transformed_anchors[anchors_nms_idx, :]])
            return val


def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='./models'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='./models'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
