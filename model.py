import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels
import timm

from geoopt import PoincareBall, ManifoldParameter
from geoopt.utils import size2shape

######## Hyperbolic components #################################################

def create_ball(ball=None, c=None):
  """
  Adapted from: https://raw.githubusercontent.com/geoopt/geoopt/master/examples/mobius_linear_example.py
  Helper to create a PoincareBall.

  Sometimes you may want to share a manifold across layers, e.g. you are using scaled PoincareBall.
  In this case you will require same curvature parameters for different layers or end up with nans.

  Parameters
  ----------
  ball : geoopt.PoincareBall
  c : float

  Returns
  -------
  geoopt.PoincareBall
  """
  if ball is None:
    assert c is not None, "curvature of the ball should be explicitly specified"
    print(f'Creating Poincare ball with c={c}.')
    ball = PoincareBall(c)
  # else trust input
  return ball

def mobius_linear(input, weight, bias=None, nonlin=None, *, ball: PoincareBall):
  """
  Adapted from: https://raw.githubusercontent.com/geoopt/geoopt/master/examples/mobius_linear_example.py
  """
  output = ball.mobius_matvec(weight, input)
  if bias is not None:
    output = ball.mobius_add(output, bias)
  if nonlin is not None:
    output = ball.logmap0(output)
    output = nonlin(output)
    output = ball.expmap0(output)
  return output

class MobiusLinear(nn.Linear):
  """
  Adapted from: https://raw.githubusercontent.com/geoopt/geoopt/master/examples/mobius_linear_example.py
  """
  def __init__(self, *args, nonlin=None, ball=None, c=1.0, **kwargs):
    super().__init__(*args, **kwargs)
    # for manifolds that have parameters like Poincare Ball
    # we have to attach them to the closure Module.
    # It is hard to implement device allocation for manifolds in other case.
    self.ball = create_ball(ball, c)
    if self.bias is not None:
      self.bias = ManifoldParameter(self.bias, manifold=self.ball)
    self.nonlin = nonlin
    self.reset_parameters()

  def forward(self, input):
    return mobius_linear(
      input,
      weight=self.weight,
      bias=self.bias,
      nonlin=self.nonlin,
      ball=self.ball,
    )

  @torch.no_grad()
  def reset_parameters(self):
    nn.init.eye_(self.weight)
    self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
    if self.bias is not None:
      self.bias.zero_()

class Distance2PoincareHyperplanes(nn.Module):
  """
  Adapted from https://github.com/geoopt/geoopt/blob/master/examples/hyperbolic_multiclass_classification.ipynb
  """
  n = 0

  def __init__(
    self,
    in_features: int,   # input dimension. e.g 200
    out_features: int,  # number of hyperplanes (classes) e.g. 5
    reparameterize=False, # whether or not to use same vector for point on plane and orthogonal
    signed=True,
    squared=False,
    *,
    ball,
    std=1.0,
  ):
    super().__init__()
    self.signed = signed
    self.squared = squared
    # Do not forget to save Manifold instance to the Module
    self.ball = ball
    self.in_features = size2shape(in_features) # e.g. (200,)
    self.out_features = out_features                        # e.g. 5
    self.reparameterize = reparameterize
    self.p = ManifoldParameter(
      torch.empty(out_features, in_features), manifold=self.ball  #e.g. shape: (5, 200)
    )
    if self.reparameterize:
      self.a = self.p
    else:
      self.a = ManifoldParameter(
        torch.empty(out_features, in_features), manifold=self.ball 
      )
    self.std = std
    self.reset_parameters()

  def forward(self, input):
    input_p = input.unsqueeze(-self.n - 1)
    p = self.p.permute(1, 0)            # convert to col vect: in_features, out_features. e.g. shape: (200, 5)
    p = p.view(p.shape + (1,) * self.n) # does nothing for n = 0
    if self.reparameterize:
      a = p
    else:
      a = self.a.permute(1, 0)
      a = a.view(a.shape + (1,) * self.n)

    distance = self.ball.dist2plane(
      x=input_p, p=p, a=a, signed=self.signed, dim=-self.n - 2
    )
    if self.squared and self.signed:
      sign = distance.sign()
      distance = distance ** 2 * sign
    elif self.squared:
      distance = distance ** 2
    return distance

  def extra_repr(self):
    return (
      "in_features={in_features}, "
      "out_features={out_features}, "
      .format(**self.__dict__)
    )

  @torch.no_grad()
  def reset_parameters(self):
    direction = torch.randn_like(self.p)
    direction /= direction.norm(dim=-1, keepdim=True)
    distance = torch.empty_like(self.p[..., 0]).normal_(std=self.std) # TODO ???
    self.p.set_(self.ball.expmap0(direction * distance.unsqueeze(-1)))
    if self.reparameterize:
      self.a = self.p
    else:
      self.a.set_(self.ball.expmap0(direction * distance.unsqueeze(-1)))

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def activate_drop(m):
    classname = m.__class__.__name__
    if classname.find('Drop') != -1:
        m.p = 0.1
        m.inplace = True

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f = False, hyperbolic=False, c=1.0):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        self.hyperbolic = hyperbolic
        if self.hyperbolic:
            self.ball = create_ball(c=c)
        add_block = []
        if linear>0:
            if self.hyperbolic:
                add_block += [MobiusLinear(input_dim, linear, ball=self.ball)]
            else:
                add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        if self.hyperbolic:
            classifier = Distance2PoincareHyperplanes(in_features=linear, out_features=class_num, ball=self.ball)
        else:
            classifier = []
            classifier += [nn.Linear(linear, class_num)]
            classifier = nn.Sequential(*classifier)
            classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        if self.hyperbolic:
            x = self.ball.expmap0(x)
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False, linear_num=512):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if ibn==True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the swin_base_patch4_window7_224 Model
# pytorch > 1.6
class ft_net_swin(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_swin, self).__init__()
        model_ft = timm.create_model('swin_base_patch4_window7_224', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle)

    def forward(self, x):
        x = self.model.forward_features(x)
        # swin is update in latest timm>0.6.0, so I add the following two lines.
        x = self.avgpool(x.permute((0,2,1)))
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

class ft_net_convnext(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, circle=False, linear_num=512):
        super(ft_net_convnext, self).__init__()
        model_ft = timm.create_model('convnext_base', pretrained=True, drop_path_rate = 0.2)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        self.model = model_ft
        #self.model.apply(activate_drop)
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f = circle)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the HRNet18-based Model
class ft_net_hr(nn.Module):
    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512):
        super().__init__()
        model_ft = timm.create_model('hrnet_w18', pretrained=True)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.classifier = nn.Sequential() # save memory
        self.model = model_ft
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride = 2, circle=False, linear_num=512):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        if stride == 1:
            model_ft.features.transition3.pool.stride = 1
        self.model = model_ft
        self.circle = circle
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the Efficient-b4-based Model
class ft_net_efficient(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512):
        super().__init__()
        #model_ft = timm.create_model('tf_efficientnet_b4', pretrained=True)
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError:
            print('Please pip install efficientnet_pytorch')
        model_ft = EfficientNet.from_pretrained('efficientnet-b4')
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.head = nn.Sequential() # save memory
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.classifier = nn.Sequential()
        self.model = model_ft
        self.circle = circle
        # For EfficientNet, the feature dim is not fixed
        # for efficientnet_b2 1408
        # for efficientnet_b4 1792
        self.classifier = ClassBlock(1792, class_num, droprate, linear=linear_num, return_f=circle)
    def forward(self, x):
        #x = self.model.forward_features(x)
        x = self.model.extract_features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the NAS-based Model
class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5, linear_num=512):
        super().__init__()  
        model_name = 'nasnetalarge' 
        # pip install pretrainedmodels
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 4032
        self.classifier = ClassBlock(4032, class_num, droprate, linear=linear_num)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num=751, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x) #use our classifier.
        return x

class ft_net_hyperbolic(nn.Module):

    def __init__(self, class_num=751, droprate=0, stride=2, circle=False, ibn=False, linear_num=512):
        super(ft_net_hyperbolic, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        if ibn==True:
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.circle = circle

        self.classifier = ClassBlock(2048, class_num, droprate, linear=linear_num, return_f = circle, hyperbolic=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, linear=256, relu=False, bnorm=True))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x[:,:,i].view(x.size(0), x.size(1))
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y

class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y
'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = ft_net_hr(751)
    #net = ft_net_swin(751, stride=1)
    net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 224, 224))
    output = net(input)
    print('net output size:')
    print(output.shape)
