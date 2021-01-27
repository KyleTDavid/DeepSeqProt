
import torch.nn as nn

#building blocks for ResNet-like architectures, mostly copied from
#https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278

#convolutional layer with dynamic padding
class conv_auto(nn.Conv1d):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.padding =  (self.dilation[0] * self.kernel_size[0] - 1) // 2 

#deconvolutional layer with dynamic padding
class trans_auto(nn.ConvTranspose1d):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.padding =  (self.dilation[0] * self.kernel_size[0] - 1) // 2

#some number of convolutional blocks,
#then add original input, shortcut to handle cases when input size != output size
class residual_block(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    self.blocks = nn.Identity()
    self.activate = nn.ReLU()
    self.shortcut = nn.Identity()

  def forward(self, x):
    residual = x
    if self.should_apply_shortcut: residual = self.shortcut(x)
    x = self.blocks(x)
    x += residual 
    x = self.activate(x)
    return x

  @property
  def should_apply_shortcut(self):
    return self.in_channels != self.out_channels

#extend to include shortcut with expanded channels
class resnet_residual_block(residual_block):
  def __init__(self, in_channels, out_channels, conv, expansion=1, sampling=1, *args, **kwargs):
    super().__init__(in_channels, out_channels, *args, **kwargs)
    self.expansion, self.sampling, self.conv = expansion, sampling, conv
    if self.should_apply_shortcut:
      if conv.func.__name__ == 'conv_auto':
        self.shortcut = nn.Sequential(
            nn.Conv1d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.sampling, bias=False),
            nn.BatchNorm1d(self.expanded_channels))
      elif conv.func.__name__ == 'trans_auto':
        self.shortcut = nn.Sequential(
            nn.ConvTranspose1d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.sampling, bias=False),
            nn.BatchNorm1d(self.expanded_channels))
    else: None
      
  @property
  def expanded_channels(self):
    return self.out_channels * self.expansion
  
  @property
  def should_apply_shortcut(self):
    return self.in_channels != self.expanded_channels

#stack conv and batchnorm layer
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
  return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm1d(out_channels))

#extend to create basic resnet block
class basic_block(resnet_residual_block):
  expansion = 1
  def __init__(self, in_channels, out_channels, *args, **kwargs):
      super().__init__(in_channels, out_channels, *args, **kwargs)
      self.blocks = nn.Sequential(
          conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.sampling),
          nn.ReLU(),
          conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
      )

#extend to create bottleneck block as defined by the original authors
class bottleneck_block(resnet_residual_block):
  expansion = 4
  def __init__(self, in_channels, out_channels, *args, **kwargs):
    super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
    self.blocks = nn.Sequential(
        conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
        nn.ReLU(),
        conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.sampling),
        nn.ReLU(),
        conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
    )

#layer built from blocks stacked on top of each other
class layer(nn.Module):
    def __init__(self, in_channels, out_channels, sampling, block=basic_block, n=1, *args, **kwargs):
        super().__init__()
        sampling = sampling if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, sampling = sampling),
            *[block(out_channels * block.expansion, 
                    out_channels, sampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x