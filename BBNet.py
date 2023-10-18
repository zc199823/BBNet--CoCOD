import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from Res2Net_v1b import res2net50_v1b_26w_4s
#from model.two.FM_F import FM_F
import numpy as np
import data

######## Block ########

class Block(nn.Module):
    def __init__(self, channel):
        super(Block, self).__init__()

        self.stack2 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(channel, channel, kernel_size = 3, stride = 1, padding = 1),
             )

        self.stack4 = nn.Sequential(
            nn.Upsample(scale_factor = 4, mode='bilinear', align_corners=True),
            nn.MaxPool2d(kernel_size = 4, stride = 4),
            nn.Conv2d(channel, channel, kernel_size = 3, stride = 1, padding = 1),
             )

        self.stack5 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True),
            nn.Conv2d(channel, channel, kernel_size = 3, stride = 1, padding = 1),
             )

        self.conv = nn.Conv2d(channel*3, channel, kernel_size = 3, stride = 1, padding = 1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, f):

        res1 = f + self.stack2(f)
        res1 = res1 + self.stack2(f)
        
        res2 = f + self.stack4(f)
        res2 = res2 + self.stack4(f)

        res3 = self.stack5(f)
        res3 = res3 + self.stack5(f)

        out = torch.cat((res1,res2,res3),1)
        #out = f + self.conv(self.avgpool(out))
        out = f * self.soft(self.conv(self.avgpool(out)))

        return   out

############ fusion module###########

class FM_F (nn.Module):
    def __init__(self, channel):
        super(FM_F, self).__init__()

        self.block = Block(channel)
        self.conv = nn.Conv2d(2 * channel, channel, kernel_size = 1, stride = 1, padding = 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_h, f_l):

        f_h = self.block(f_h)
        f_l = self.block(f_l)
        f1 = f_h + f_l

        f_h_in = torch.reshape(f_h,[f_h.shape[0],1,f_h.shape[1],f_h.shape[2],f_h.shape[3]])
        f_l_in = torch.reshape(f_l,[f_l.shape[0],1,f_l.shape[1],f_l.shape[2],f_l.shape[3]])
        f_cat  = torch.add(f_h_in, f_l_in)
        f2 = f_cat.max(dim=1)[0]

        f = torch.cat((f1, f2), 1)
        f = self.conv(f)
        f = self.block(f)


        return  f


def resize(input, target_size=(256, 256)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class SA(nn.Module):
    def __init__(self, in_dim):
        super(SA, self).__init__()
        # Co-attention
        self.query_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.key_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.scale = 1.0 / (in_dim ** 0.5)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.size()
        # Co-attention Module
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)  # [10, 49, 64]
        proj_query = torch.transpose(proj_query, 1, 2).contiguous().view(-1, C)  # BHW, C

        proj_key = self.key_conv(x).view(B, -1, W * H)  # [10,64,49]
        proj_key = torch.transpose(proj_key, 0, 1).contiguous().view(C, -1)  # C, BHW

        x_w = torch.matmul(proj_query, proj_key)  # * self.scale # BHW, BHW

        x_w = x_w.view(B * H * W, B, H * W)  # [BHW, B, HW]
        out_max = torch.max(x_w, dim=-1)[0]  # [BHW, B] 144,1
        out_avg = torch.mean(x_w, dim=-1)  # [BHW, B] 144,1

        out_co = out_max + out_avg  # [BHW, B]144,1
        out_avg = out_co.mean(-1)  # [BWH]144

        x_co = out_avg.view(B, -1) * self.scale
        x_co = F.softmax(x_co, dim=-1)
        x_co = x_co.view(B, H, W).unsqueeze(1)
        out = x * x_co
        out = self.conv6(out)

        return out


class Featurechunk(nn.Module):
    def __init__(self):
        super(Featurechunk, self).__init__()

    def forward(self, x3_img):
        H = x3_img.size(2)
        W = x3_img.size(3)

        x2top_img = x3_img[:, :, 0:int(H / 3), :]  # [b, c, h//2, w]
        x2mid_img = x3_img[:, :, int(H / 3):int(2*H / 3), :]
        x2bot_img = x3_img[:, :, int(2*H / 3):H, :]  # [b, c, h//2, w]

        x1ltop_img = x2top_img[:, :, :, 0:int(W / 3)]  # [b, c, h//2, w//2]
        x1mtop_img = x2top_img[:, :, :, int(W / 3):int(2*W /3)]
        x1rtop_img = x2top_img[:, :, :, int(2*W / 3):W]  # [b, c, h//2, w//2]

       
        x1lmid_img = x2mid_img[:, :, :, 0:int(W / 3)]  # [b, c, h//2, w//2]
        x1mmid_img = x2mid_img[:, :, :, int(W / 3):int(2*W /3)]
        x1rmid_img = x2mid_img[:, :, :, int(2*W / 3):W]  # [b, c, h//2, w//2]

        x1lbot_img = x2bot_img[:, :, :, 0:int(W / 3)]  # [b, c, h//2, w//2]
        x1mbot_img = x2bot_img[:, :, :, int(W / 3):int(2*W /3)]  # [b, c, h//2, w//2]
        x1rbot_img = x2bot_img[:, :, :, int(2*W / 3):W]  # [b, c, h//2, w//2]

        return x1ltop_img,x1mtop_img, x1rtop_img, x1lmid_img, x1mmid_img, x1rmid_img, x1lbot_img, x1mbot_img,  x1rbot_img


class Featurecat(nn.Module):
    def __init__(self):
        super(Featurecat, self).__init__()

    def forward(self, x1ltop_img,x1mtop_img, x1rtop_img, x1lmid_img, x1mmid_img, x1rmid_img, x1lbot_img, x1mbot_img,  x1rbot_img):
        y1 = torch.cat([x1ltop_img, x1mtop_img, x1rtop_img], 3)
        y2 = torch.cat([x1lmid_img, x1mmid_img, x1rmid_img], 3)
        y3 = torch.cat([x1lbot_img, x1mbot_img, x1rbot_img], 3)
        y = torch.cat([y1, y2,y3], 2)

        return y


class Decoder(nn.Module):

    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.conv1 = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv5 = BasicConv2d(channel, 1, 3, padding=1)
        self.conv_f1 = nn.Sequential(
            BasicConv2d(10 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.upsample1 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True)

        self.sa = SA(32)

        self.conv_fg =BasicConv2d(channel*2, channel,3,1,1)
        self.convfs = BasicConv2d(channel, channel, 3, 1, 1)

        self.chunk = Featurechunk()
        self.cat = Featurecat()

    def forward(self, x1, x2, x3):
        f3 = self.upsample2(x3)  # [8,8]
        f2 = self.upsample1(x2)  # [8,8]

        f1 = torch.cat([f3, f2, x1], dim=1)
        f1 = self.conv1(f1)
        fl = fg = f1 #10 32 12 12

        # ----------Local feature extraction---------
        fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8,fl9 = self.chunk(fl)#   10 32 4 4

        f_r = self.chunk(fl)
        sigmoid1 = torch.max(torch.sigmoid(fl1))#10 32 4 4
        sigmoid2 = torch.max(torch.sigmoid(fl2))
        sigmoid3 = torch.max(torch.sigmoid(fl3))
        sigmoid4 = torch.max(torch.sigmoid(fl4))
        sigmoid5 = torch.max(torch.sigmoid(fl5))
        sigmoid6 = torch.max(torch.sigmoid(fl6))
        sigmoid7 = torch.max(torch.sigmoid(fl7))
        sigmoid8 = torch.max(torch.sigmoid(fl8))
        sigmoid9 = torch.max(torch.sigmoid(fl9))

        sigmoid1 = sigmoid1.cpu().detach().numpy()
        sigmoid2 = sigmoid2.cpu().detach().numpy()
        sigmoid3 = sigmoid3.cpu().detach().numpy()
        sigmoid4 = sigmoid4.cpu().detach().numpy()
        sigmoid5 = sigmoid5.cpu().detach().numpy()
        sigmoid6 = sigmoid6.cpu().detach().numpy()
        sigmoid7 = sigmoid7.cpu().detach().numpy()
        sigmoid8 = sigmoid8.cpu().detach().numpy()
        sigmoid9 = sigmoid9.cpu().detach().numpy()
        sigmoid = [sigmoid1,sigmoid2,sigmoid3,sigmoid4,sigmoid5,sigmoid6,sigmoid7,sigmoid8,sigmoid9]

        arr = np.array(sigmoid)
        arr = np.sort(arr)
        arr = np.argsort(-arr)
          
        f_s_1 = f_r[arr[0]]
        f_s_1 = self.upsample3(f_s_1)    

        # ----------Global feature extraction---------
     
        fg = self.sa(fg)
        f_s_1 = self.sa(f_s_1) 

        fgl_cat = self.convfs(fg + self.sa(f_s_1))

        out = self.conv4(fgl_cat) + fg
        out = self.conv5(out)
        return out

class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()

        self.weak_gra = Decoder_Block(channel)
        self.medium_gra = Decoder_Block(channel)
        self.strong_gra = Decoder_Block(channel)

    def forward(self, x, y):
        # reverse guided block
        y = -1 * (torch.sigmoid(y)) + 1

        # three group-reversal attention blocks

        y = self.weak_gra(x, y)
        y = self.medium_gra(x, y)
        y = self.strong_gra(x, y)

        return y


class Decoder_Block(nn.Module):
    def __init__(self, in_channel):
        super(Decoder_Block, self).__init__()
        self.cmprs = nn.Conv2d(in_channel, 32, 1)
        self.conv1_32 = nn.Conv2d(1, 32, 1)
        self.conv32_1 = nn.Conv2d(32, 1, 1)
        self.merge_conv = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

    def forward(self, low_level_feats, cosal_map):
        _, _, H, W = low_level_feats.shape
        cosal_map = resize(cosal_map, [H, W])
        cmprs = self.cmprs(low_level_feats)
        cosal_map2 = self.conv1_32(cosal_map)
        new_cosal_map = self.merge_conv(torch.cat([cmprs * cosal_map, cosal_map2], dim=1))
        new_cosal_map = self.conv32_1(new_cosal_map)
        return new_cosal_map


class OFS(nn.Module):
    def __init__(self, channel):
        super(OFS, self).__init__()
        self.sab = TRM(channel)

    def forward(self, x):

        sab = self.sab(x)

        return sab, map

class TRM(nn.Module):
    def __init__(self, in_dim):
        super(TRM, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.scale = 1.0 / (in_dim ** 0.5)
        self.conv6 = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        B, C, height, width = x.size()
        proj_query = self.query_conv(x).view(B, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, width * height)

        energy = torch.bmm(proj_query, proj_key) 
        attention = self.softmax(energy) 
        
        proj_value = self.value_conv(x).view(B, -1, width * height) 

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) 
        out = out.permute(2,0,1)

        out_max = torch.max(out, dim=-1)[0] 
        out_avg = torch.mean(out, dim=-1)


        out_co = out_max + out_avg  
        out_avg = out_co.mean(-1)       
        x_co = out_avg.view(B, -1) * self.scale
        x_co = F.softmax(x_co, dim=-1)
        x_co = x_co.view(B, height, width).unsqueeze(1)
        out = x * x_co
        out = self.conv6(out)
        
        out = out.view(B, C, height, width)

        out = self.gamma * out + x + out
        
        return out


class H_shuff(nn.Module):
    def _init_(self):
        super(H_shuff, self)._init_()

    def forward(self, x):
        N,C,H,W = x.size()
        groups = 6
        out = x.view(N, C, groups, H // groups, W).permute(0, 1, 3, 2, 4).reshape(N,C,-1,W)
        return out

class W_shuff(nn.Module):
    def _init_(self):
        super(W_shuff, self)._init_()

    def forward(self, x):
        N,C,H,W = x.size()
        groups = 6
        out = x.view(N, C, H, groups, W // groups).permute(0, 1, 2, 4, 3).reshape(N,C,H,-1)
        return out


class BBNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(BBNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)

        # ------Channel reduction-----
        self.reduce_conv4 = Reduction(512, channel)
        self.reduce_conv5 = Reduction(1024, channel)
        self.reduce_conv6 = Reduction(2048, channel)

        # ---- Decoder ----
        self.decoder = Decoder(channel)

        # ---- reverse stage ----
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

        self.cr4 = nn.Sequential(nn.Conv2d(2048, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(1024, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())
        self.ofs = OFS(32)

        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.sig = nn.Sigmoid()

        self.refine_3 = Decoder_Block(256)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(1,1)

        self.FM = FM_F(32)

        self.H = H_shuff()
        self.W = W_shuff()

        self.de = nn.Sequential(nn.Conv2d(96, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU())

    def forward(self, x):
#--------------Single-------------------
        a = len(x)
        b = random.randint(0,a-1)
        x_1 = x[b]
        C,H,W = x_1.size()
        x_s = x_1.view(-1, C, H, W)

        x_s = self.resnet.conv1(x_s)
        x_s = self.resnet.bn1(x_s)
        x_s = self.resnet.relu(x_s)
        x_s = self.resnet.maxpool(x_s)      # bs, 64, 64, 64
        x_s1 = self.resnet.layer1(x_s)      # bs, 256, 64, 64
        x_s2 = self.resnet.layer2(x_s1)     # bs, 512, 32, 32
        x_s3 = self.resnet.layer3(x_s2)     # 16,16
        x_s4 = self.resnet.layer4(x_s3)     # bs, 2048, 8, 8

        # Channel reduction
        cr4 = self.reduce_conv6(x_s4)
        ofs, predict4 = self.ofs(cr4) 

#--------------Couple-------------------
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 64, 64
        x1 = self.resnet.layer1(x)      # bs, 256, 64, 64
        x2 = self.resnet.layer2(x1)     # bs, 512, 32, 32
        x_c3 = self.resnet.layer3(x2)     # 16,16
        x_c4 = self.resnet.layer4(x_c3)     # bs, 2048, 8, 8


        x4_rfb = self.reduce_conv4(x2)  # channel---->32 48
        x5_rfb = self.reduce_conv5(x_c3)  # channel---->32 24
        x6_rfb = self.reduce_conv6(x_c4)  # channel---->32 12

        x4_rfb_1 = F.interpolate(x4_rfb, scale_factor=0.25, mode='bilinear')
        x5_rfb_1 = F.interpolate(x5_rfb, scale_factor=0.5, mode='bilinear')
        
        cat = torch.cat((x4_rfb_1,x5_rfb_1,x6_rfb),1)
        
        #FS
        x6_rfb_1 = self.W(self.H(cat))

        #mv
        x6_rfb_1 = self.de(x6_rfb_1)
        x6_rfb = self.FM(x6_rfb,x6_rfb_1) 

        x6_rfb = torch.mul(ofs, x6_rfb)


        # LGR
        S_g = self.decoder(x6_rfb, x5_rfb, x4_rfb)  # [1,8,8]
        S_g_pred = F.interpolate(S_g, scale_factor=32, mode='bilinear')  # [1,256,256]

        # --------------------hight level -----------------------------
        # ---------------- stage 5 ---------------------------------
        guidance_g = F.interpolate(S_g, scale_factor=1, mode='bilinear')
        ra4_feat = self.RS5(x6_rfb, guidance_g)
        S_5 = ra4_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')

        # ------------------stage 4 ---------------------------------
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra3_feat = self.RS4(x5_rfb, guidance_5)
        S_4 = ra3_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')

        # ------------------stage 3 ---------------------------------
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        ra2_feat = self.RS3(x4_rfb, guidance_4)
        S_3 = ra2_feat + guidance_4
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')

        # -------------------low level --------------------------------
        hight_pred = self.sig(S_3)  # [1,32,32]
        S_2 = self.refine_3(x1, hight_pred)

        # -------------------heat map---------------------------------
        pred = F.interpolate(S_2, scale_factor=4, mode='bilinear')


        return pred


if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
