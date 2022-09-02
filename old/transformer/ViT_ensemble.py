import torch.nn as nn
import torch
from torch.nn import init
from model.v_convgru import ConvGRU
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):       # Used in Transformer
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):       # Used in Transformer
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):     # Used in Transformer
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):       # Used in ViT
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):       # Used in ensemble_ViT
    def __init__(self, *, image_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls',
                 channels = 20, dim_head = 64, dropout = 0., emb_dropout = 0., seq_len = 4):
        super().__init__()

        self.image_size = image_size
        self.num_classes = num_classes
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c h w -> (b h w ) t c'),
            nn.Linear(channels, dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes))

        self.out_trans = Rearrange('(b h w) c -> b c h w', h = self.image_size, w = self.image_size)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        out = self.mlp_head(x)

        return self.out_trans(out)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.fill_(0.01)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class ChannelAttention_forAdd(nn.Module):       # Used in ensemble_ViT
    def __init__(self, num_feat, squeeze_factor=2):
        super(ChannelAttention_forAdd, self).__init__()
        self.relu = nn.ReLU()
        self.batch_f_norm = batch_f_norm(7)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        b,f,c,h,w = x.size()
        # x = self.relu(self.batch_f_norm(x))
        x = self.relu(x)
        ca = self.attention(x.view(b*f,c,h,w))
        out = x.view(b*f,c,h,w) * ca
        out = out.view(b,f,c,h,w)
        return out, ca

class batch_norm(nn.Module):        # Used in ensemble_ViT
    def __init__(self, num_feat):
        super(batch_norm, self).__init__()
        self.batch_norm = torch.nn.BatchNorm1d(num_feat)
    def forward(self, x):
        b,f,c,h,w = x.size() # 12, 4, 7, 9, 256, 256
        x_norm = self.batch_norm(x.reshape(b * f, c, h * w))  # 12*4*7, 9, 256*256
        x_out = x_norm.reshape(b,f,c,h,w)
        return x_out

class batch_f_norm(nn.Module):      # Used in ChannelAttention_forAdd
    def __init__(self, num_feat):
        super(batch_f_norm, self).__init__()
        self.batch_f_norm = torch.nn.BatchNorm1d(num_feat)
    def forward(self, x):
        b,f,c,h,w = x.size() # 12, 4, 7, 9, 256, 256
        x_norm = self.batch_f_norm(x.reshape(b * c, f, h * w))  # 12*4*9, 7, 256*256
        x_out = x_norm.reshape(b,f,c,h,w)
        return x_out

class v_cnn(nn.Module):     # Used in ensemble_ViT
    def __init__(self, input_chans, out_dim, filter_size, multi_temporal = True):
        super(v_cnn, self).__init__()
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.multi_temporal = multi_temporal
        self.padding = (filter_size - 1) // 2  # in this way the output has the same size
        self.hidden_dims = out_dim
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_chans, self.hidden_dims, self.filter_size, 1, self.padding),
            nn.BatchNorm2d(self.hidden_dims),
            nn.Sigmoid())

    def forward(self, input):
        if self.multi_temporal == True:
            b,t,c,h,w = input.size()
            tensor = input[:, -1, :, :, :].reshape(b,c,h,w)
            out = self.conv(tensor)
            out = out.reshape(b,self.hidden_dims,h,w)
        else:
            out = self.conv(input)

        return out


class ensemble_ViT(nn.Module):      # Final model
    def __init__(self, n_classes,seq_len =4, inference = False):
        super(ensemble_ViT, self).__init__()
        self.inference = inference
        self.n_classes = n_classes
        self.input_dim = 9

        # ViT hyper-parameters
        self.depth = 1
        self.mlp_dim = 32
        self.head = 8
        self.img_size =256
        self.dim = 20
        self.dim_head =16
        self.seq_len =seq_len

        self.vit_1 = ViT(image_size= self.img_size, num_classes=self.n_classes, dim=self.dim,
                         depth=self.depth,heads=self.head, mlp_dim=self.mlp_dim, pool = 'cls',
                         channels = 6, dim_head = self.dim_head,dropout = 0.1, emb_dropout = 0.1, seq_len = self.seq_len)
        self.vit_2 = ViT(image_size= self.img_size, num_classes=self.n_classes, dim=self.dim,
                         depth=self.depth,heads=self.head, mlp_dim=self.mlp_dim, pool = 'cls',
                         channels = 1, dim_head = self.dim_head,dropout = 0.1, emb_dropout = 0.1, seq_len = self.seq_len)
        self.vit_3 = ViT(image_size= self.img_size, num_classes=self.n_classes, dim=self.dim,
                         depth=self.depth,heads=self.head, mlp_dim=self.mlp_dim, pool = 'cls',
                         channels = 1, dim_head = self.dim_head,dropout = 0.1, emb_dropout = 0.1, seq_len = self.seq_len)
        self.cnn_4 = v_cnn(input_chans=1, out_dim=self.n_classes, filter_size=3)
        self.vit_5 = ViT(image_size= self.img_size, num_classes=self.n_classes, dim=self.dim,
                         depth=self.depth,heads=self.head, mlp_dim=self.mlp_dim, pool = 'cls',
                         channels = 1, dim_head = self.dim_head,dropout = 0.1, emb_dropout = 0.1, seq_len = self.seq_len)
        self.cnn_6 = v_cnn(input_chans=1, out_dim=self.n_classes, filter_size=3)
        self.cnn_7 = v_cnn(input_chans=1, out_dim=self.n_classes, filter_size=3)
        self.vit_8 = ViT(image_size= self.img_size, num_classes=self.n_classes, dim=self.dim,
                         depth=self.depth,heads=self.head, mlp_dim=self.mlp_dim, pool = 'cls',
                         channels = 5, dim_head = self.dim_head,dropout = 0.1, emb_dropout = 0.1, seq_len = self.seq_len)
        self.vit_9 = ViT(image_size= self.img_size, num_classes=self.n_classes, dim=self.dim,
                         depth=self.depth,heads=self.head, mlp_dim=self.mlp_dim, pool = 'cls',
                         channels = 2, dim_head = self.dim_head,dropout = 0.1, emb_dropout = 0.1, seq_len = self.seq_len)
        self.norm = batch_norm(num_feat=self.input_dim)

        self.ca = ChannelAttention_forAdd(self.input_dim)

    def forward(self, x):
        x_1 = self.vit_1(x[:, :, 1:self.n_classes, :, :]) # lulc
        x_2 = self.vit_2(torch.unsqueeze(x[:, :, self.n_classes, :, :], dim=2)) # housing
        x_3 = self.vit_3(torch.unsqueeze(x[:, :, self.n_classes + 1, :, :], dim=2)) # metro
        x_4 = self.cnn_4(torch.unsqueeze(x[:, :, self.n_classes + 2, :, :], dim=2))
        x_5 = self.vit_5(torch.unsqueeze(x[:, :, self.n_classes + 3, :, :], dim=2)) # policy
        x_6 = self.cnn_6(torch.unsqueeze(x[:, :, self.n_classes + 4, :, :], dim=2))
        x_7 = self.cnn_7(torch.unsqueeze(x[:, :, self.n_classes + 5, :, :], dim=2))
        x_8 = self.vit_8(x[:, :, self.n_classes+6 : self.n_classes+11, :, :]) # dis_roads
        x_9 = self.vit_9(x[:, :, self.n_classes+11 :, :, :]) # dis_urban

        combined = torch.stack([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9], dim=2)
        combined_norm = self.norm(combined)
        combined_ca, ca = self.ca(combined_norm)
        out = torch.sum(combined_ca, dim=2) # b,f,c,h,w -> # b,f,h,w , b,h,w

        if self.inference == True:
            return [out, ca, combined_norm]
        else:
            return [out, ca]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    input = torch.rand(2, 4, 13, 256, 256)

    net = ensemble_ViT(n_classes=7)
    pred_list = net(input)
    print(pred_list[0].size())


