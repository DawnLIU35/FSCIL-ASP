import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
from utils.inc_net import get_backbone, BaseNet
import copy
import math
import numpy as np
from torch.nn import functional as F
import time

def build_promptmodel(modelname='vit_base_patch16_224',  Prompt_Token_num=10, VPT_type="Deep", args=None):
    
    basic_model = timm.create_model(modelname, pretrained=True)
    if modelname in ['vit_base_patch16_224']:
        model = VPT_ViT(Prompt_Token_num=Prompt_Token_num,VPT_type=VPT_type, args=args)
    else:
        raise NotImplementedError("Unknown type {}".format(modelname))

    # drop head.weight and head.bias
    basicmodeldict=basic_model.state_dict()
    basicmodeldict.pop('head.weight')
    basicmodeldict.pop('head.bias')

    model.load_state_dict(basicmodeldict, False)
    
    model.head = torch.nn.Identity()
    
    model.Freeze()
    
    return model


class VPT_ViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, Prompt_Token_num=1,
                 VPT_type="Shallow", basic_state_dict=None, args=None):

        # Recreate ViT
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer)
                         
        print('Using VPT model')
        self.args = args
        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.VPT_type = VPT_type
        if VPT_type == "Deep":
            print("Using Deep Prompt")
            if self.args["TIP_init"] == 'zero':
                self.TIP = nn.Parameter(torch.zeros(depth, int(Prompt_Token_num/2), embed_dim))
            elif self.args["TIP_init"] == 'random':
                self.TIP = nn.Parameter(torch.randn(depth, int(Prompt_Token_num/2), embed_dim))
            self.Prompt_Encoder = PROMPT_Encoder(args, depth, prompt_length=int(Prompt_Token_num/2), prompt_featuers=embed_dim)
            self.Avg_TSP = torch.zeros(depth, int(Prompt_Token_num/2), embed_dim)  

        else:  # "Shallow"
            print("Using Shallow Prompt")
            if self.args["TIP_init"] == 'zero':
                self.TIP = nn.Parameter(torch.zeros(1, int(Prompt_Token_num/2), embed_dim))
            elif self.args["TIP_init"] == 'random':
                self.TIP = nn.Parameter(torch.randn(1, int(Prompt_Token_num/2), embed_dim))
            self.Prompt_Encoder = PROMPT_Encoder(args, 1, prompt_length=int(Prompt_Token_num/2), prompt_featuers=embed_dim)
            self.Avg_TSP = torch.zeros(1, int(Prompt_Token_num/2), embed_dim) 

        self.Prompt_Token_num = Prompt_Token_num

    def New_CLS_head(self, new_classes=15):
        self.head = nn.Linear(self.embed_dim, new_classes)

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.TIP.requires_grad = True
        try:
            for param in self.head.parameters():
                param.requires_grad = True
            for param in self.Prompt_Encoder.fc_mu.parameters():
                param.requires_grad = True
            for param in self.Prompt_Encoder.fc_std.parameters():
                param.requires_grad = True
        except:
            pass

    def obtain_prompt(self):
        return 0

    def load_prompt(self, prompt_state_dict):
        pass

    def forward_features(self, x, perturb_var=0):
        Prompt_Token_num = self.TIP.shape[1] * 2

        tsp, (mu, std) = self.Prompt_Encoder(x, self.TIP, perturb_var)        
        
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.VPT_type == "Deep":
            for i in range(len(self.blocks)):
                TIP = self.TIP[i].unsqueeze(0)
                TIP = TIP.expand(x.shape[0], -1, -1)
                TSP = self.args["avg_alpha"] * self.Avg_TSP[i].expand(x.shape[0], -1, -1).to(x.device) + (1-self.args["avg_alpha"]) * tsp[:,i,:,:]

                Prompt_Tokens = torch.cat([TIP, TSP], dim=1)
                x = torch.cat((x, Prompt_Tokens), dim=1)
                num_tokens = x.shape[1]
                x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

        else:  # self.VPT_type == "Shallow"
            tsp = self.args["avg_alpha"] * self.Avg_TSP[0].expand(x.shape[0], -1, -1).to(x.device) + (1-self.args["avg_alpha"]) * tsp
            TIP = self.TIP.expand(x.shape[0], -1, -1)
            TSP = tsp[:,0,:,:]

            Prompt_Tokens = torch.cat([TIP, TSP], dim=1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            num_tokens = x.shape[1]
            x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

        x = self.norm(x)
        return x, (mu, std)

    def forward(self, x, perturb_var=0):
        x, (mu, std) = self.forward_features(x, perturb_var)
        x=x[:, 0, :]
        return x, (mu, std)

class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, perturb_var=0):
        x, (mu, std) = self.backbone(x, perturb_var)
        out = self.fc(x)
        out.update({"features": x})
        out.update({"kl": (mu, std) })
        return out

class PROMPT_Encoder(nn.Module):
    def __init__(self, args, depth, prompt_length, prompt_featuers=768):
        super(PROMPT_Encoder, self).__init__()
        self.prompt_length = prompt_length
        self.prompt_featuers = prompt_featuers
        self.depth = depth
        self.prompt_featuers=prompt_featuers

        newargs=copy.deepcopy(args)
        newargs['backbone_type']=newargs['backbone_type'].replace('_vpt','')
        self.prompt_backbone = get_backbone(newargs)

        self.fc_mu = nn.Sequential(
            nn.Linear(prompt_featuers*2, 256),
            nn.Linear(256, prompt_length*prompt_featuers)
        )
        self.fc_std = nn.Sequential(
            nn.Linear(prompt_featuers*2, 256),
            nn.Linear(256, prompt_length*prompt_featuers)
        )

    def forward(self, x, tip, perturb_var=0):
        bs = x.size(0)
        fea_x = self.prompt_backbone(x) 

        tip = tip.detach()[:,0,:].expand(bs, -1, -1).reshape(-1, self.prompt_featuers)   
        fea_x = fea_x.unsqueeze(1).expand(-1, self.depth, -1).reshape(-1, self.prompt_featuers)         
        fea = torch.cat([tip, fea_x], dim=1)

        mu = self.fc_mu(fea)
        std = F.softplus(self.fc_std(fea)-5, beta=1)
        prompt = self.reparameterise(mu, std, perturb_var)

        prompt = prompt.reshape(bs, self.depth, self.prompt_length, self.prompt_featuers)
        return prompt, (mu, std)

    def reparameterise(self, mu, std, perturb_var):
        eps = torch.randn_like(std)*perturb_var
        return mu + std*eps


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}

    
def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)