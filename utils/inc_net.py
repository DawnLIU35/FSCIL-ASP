import copy
from torch import nn
import timm

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    
    # VPT
    elif '_vpt' in name:

        if args["model_name"] == "asp":
            from backbone.asp_backbone import build_promptmodel
            if name == "pretrained_vit_b16_224_vpt":
                basicmodelname = "vit_base_patch16_224" 
            
            print("modelname,", name, "basicmodelname", basicmodelname)
            VPT_type = "Deep"
            if args["vpt_type"] == 'shallow' or args["vpt_type"] == 'Shallow':
                VPT_type = "Shallow"
            Prompt_Token_num = args["prompt_token_num"]

            model = build_promptmodel(modelname=basicmodelname, Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type, args=args)
            prompt_state_dict = model.obtain_prompt()
            model.load_prompt(prompt_state_dict)

            if name == "pretrained_vit_b16_224_vpt":
                model.out_dim = 768
            
            return model.eval()

        else:
            raise NotImplementedError("Inconsistent model name and model type")

    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'
        self.model_type = 'vit'   

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
