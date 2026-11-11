"""PiSA-SR adaptation: paired dehazing, one adapter, joint objective, no color alignment."""
import importlib
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel, BertTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from peft import LoraConfig
from safetensors.torch import save_file, load_file

PISA_TARGET_PATTERNS = ['to_k','to_q','to_v','to_out.0','conv','conv1','conv2','conv_in',
    'conv_shortcut','conv_out','proj_out','proj_in','ff.net.2','ff.net.0.proj']


def target_modules(unet):
    return [n.replace('.weight','') for n,p in unet.named_parameters()
            if 'bias' not in n and 'norm' not in n and any(s in n for s in PISA_TARGET_PATTERNS)]


def csd_surrogate(latent, conditional_x0, unconditional_x0):
    with torch.no_grad():
        denominator = (latent.float()-conditional_x0.float()).abs().mean((1,2,3),keepdim=True)
        if not torch.isfinite(denominator).all() or (denominator==0).any():
            raise FloatingPointError('Invalid CSD normalization')
        gradient = (unconditional_x0.float()-conditional_x0.float())/denominator
    return F.mse_loss(latent.float(), (latent.float()-gradient).detach())


def local_ram_tokenizer(path):
    tokenizer = BertTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


class PisaSingle(nn.Module):
    def __init__(self, config, training_resources=True):
        super().__init__()
        self.config, self.device = config, torch.device('cuda')
        root = config['model']
        self.unet = UNet2DConditionModel.from_pretrained(root,subfolder='unet',local_files_only=True).cuda()
        self.unet.requires_grad_(False)
        self.targets = target_modules(self.unet)
        self.unet.add_adapter(LoraConfig(r=config['rank'],lora_alpha=8,init_lora_weights='gaussian',
            target_modules=self.targets),adapter_name='dehaze')
        for n,p in self.unet.named_parameters():
            p.requires_grad_('lora_' in n)
        self.unet.enable_gradient_checkpointing()
        self.unet.enable_xformers_memory_efficient_attention()
        self.vae = AutoencoderKL.from_pretrained(root,subfolder='vae',local_files_only=True).cuda().eval().requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(root,subfolder='tokenizer',local_files_only=True)
        self.text_encoder = CLIPTextModel.from_pretrained(root,subfolder='text_encoder',local_files_only=True,
            torch_dtype=torch.float16).cuda().eval().requires_grad_(False)
        self.scheduler = DDPMScheduler.from_pretrained(root,subfolder='scheduler',local_files_only=True)
        if self.scheduler.config.prediction_type != 'epsilon':
            raise ValueError('PiSA CSD requires epsilon SD2.1-base')
        ram_module = importlib.import_module('ram.models.ram_lora')
        # Local resource-path adaptation; the upstream tokenizer and special tokens are unchanged.
        ram_module.init_tokenizer = lambda:local_ram_tokenizer(config['ram_tokenizer'])
        from ram.models.utils import load_checkpoint_swinlarge
        self.ram = ram_module.RAMLora(image_size=384,vit='swin_l')
        self.ram, msg = load_checkpoint_swinlarge(self.ram,config['ram_weights'],{'image_size':384,'vit':'swin_l'})
        self.ram_load_report = dict(missing=list(msg.missing_keys),unexpected=list(msg.unexpected_keys))
        # Upstream discards these generated masks and permits new zero-output LoRA parameters.
        unsafe = [k for k in msg.missing_keys if not any(s in k for s in
            ('relative_position_index','attn_mask','lora_'))]
        if unsafe:
            raise ValueError(f'RAM missing learned weights: {unsafe[:20]}')
        self.ram = self.ram.half().cuda().eval().requires_grad_(False)
        self.ram_transform = transforms.Compose([transforms.Resize((384,384)),
            transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
        self.teacher = None
        self.perceptual = None
        if training_resources:
            self.teacher = UNet2DConditionModel.from_pretrained(root,subfolder='unet',local_files_only=True,
                torch_dtype=torch.float16).cuda().eval().requires_grad_(False)
            self.teacher.enable_xformers_memory_efficient_attention()
            import lpips
            self.perceptual = lpips.LPIPS(net='vgg').cuda().eval().requires_grad_(False)
        self.eval_frozen()

    def eval_frozen(self):
        for module in [self.vae,self.text_encoder,self.ram,self.teacher,self.perceptual]:
            if module is not None:
                module.eval()

    def adapter_parameters(self):
        return [p for p in self.unet.parameters() if p.requires_grad]

    @torch.no_grad()
    def captions(self, image):
        from ram import inference_ram
        return inference_ram(self.ram_transform(image*.5+.5).half(),self.ram)

    @torch.no_grad()
    def encode_text(self, captions):
        tokens=self.tokenizer(captions,max_length=self.tokenizer.model_max_length,padding='max_length',
            truncation=True,return_tensors='pt').input_ids.cuda()
        return self.text_encoder(tokens)[0].float()

    def forward(self, image, captions=None, trace=False, alpha=1.0, mask=None):
        if captions is None:
            captions=self.captions(image)
        positive=self.encode_text(captions)
        # Training-only empty-text augmentation; evaluation uses the input's RAM tags.
        selected=self.encode_text(['']*len(captions)) if self.training and random.random()<.5 else positive
        with torch.no_grad():
            z_hazy=self.vae.encode(image.float()).latent_dist.sample()*self.vae.config.scaling_factor
        with torch.autocast('cuda',dtype=torch.float16):
            prediction=self.unet(z_hazy,torch.ones(image.shape[0],device='cuda',dtype=torch.long),
                encoder_hidden_states=selected).sample.float()
        if trace:
            prediction.retain_grad()
        if mask is None and alpha == 1.0:
            latent=z_hazy-prediction
        else:
            residual_mask = 1.0 if mask is None else F.interpolate(
                mask.to(device=prediction.device, dtype=prediction.dtype),
                size=prediction.shape[-2:], mode='bilinear', align_corners=False).clamp(0, 1)
            latent=z_hazy-alpha*residual_mask*prediction
        output=self.vae.decode(latent/self.vae.config.scaling_factor).sample.clamp(-1,1)
        return output,dict(latent=latent,prediction=prediction,positive=positive,captions=captions)

    def csd(self, latent, positive):
        negative=self.encode_text([self.config['negative_prompt']]*latent.shape[0])
        with torch.no_grad():
            timestep=torch.randint(20,500,(latent.shape[0],),device='cuda')
            noise=torch.randn_like(latent)
            noisy=self.scheduler.add_noise(latent,noise,timestep)
            pred=self.teacher(torch.cat([noisy,noisy]).half(),torch.cat([timestep,timestep]),
                encoder_hidden_states=torch.cat([negative,positive]).half()).sample.float()
            uncond,cond=pred.chunk(2)
            guided=uncond+7.5*(cond-uncond)
            a=self.scheduler.alphas_cumprod.to(latent.device)[timestep,None,None,None]
            conditional_x0=(noisy-(1-a).sqrt()*guided)/a.sqrt()
            unconditional_x0=(noisy-(1-a).sqrt()*uncond)/a.sqrt()
        return csd_surrogate(latent,conditional_x0,unconditional_x0)

    def losses(self, image, target, trace=False):
        captions=[f'{caption}, ' for caption in self.captions(target)]
        output,aux=self(image,captions,trace)
        mse=F.mse_loss(output.float(),target.float())
        perceptual=self.perceptual(output.float(),target.float()).mean()
        csd=self.csd(aux['latent'],aux['positive'])
        return dict(loss=mse+2*perceptual+csd,mse=mse,lpips=perceptual,csd=csd),aux

    def save_adapters(self,path):
        save_file({n:p.detach().cpu().contiguous() for n,p in self.unet.named_parameters() if p.requires_grad},str(path))

    def load_adapters(self,path):
        weights=load_file(str(path))
        self.load_adapter_state(weights)

    def load_adapter_state(self,weights):
        named={n:p for n,p in self.unet.named_parameters() if p.requires_grad}
        if named.keys()!=weights.keys():
            raise ValueError('Adapter key mismatch')
        with torch.no_grad():
            for n,p in named.items():
                p.copy_(weights[n])
