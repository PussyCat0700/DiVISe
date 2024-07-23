import clip
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from avhubert.avhubert_as_upstream import AVHubertEncoder
from vocoders.bigvgan.bigvgan_model import BigVGAN
from constants import BIGVGAN_NO_GRAD, GRIFFINLIM, HIFIGAN_NO_GRAD, HIFIGAN_WITH_GRAD, PWG_NO_GRAD, UNIT_METHODS, UNIT_SPEECH_TOKENIZER_NO_GRAD, UNIT_HIFIGAN_NO_GRAD
from speechtokenizer import SpeechTokenizer
from utils import init_weights, get_padding, mpd_length_variators, msd_length_variators

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h, conv_indim, unit_nums=None):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.mode = UNIT_HIFIGAN_NO_GRAD if unit_nums is not None else None
        # initial upsampling layers
        if self.mode == UNIT_HIFIGAN_NO_GRAD:
            # lookup table as in https://arxiv.org/abs/2104.00355
            # The extra embedding to the end is used for padding in dataset collating.
            self.lut = nn.Embedding(unit_nums+1, conv_indim)
        self.conv_pre = weight_norm(Conv1d(conv_indim, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        if self.mode == UNIT_HIFIGAN_NO_GRAD:
            x = self.lut(x)
        x = x.transpose(-1, -2).contiguous()
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

class AVHuBERT2UnitHiFiGAN(nn.Module):
    def __init__(self, attention_dim, unit_nums) -> None:
        super().__init__()
        self.attention_dim = attention_dim
        # Define the transposed convolution layer
        # Assuming the number of input channels is also 768, change it if it's different
        self.transposed_conv = nn.ConvTranspose1d(in_channels=attention_dim*4, out_channels=unit_nums,
                                                  kernel_size=4, stride=2, padding=1)
        # Define the GeLU activation
        self.gelu = nn.GELU()
    
    def forward(self, encoder_out):
        # encoder_out is (B, T, C)
        # Apply transposed convolution
        x = self.transposed_conv(encoder_out.transpose(-1, -2)).transpose(-1, -2)
        # Apply GeLU activation
        x = self.gelu(x)
        return x
    
class SpeechTokenizerGenerator(nn.Module):
    def __init__(self, speechtokenizer_config) -> None:
        super().__init__()
        self.model = SpeechTokenizer.load_from_checkpoint(
            config_path=speechtokenizer_config['config_path'],
            ckpt_path=speechtokenizer_config['ckpt_path'],
            )
    
    def forward(self, x, st):
        return self.model.decode(x, st)
        
    
class AVHuBERTGenerator(nn.Module):
    def __init__(self, hifigenerator_config, avhubert_model_config, prosody_minmax_dict, unit_dict, hu_dict, generator_mode:str=GRIFFINLIM, use_farl=False) -> None:
        super().__init__()
        # Intuitively I think generating mel-spectrograms after conformer will be better regardless of generator.
        # To load runs done by previous commits, set mel_before_conformer to True.
        self.early_return = generator_mode in UNIT_METHODS
        self.frontend_with_encoder = AVHubertEncoder(avhubert_model_config, hifigenerator_config.num_mels, prosody_minmax_dict=prosody_minmax_dict, unit_dict=unit_dict, hu_dict=hu_dict, mel_before_conformer=False, early_return=self.early_return)
        self.generator_mode = generator_mode
        self.use_farl = use_farl
        self.with_generator = generator_mode != GRIFFINLIM
        self.with_extra_padding_unit = self.generator_mode != UNIT_SPEECH_TOKENIZER_NO_GRAD
        attention_dim = self.frontend_with_encoder.attention_dim
        eval_mode_for_vocoder = self.with_generator and self.generator_mode!=HIFIGAN_WITH_GRAD
        if self.generator_mode == HIFIGAN_WITH_GRAD:
            self.generator = Generator(hifigenerator_config, attention_dim)
        elif self.generator_mode == HIFIGAN_NO_GRAD:
            mel_dim = hifigenerator_config.num_mels
            self.generator = Generator(hifigenerator_config, mel_dim)
        elif self.generator_mode == BIGVGAN_NO_GRAD:
            self.generator = BigVGAN(hifigenerator_config)
        elif self.generator_mode == PWG_NO_GRAD:
            self.generator = nn.Identity()  # should be initialized and loaded by exterior module.
        elif self.generator_mode in UNIT_METHODS:
            n_units = hifigenerator_config.k
            if self.with_extra_padding_unit:
                n_units += 1
                self.generator = Generator(hifigenerator_config, hifigenerator_config.num_mels, unit_nums=hifigenerator_config.k)
            elif self.generator_mode == UNIT_SPEECH_TOKENIZER_NO_GRAD:
                self.generator = SpeechTokenizerGenerator(hifigenerator_config.speechtokenizer)
            self.unit_upsampler = AVHuBERT2UnitHiFiGAN(attention_dim, n_units)
        if eval_mode_for_vocoder:
            self.generator.eval()
            for param in self.generator.parameters():
                param.requires_grad = False
        if self.use_farl:
            self.farl_model, _ = clip.load("ViT-B/16")
            self.farl_proj = nn.Linear(512, 1024)
            for param in self.farl_model.parameters():
                param.requires_grad = False
    
    def forward(self, video, prosody_targets=None, unit_target=None, hu_target=None, farl_img_input=None, mel_masks=None):
        avhubert_input = {"video": video, "audio": None,}
        encoder_out = self.frontend_with_encoder(avhubert_input, prosody_targets, unit_target, hu_target, mel_masks)
        if self.use_farl:
            with torch.no_grad():
                farl_output = self.farl_model.encode_image(farl_img_input)
            # TODO suit not only early return
            farl_feature = self.farl_proj(farl_output.float().unsqueeze(1))
            encoder_out = encoder_out + farl_feature
        downsampled_encoder_out = None
        if self.with_generator:
            if self.generator_mode in [HIFIGAN_NO_GRAD, BIGVGAN_NO_GRAD, PWG_NO_GRAD]:
                with torch.inference_mode():
                    wav_generated = self.generator(encoder_out["melspec_out"])  # generator takes in tensor shaped (bs, mellen, num_mel)
            elif self.generator_mode == HIFIGAN_WITH_GRAD:
                wav_generated = self.generator(encoder_out["output"])  # generator takes in tensor shaped (bs, mellen, attention_dim)
            elif self.generator_mode in UNIT_METHODS:
                downsampled_encoder_out = self.unit_upsampler(encoder_out)
                # upsampler returns (bs, mellen/2, k)
                indices = downsampled_encoder_out.argmax(dim=-1)
                with torch.inference_mode():
                    if self.generator_mode == UNIT_HIFIGAN_NO_GRAD:
                        wav_generated = self.generator(indices)
                    elif self.generator_mode == UNIT_SPEECH_TOKENIZER_NO_GRAD:
                        wav_generated = self.generator(indices.unsqueeze(0), st=0).squeeze(0)
        else:
            wav_generated = None
        # (bs, mellen, num_mels) -> (bs, num_mels, mellen)
        ret = {
            "wav_generated":wav_generated,  # (bs, wavlen) or None if self.with_generator is False
            "melspec_out":None,
            "prosody": None,
            "unit": None,
            "hu": None,
            "revise_logits": downsampled_encoder_out,
        }
        if not self.early_return:
            mel_generated = encoder_out["melspec_out"]
            mel_generated = mel_generated.permute(0, 2, 1).contiguous()
            ret.update({
                "melspec_out":mel_generated,  # (bs, mellen, num_mels)
                "prosody": encoder_out["prosody"],
                "unit": encoder_out["unit"],
                "hu": encoder_out["hu"],
            })
        return ret
    
    def load_pretrained_avhubertmodel(self, pretrained_avhubert_path:str, map_location):
        avhubert_weight = torch.load(pretrained_avhubert_path, map_location=map_location)['model']
        #  label_embs_concat and final_proj will not be used in feature extraction.
        self.frontend_with_encoder.avhubert_model.load_state_dict(avhubert_weight)
        
    def load_pretrained_farlmodel(self, pretrained_farl_path:str, map_location):
        self.farl_model = self.farl_model.to(map_location)
        farl_state=torch.load(pretrained_farl_path) # you can download from https://github.com/FacePerceiver/FaRL#pre-trained-backbones
        self.farl_model.load_state_dict(farl_state["state_dict"],strict=False)
    
    def load_full_model_weight(self, state_dict, ignore_generator=False):
        state_dict = {k:v for k,v in state_dict.items() if not k.startswith('generator')}
        incompatiblekeys = self.load_state_dict(state_dict, strict=not ignore_generator)
        if ignore_generator:
            assert all(x.startswith('generator') for x in incompatiblekeys.missing_keys) and (not incompatiblekeys.unexpected_keys), f"When loading model, got {incompatiblekeys=}"

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def _get_varied_mask(wav_mask, loss_type, default_length):
    if wav_mask is not None:
        if loss_type == 'msd':
            loss_masks = msd_length_variators(wav_mask)
        elif loss_type == 'mpd':
            loss_masks = mpd_length_variators(wav_mask)
    else:
        loss_masks = [None] * default_length
    return loss_masks

def _patch_for_unaligned_mask(loss_mask, loss_input_len):
    loss_mask_len = loss_mask.shape[-1]
    diff = loss_input_len - loss_mask_len
    if diff > 0:
        # This will only happen when input audio finishes sooner than its video clip counterpart.
        padding = torch.zeros((*loss_mask.shape[:-1], diff), device=loss_mask.device).bool()
        loss_mask = torch.cat((loss_mask, padding), dim=-1)
    return loss_mask

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs, wav_mask=None, loss_type:str=None):
    loss = 0
    r_losses = []
    g_losses = []
    loss_masks = _get_varied_mask(wav_mask, loss_type, len(disc_generated_outputs))
    for dr, dg, loss_mask in zip(disc_real_outputs, disc_generated_outputs, loss_masks):
        r_loss_input = (1-dr)**2
        g_loss_input = dg**2
        if loss_mask is not None:
            loss_mask = _patch_for_unaligned_mask(loss_mask, loss_input_len=r_loss_input.shape[-1])
            r_loss_input = r_loss_input.masked_select(loss_mask)
            g_loss_input = g_loss_input.masked_select(loss_mask)
        r_loss = torch.mean(r_loss_input)
        g_loss = torch.mean(g_loss_input)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs, wav_mask=None, loss_type:str=None):
    loss = 0
    gen_losses = []
    loss_masks = _get_varied_mask(wav_mask, loss_type, len(disc_outputs))
    for dg, loss_mask in zip(disc_outputs, loss_masks):
        g_loss_input = (1-dg)**2
        if loss_mask is not None:
            loss_mask = _patch_for_unaligned_mask(loss_mask, loss_input_len=g_loss_input.shape[-1])
            g_loss_input = g_loss_input.masked_select(loss_mask)
        l = torch.mean(g_loss_input)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

