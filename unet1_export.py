from dalle2_laion import DalleModelManager, ModelLoadConfig, utils
from dalle2_laion.scripts import BasicInference, ImageVariation, BasicInpainting
from typing import List
import os
import click
from pathlib import Path
import json
import torch
from torch import nn


class Unet1(nn.Module):
    def __init__(self, prior):
        super().__init__()
        # self.prior = prior
        self.net = prior

    def forward(self, img, time_cond, img_emb, text_encodings, cond_scale,
            l_img=None):
        '''

        pred = unet.forward_with_cond_scale(img, time_cond, image_embed = img_emb, text_encodings = text_encodings, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_level = lowres_noise_level)
        '''
        res = self.net.forward_with_cond_scale(
                img, time_cond,
                image_embed=img_emb,
                text_encodings = text_encodings,
                cond_scale = cond_scale,
                lowres_cond_img =l_img,
                lowres_noise_level = None)
        return res


@click.group()
@click.option('--verbose', '-v', is_flag=True, default=False, help='Print verbose output.')
@click.option('--suppress-updates', '-s', is_flag=True, default=False, help='Suppress updating models if checksums do not match.')
@click.pass_context
def inference(ctx, verbose, suppress_updates):
    ctx.obj['verbose'] = verbose
    ctx.obj['suppress_updates'] = suppress_updates


@inference.command()
@click.option('--model-config', default='./configs/upsampler.example.json', help='Path to model config file')
@click.option('--output-path', default='./output/basic/', help='Path to output directory')
@click.option('--decoder-batch-size', default=10, help='Batch size for decoder')
@click.pass_context
def export_unet1(ctx, model_config: str, output_path: str, decoder_batch_size: int):
    verbose = ctx.obj['verbose']
    prompts = ['a cat on the table']
    num_prior_samples = 5
    dreamer: BasicInference = BasicInference.create(model_config, verbose=verbose, check_updates=not ctx.obj['suppress_updates'])
    load_config = dreamer.model_manager.model_config.decoder
    unet_configs = load_config.unet_sources

    decoder = dreamer.model_manager.decoder_info.model.unets[0]
    max_seq_len = 77
    model = Unet1(decoder).to('cuda')
    model_path='unet1.onnx'
    img_size = 64
    h_dim = 768

    # forward(self, img, img_cond, img_emb, text_encodings, cond_scale)
    img = torch.randn([1, 3, img_size, img_size]).cuda()
    img_cond = torch.randint(1000, [1]).cuda()
    img_emb = torch.randn([1, h_dim]).cuda()
    text_encodings = torch.randn([1, max_seq_len, h_dim]).cuda()
    cond_scale = 1.7
    torch.onnx.export(model,
                      (img, img_cond, img_emb, text_encodings, cond_scale),
                      model_path,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      verbose=True,
                      use_external_data_format=True,
                      input_names=['img', 'img_cond', 'img_emb',
                          'text_encodings', 'cond_scale'],
                      output_names=['out_img'],
                      dynamic_axes = {
                          'img': {0: 'batch_size', 2: 'H', 3: 'W'},
                          'img_cond': {0: 'batch_size'},
                          'img_emb': {0: 'batch_size'},
                          'text_encodings': {0: 'batch_size'},
                          'out_img': {0: 'batch_size', 2: 'H', 3: 'W'},
                        }
                    )


@inference.command()
@click.option('--model-config', default='./configs/upsampler.example.json', help='Path to model config file')
@click.option('--output-path', default='./output/basic/', help='Path to output directory')
@click.option('--decoder-batch-size', default=10, help='Batch size for decoder')
@click.pass_context
def export_unet2(ctx, model_config: str, output_path: str, decoder_batch_size: int):
    verbose = ctx.obj['verbose']
    prompts = ['a cat on the table']
    num_prior_samples = 5
    dreamer: BasicInference = BasicInference.create(model_config, verbose=verbose, check_updates=not ctx.obj['suppress_updates'])
    decoder = dreamer.model_manager.decoder_info.model.unets[1]
    max_seq_len = 77
    model = Unet1(decoder).to('cuda')
    model_path='unet2/unet2.onnx'
    img_size = 256
    h_dim = 768

    # forward(self, img, img_cond, img_emb, text_encodings, cond_scale)
    img = torch.randn([1, 3, img_size, img_size]).cuda()
    img_cond = torch.randint(1000, [1]).cuda()
    img_emb = torch.randn([1, h_dim]).cuda()
    text_encodings = torch.randn([1, max_seq_len, h_dim]).cuda()
    cond_scale = 1.0
    lowres_cond_img = torch.randn([1, 3, img_size, img_size]).cuda()

    torch.onnx.export(model,
                      (img, img_cond, img_emb, text_encodings, cond_scale,
                       lowres_cond_img),
                      model_path,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      verbose=True,
                      use_external_data_format=True,
                      input_names=['img', 'img_cond', 'img_emb',
                          'text_encodings', 'cond_scale', 'lowres_cond_img'],
                      output_names=['out_img'],
                      dynamic_axes = {
                          'img': {0: 'batch_size', 2: 'H', 3: 'W'},
                          'img_cond': {0: 'batch_size'},
                          'img_emb': {0: 'batch_size'},
                          'text_encodings': {0: 'batch_size'},
                          'lowres_cond_img': {0: 'batch_size', 2: 'H', 3: 'W'},
                          'out_img': {0: 'batch_size', 2: 'H', 3: 'W'},
                        }
                    )






if __name__ == "__main__":
    inference(obj={})
