from dalle2_laion import DalleModelManager, ModelLoadConfig, utils
from dalle2_laion.scripts import BasicInference, ImageVariation, BasicInpainting
from typing import List
import os
import click
from pathlib import Path
import json
import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, prior):
        super().__init__()
        # self.prior = prior
        self.net = prior

    def forward(self, image_embed, timesteps,  text_emb, text_encodings):
        '''
        def forward(
        self,
        image_embed, # shape: [batch_size, 768]
        diffusion_timesteps, # shape: [batch_size], type: long
        *,
        text_embed, # shape: [batch_size, 768]
        text_encodings = None, # shape: [batch_size, 768]
        cond_drop_prob = 0.
        )
        '''
        res = self.net.forward_simple(image_embed, timesteps, text_emb,
                text_encodings=text_encodings, cond_drop_prob=0.)
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
def export_prior(ctx, model_config: str, output_path: str, decoder_batch_size: int):
    verbose = ctx.obj['verbose']
    prompts = ['a cat on the table']
    num_prior_samples = 5
    dreamer: BasicInference = BasicInference.create(model_config, verbose=verbose, check_updates=not ctx.obj['suppress_updates'])
    prior = dreamer.model_manager.prior_info.model
    net = prior.net
    max_seq_len = net.max_text_len
    model = ImageEncoder(net).to('cuda')
    model_path='img_encoder.onnx'
    h_dim = 768

    # def forward(self, image_embed, timesteps,  text_emb, text_encodings):
    img_emb = torch.randn([1, h_dim]).cuda()
    timesteps = torch.randint(1000, [1]).cuda()
    text_emb = torch.randn([1, h_dim]).cuda()
    text_encodings = torch.randn([1, max_seq_len, h_dim]).cuda()
    torch.onnx.export(model,
                      (img_emb, timesteps, text_emb, text_encodings),
                      model_path,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      verbose=True,
                      input_names=['image_embed', 'timesteps', 'text_emb',
                          'text_encodings'],
                      output_names=['out_img_emb'],
                      dynamic_axes = {
                          'image_embed': {0: 'batch_size'},
                          'timesteps': {0: 'batch_size'},
                          'text_emb': {0: 'batch_size'},
                          'text_encodings': {0: 'batch_size'},
                          'out_img_emb': {0: 'batch_size'}
                        }
                    )


if __name__ == "__main__":
    inference(obj={})
