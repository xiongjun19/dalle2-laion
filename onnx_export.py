from dalle2_laion import DalleModelManager, ModelLoadConfig, utils
from dalle2_laion.scripts import BasicInference, ImageVariation, BasicInpainting
from typing import List
import os
import click
from pathlib import Path
import json
import torch
from torch import nn


class ClipTextEmb(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, tokens):
        '''
        text_tokens = self._tokenize_text(text)
        with self._clip_in_gpu() as clip:
            text_embed = clip.embed_text(text_tokens.to(self.device))
        return text_embed.text_encodings
        '''
        # import ipdb; ipdb.set_trace()
        text_embed = self.clip.encode_text(tokens)
        # text_embed = self.clip.embed_text(tokens)
        # text_encodings = text_embed.text_encodings
        return text_embed


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
def dream(ctx, model_config: str, output_path: str, decoder_batch_size: int):
    verbose = ctx.obj['verbose']
    prompts = []
    print("Enter your prompts one by one. Enter an empty prompt to finish.")
    while True:
        prompt = click.prompt(f'Prompt {len(prompts)+1}', default='', type=str, show_default=False)
        if prompt == '':
            break
        prompt_file = Path(prompt)
        if utils.is_text_file(prompt_file):
            # Then we can read the prompts line by line
            with open(prompt_file, 'r') as f:
                for line in f:
                    prompts.append(line.strip())
        elif utils.is_json_file(prompt_file):
            # Then we assume this is an array of prompts
            with open(prompt_file, 'r') as f:
                prompts.extend(json.load(f))
        else:
            prompts.append(prompt)
    num_prior_samples = click.prompt('How many samples would you like to generate for each prompt?', default=1, type=int)


    dreamer: BasicInference = BasicInference.create(model_config, verbose=verbose, check_updates=not ctx.obj['suppress_updates'])
    # import ipdb; ipdb.set_trace()
    output_map = dreamer.run(prompts, prior_sample_count=num_prior_samples, decoder_batch_size=decoder_batch_size)
    os.makedirs(output_path, exist_ok=True)
    for text in output_map:
        for embedding_index in output_map[text]:
            for image in output_map[text][embedding_index]:
                image.save(os.path.join(output_path, f"{text}_{embedding_index}.png"))



@inference.command()
@click.option('--model-config', default='./configs/upsampler.example.json', help='Path to model config file')
@click.option('--output-path', default='./output/basic/', help='Path to output directory')
@click.option('--decoder-batch-size', default=10, help='Batch size for decoder')
@click.pass_context
def export_clip_text(ctx, model_config: str, output_path: str, decoder_batch_size: int):
    verbose = ctx.obj['verbose']
    prompts = ['a cat on the table']
    num_prior_samples = 5
    max_seq_len = 77

    dreamer: BasicInference = BasicInference.create(model_config, verbose=verbose, check_updates=not ctx.obj['suppress_updates'])
    assert dreamer.model_manager.clip is not None, "Cannot generate embeddings for this model."
    model = ClipTextEmb(dreamer.model_manager.clip.clip)
    model_path='clip_text_encoding.onnx'
    text_tokens = dreamer._tokenize_text(prompts).cuda()
    text_tokens = text_tokens[:, :max_seq_len]
    torch.onnx.export(model,
                      text_tokens,
                      model_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['tokens'],
                      output_names=['text_encodings'],
                      dynamic_axes = {
                          'tokens': {1: 'seq_len'},
                          # 'text_encodings': {1: 'seq_len'}
                        }
                    )


if __name__ == "__main__":
    inference(obj={})
