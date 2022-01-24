# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import click
import pickle
import re
import numpy as np
import torch
import stylegan2_pytorch.dnnlib as dnnlib
from stylegan2_pytorch.training.torch_utils import misc
from stylegan2_pytorch.training import networks

import collections

from os import path
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from io import BytesIO
import IPython.display

#----------------------------------------------------------------------------
def load_torch_generator(pkl_file_path='generator_kwargs.pkl', pth_file='generator.pth'):
    print('Loading generator\'s necessary kwargs...')
    with open(pkl_file_path, 'rb') as f:
        kwargs = pickle.load(f)
    print('Creating generator model...')
    G = networks.Generator(**kwargs).eval().requires_grad_(False)
    print('Loading generator\'s state dict...')
    G.load_state_dict(torch.load(pth_file))
    print('Done')
    return G

#----------------------------------------------------------------------------
def convert_tf_component_to_torch(name='G', component=None):
    
    comp = None
    if name == 'G':
        G = convert_tf_generator(component)
    
    return G

#----------------------------------------------------------------------------

def show_images(images, fmt='png'):
  for i in range(images.shape[0]):
    image = np.array(images[i])
    if image.dtype == np.float32:
        image = np.uint8(image * 127.5 + 127.5)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    plt.figure()
    plt.imshow(image)
    bytes_io = BytesIO()
    Image.fromarray(image).save(bytes_io, fmt)
    IPython.display.display(IPython.display.Image(data=bytes_io.getvalue()))
#----------------------------------------------------------------------------
def generate_sspace_per_index(G,dlat_path='saved_dlantents.pkl', num_layers=14):
    
    with open(dlat_path, 'rb') as f:
        dlatents_file = pickle.load(f)
    
    values_per_index = collections.defaultdict(list)
    for _, dlatent in dlatents_file:
        # Get the style vector: 
        dlatent = torch.Tensor(dlatent)
        expanded_dlatent_tmp = torch.tile(dlatent,[1, num_layers, 1])
        s_img = torch.cat(G.synthesis.style_vector_calculator(
            expanded_dlatent_tmp)[1], dim=1).numpy()[0]
        for i, s_val in enumerate(s_img):
            values_per_index[i].append(s_val)

    values_per_index = dict(values_per_index)
    s_indices_num = len(values_per_index.keys())
    minimums = [min(values_per_index[i]) for i in range(s_indices_num)] 
    maximums = [max(values_per_index[i]) for i in range(s_indices_num)] 

#----------------------------------------------------------------------------
def create_image_from_dlatent(G,dlat_path='saved_dlantents.pkl', num_layers=14):
    
    with open(dlat_path, 'rb') as f:
        dlatents_file = pickle.load(f)
    dlatents = []
    for dlat in dlatents_file:
        dlatents.append(dlat[1])
    dlatents = torch.Tensor(np.array(dlatents))
    expanded_dlatent_tmp = torch.tile(dlatents,[1, num_layers, 1])
    
    if expanded_dlatent_tmp is not None:
        style_vector_block_grouped, stl_vec_block, stl_vec_torgb = G.synthesis.style_vector_calculator(expanded_dlatent_tmp[:1,:,:])
        gen_output = G.synthesis.image_given_dlatent(expanded_dlatent_tmp[:1,:,:] ,style_vector_block_grouped)
        img_out = torch.maximum(torch.minimum(gen_output, torch.Tensor([1])), torch.Tensor([-1]))
        show_images(img_out)

    
#----------------------------------------------------------------------------
def convert_tfpkl_to_pytorch(source_path='saved_g_dictionary.pkl', dest_path='./generator.pth'):

    print(f'Loading "{source_path}"...')
    with open(source_path, 'rb') as f:
        tf_net = pickle.load(f)
    
    print(f'Converting network from Tensorflow to Pytorch...')
    torch_model = convert_tf_generator(tf_net)
    
    print(f'Saving "{dest_path}"...')
    torch.save(torch_model.state_dict(), dest_path)
    print(f'Done!')


#----------------------------------------------------------------------------

def _collect_tf_params(tf_net):
    # pylint: disable=protected-access
    tf_params = dict()
    def recurse(prefix, tf_net):
        for name, value in tf_net.variables:
            tf_params[prefix + name] = value
        for name, comp in tf_net.components.items():
            recurse(prefix + name + '/', comp)
    recurse('', tf_net)
    return tf_params

def _collect_tf_params_mod(tf_net):
    # pylint: disable=protected-access
    tf_params = dict()
    for param in tf_net:
        tf_params[param.name] = param
    return tf_params
#----------------------------------------------------------------------------

def _populate_module_params(module, *patterns):
    for name, tensor in misc.named_params_and_buffers(module):
        found = False
        value = None
        for pattern, value_fn in zip(patterns[0::2], patterns[1::2]):
            match = re.fullmatch(pattern, name)
            if match:
                found = True
                if value_fn is not None:
                    value = value_fn(*match.groups())
                break
        try:
            assert found
            if value is not None:
                print(name,tensor.shape)
                tensor.copy_(torch.from_numpy(np.array(value)))
        except:
            print(name, list(tensor.shape))
            raise

#----------------------------------------------------------------------------

def convert_tf_generator(tf_G):
   
    tf_kwargs = {}
    known_kwargs = set()
    def kwarg(tf_name, default=None, none=None):
        known_kwargs.add(tf_name)
        val = tf_kwargs.get(tf_name, default)
        return val if val is not None else none

    # Convert kwargs.
    kwargs = dnnlib.EasyDict(
        z_dim                   = kwarg('latent_size',          512), 
        c_dim                   = kwarg('label_size',           0),   #from 0 to 2
        w_dim                   = kwarg('dlatent_size',         512), #from 512 to 514
        img_resolution          = kwarg('resolution',           256), #from 1024 to 256
        img_channels            = kwarg('num_channels',         3),
        mapping_kwargs = dnnlib.EasyDict(
            num_layers          = kwarg('mapping_layers',       8),
            embed_features      = kwarg('label_fmaps',          None),
            layer_features      = kwarg('mapping_fmaps',        None),
            activation          = kwarg('mapping_nonlinearity', 'lrelu'),
            lr_multiplier       = kwarg('mapping_lrmul',        0.01),
            w_avg_beta          = kwarg('w_avg_beta',           0.995,  none=1),
        ),
        synthesis_kwargs = dnnlib.EasyDict(
            channel_base        = kwarg('fmap_base',            16384) * 2,
            channel_max         = kwarg('fmap_max',             512),
            num_fp16_res        = kwarg('num_fp16_res',         0),
            conv_clamp          = kwarg('conv_clamp',           None),
            architecture        = kwarg('architecture',         'skip'),
            resample_filter     = kwarg('resample_kernel',      [1,3,3,1]),
            use_noise           = kwarg('use_noise',            True),
            activation          = kwarg('nonlinearity',         'lrelu'),
        ),
    )

    # Check for unknown kwargs.
    kwarg('truncation_psi')
    kwarg('truncation_cutoff')
    kwarg('style_mixing_prob')
    kwarg('structure')
    unknown_kwargs = list(set(tf_kwargs.keys()) - known_kwargs)
    if len(unknown_kwargs) > 0:
        raise ValueError('Unknown TensorFlow kwarg', unknown_kwargs[0])

    # Collect params.
    tf_params = _collect_tf_params_mod(tf_G)
    for name, value in list(tf_params.items()):
        match = re.fullmatch(r'ToRGB_lod(\d+)/(.*)', name)
        if match:
            r = kwargs.img_resolution // (2 ** int(match.group(1)))
            tf_params[f'{r}x{r}/ToRGB/{match.group(2)}'] = value
            kwargs.synthesis.kwargs.architecture = 'orig'
    #for name, value in tf_params.items(): print(f'{name:<50s}{list(value.shape)}')

    # Save kwargs of the generator
    with open('generator_kwargs.pkl', 'wb') as f:
        pickle.dump(kwargs, f)

    # Convert params.
    G = networks.Generator(**kwargs).eval().requires_grad_(False)
    # pylint: disable=unnecessary-lambda
    # _populate_module_params(G,
    #     r'mapping\.w_avg',                                  lambda:     tf_params[f'dlatent_avg'],
    #     r'mapping\.embed\.weight',                          lambda:     tf_params[f'mapping/LabelEmbed/weight'].transpose(),
    #     r'mapping\.embed\.bias',                            lambda:     tf_params[f'mapping/LabelEmbed/bias'],
    #     r'mapping\.fc(\d+)\.weight',                        lambda i:   tf_params[f'mapping/Dense{i}/weight'].transpose(),
    #     r'mapping\.fc(\d+)\.bias',                          lambda i:   tf_params[f'mapping/Dense{i}/bias'],
    #     r'synthesis\.b4\.const',                            lambda:     tf_params[f'synthesis/4x4/Const/const'][0],
    #     r'synthesis\.b4\.conv1\.weight',                    lambda:     tf_params[f'synthesis/4x4/Conv/weight'].transpose(3, 2, 0, 1),
    #     r'synthesis\.b4\.conv1\.bias',                      lambda:     tf_params[f'synthesis/4x4/Conv/bias'],
    #     r'synthesis\.b4\.conv1\.noise_const',               lambda:     tf_params[f'synthesis/noise0'][0, 0],
    #     r'synthesis\.b4\.conv1\.noise_strength',            lambda:     tf_params[f'synthesis/4x4/Conv/noise_strength'],
    #     r'synthesis\.b4\.conv1\.affine\.weight',            lambda:     tf_params[f'synthesis/4x4/Conv/mod_weight'].transpose(),
    #     r'synthesis\.b4\.conv1\.affine\.bias',              lambda:     tf_params[f'synthesis/4x4/Conv/mod_bias'] + 1,
    #     r'synthesis\.b(\d+)\.conv0\.weight',                lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/weight'][::-1, ::-1].transpose(3, 2, 0, 1),
    #     r'synthesis\.b(\d+)\.conv0\.bias',                  lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/bias'],
    #     r'synthesis\.b(\d+)\.conv0\.noise_const',           lambda r:   tf_params[f'synthesis/noise{int(np.log2(int(r)))*2-5}'][0, 0],
    #     r'synthesis\.b(\d+)\.conv0\.noise_strength',        lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/noise_strength'],
    #     r'synthesis\.b(\d+)\.conv0\.affine\.weight',        lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/mod_weight'].transpose(),
    #     r'synthesis\.b(\d+)\.conv0\.affine\.bias',          lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/mod_bias'] + 1,
    #     r'synthesis\.b(\d+)\.conv1\.weight',                lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/weight'].transpose(3, 2, 0, 1),
    #     r'synthesis\.b(\d+)\.conv1\.bias',                  lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/bias'],
    #     r'synthesis\.b(\d+)\.conv1\.noise_const',           lambda r:   tf_params[f'synthesis/noise{int(np.log2(int(r)))*2-4}'][0, 0],
    #     r'synthesis\.b(\d+)\.conv1\.noise_strength',        lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/noise_strength'],
    #     r'synthesis\.b(\d+)\.conv1\.affine\.weight',        lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/mod_weight'].transpose(),
    #     r'synthesis\.b(\d+)\.conv1\.affine\.bias',          lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/mod_bias'] + 1,
    #     r'synthesis\.b(\d+)\.torgb\.weight',                lambda r:   tf_params[f'synthesis/{r}x{r}/ToRGB/weight'].transpose(3, 2, 0, 1),
    #     r'synthesis\.b(\d+)\.torgb\.bias',                  lambda r:   tf_params[f'synthesis/{r}x{r}/ToRGB/bias'],
    #     r'synthesis\.b(\d+)\.torgb\.affine\.weight',        lambda r:   tf_params[f'synthesis/{r}x{r}/ToRGB/mod_weight'].transpose(),
    #     r'synthesis\.b(\d+)\.torgb\.affine\.bias',          lambda r:   tf_params[f'synthesis/{r}x{r}/ToRGB/mod_bias'] + 1,
    #     r'synthesis\.b(\d+)\.skip\.weight',                 lambda r:   tf_params[f'synthesis/{r}x{r}/Skip/weight'][::-1, ::-1].transpose(3, 2, 0, 1),
    #     r'.*\.resample_filter',                             None,
    # )
    
    _populate_module_params(G,
        r'mapping\.w_avg',                                  lambda:     np.array(tf_params[f'dlatent_avg:0']), #done
        r'mapping\.embed\.weight',                          lambda:     np.array(tf_params[f'mapping/LabelEmbed/weight']).transpose(), #?
        r'mapping\.embed\.bias',                            lambda:     np.array(tf_params[f'mapping/LabelEmbed/bias']), #?
        r'mapping\.fc(\d+)\.weight',                        lambda i:   np.array(tf_params[f'generator/generator_mapping/Dense{i}/MappingDense_{i}:0']).transpose(), #done
        r'mapping\.fc(\d+)\.bias',                          lambda i:   np.array(tf_params[f'generator/generator_mapping/DenseBiasAct{i}/MappingBias_{i}:0']).squeeze(), #done
        r'synthesis\.b4\.const',                            lambda:     np.array(tf_params[f'generator_synthesis/ConstBlock4x4/Const4x4:0'])[0], #done
        r'synthesis\.b4\.conv1\.weight',                    lambda:     np.array(tf_params[f'generator_synthesis/ConstBlock4x4/GeneratorLayer4x4/ModulatedConv2D/weight:0']).transpose(3, 2, 0, 1), #done
        r'synthesis\.b4\.conv1\.bias',                      lambda:     np.array(tf_params[f'generator_synthesis/ConstBlock4x4/GeneratorLayer4x4/FusedBiasAct/BiasAct:0']).squeeze(), #done
        r'synthesis\.b4\.conv1\.noise_const',               lambda:     np.array(tf_params[f'Noise0:0'])[0, 0], #done
        r'synthesis\.b4\.conv1\.noise_strength',            lambda:     np.array(tf_params[f'generator_synthesis/ConstBlock4x4/GeneratorLayer4x4/NoiseStrength:0']).squeeze(), #done
        r'synthesis\.b4\.conv1\.affine\.weight',            lambda:     np.array(tf_params[f'generator_style_calculator/StyleVectorConst/StyleGAN2Dense/weight:0']).transpose(), #done
        r'synthesis\.b4\.conv1\.affine\.bias',              lambda:     np.array(tf_params[f'generator_style_calculator/StyleVectorConst/FusedBiasActivation/bias:0']).squeeze() + 1, #done
        r'synthesis\.b4\.torgb\.weight',                    lambda:     np.array(tf_params[f'generator_synthesis/ToRGB4x4/ModulatedConv2D/weight:0']).transpose(3, 2, 0, 1), #done
        r'synthesis\.b4\.torgb\.bias',                      lambda:     np.array(tf_params[f'generator_synthesis/ToRGB4x4/FusedBiasActivation/ToRGBBiasAct:0']).squeeze(), #done
        r'synthesis\.b4\.torgb\.affine\.weight',            lambda:     np.array(tf_params[f'generator_style_calculator/StyleVectorConstRGB/StyleGAN2Dense/weight:0']).transpose(), #done
        r'synthesis\.b4\.torgb\.affine\.bias',              lambda:     np.array(tf_params[f'generator_style_calculator/StyleVectorConstRGB/FusedBiasActivation/bias:0']).squeeze() + 1, #done
        r'synthesis\.b(\d+)\.conv0\.weight',                lambda r:   np.array(tf_params[f'generator_synthesis/GeneratorBlock{r}x{r}/Conv0Up/ModulatedConv2D/weight:0'])[::-1, ::-1].transpose(3, 2, 0, 1), #done
        r'synthesis\.b(\d+)\.conv0\.bias',                  lambda r:   np.array(tf_params[f'generator_synthesis/GeneratorBlock{r}x{r}/Conv0Up/FusedBiasAct/BiasAct:0']).squeeze(), #done
        r'synthesis\.b(\d+)\.conv0\.noise_const',           lambda r:   np.array(tf_params[f'Noise{int(np.log2(int(r)))*2-5}:0'])[0, 0], #done
        r'synthesis\.b(\d+)\.conv0\.noise_strength',        lambda r:   np.array(tf_params[f'generator_synthesis/GeneratorBlock{r}x{r}/Conv0Up/NoiseStrength:0']).squeeze(), #done
        r'synthesis\.b(\d+)\.conv0\.affine\.weight',        lambda r:   np.array(tf_params[f'generator_style_calculator/StyleVector{int(np.log2(int(r)))-3}_0/StyleGAN2Dense/weight:0']).transpose(), #done
        r'synthesis\.b(\d+)\.conv0\.affine\.bias',          lambda r:   np.array(tf_params[f'generator_style_calculator/StyleVector{int(np.log2(int(r)))-3}_0/FusedBiasActivation/bias:0']).squeeze() + 1, #done
        r'synthesis\.b(\d+)\.conv1\.weight',                lambda r:   np.array(tf_params[f'generator_synthesis/GeneratorBlock{r}x{r}/Conv1/ModulatedConv2D/weight:0']).transpose(3, 2, 0, 1), #done
        r'synthesis\.b(\d+)\.conv1\.bias',                  lambda r:   np.array(tf_params[f'generator_synthesis/GeneratorBlock{r}x{r}/Conv1/FusedBiasAct/BiasAct:0']).squeeze(), #done
        r'synthesis\.b(\d+)\.conv1\.noise_const',           lambda r:   np.array(tf_params[f'Noise{int(np.log2(int(r)))*2-4}:0'])[0, 0], #done
        r'synthesis\.b(\d+)\.conv1\.noise_strength',        lambda r:   np.array(tf_params[f'generator_synthesis/GeneratorBlock{r}x{r}/Conv1/NoiseStrength:0']).squeeze(), #done
        r'synthesis\.b(\d+)\.conv1\.affine\.weight',        lambda r:   np.array(tf_params[f'generator_style_calculator/StyleVector{int(np.log2(int(r)))-3}_1/StyleGAN2Dense/weight:0']).transpose(), #done
        r'synthesis\.b(\d+)\.conv1\.affine\.bias',          lambda r:   np.array(tf_params[f'generator_style_calculator/StyleVector{int(np.log2(int(r)))-3}_1/FusedBiasActivation/bias:0']).squeeze() + 1, #done
        r'synthesis\.b(\d+)\.torgb\.weight',                lambda r:   np.array(tf_params[f'generator_synthesis/ToRGB{r}x{r}/ModulatedConv2D/weight:0']).transpose(3, 2, 0, 1), #done
        r'synthesis\.b(\d+)\.torgb\.bias',                  lambda r:   np.array(tf_params[f'generator_synthesis/ToRGB{r}x{r}/FusedBiasActivation/ToRGBBiasAct:0']).squeeze(), #done
        r'synthesis\.b(\d+)\.torgb\.affine\.weight',        lambda r:   np.array(tf_params[f'generator_style_calculator/StyleVectorRGB{int(np.log2(int(r)))-3}/StyleGAN2Dense/weight:0']).transpose(), #done
        r'synthesis\.b(\d+)\.torgb\.affine\.bias',          lambda r:   np.array(tf_params[f'generator_style_calculator/StyleVectorRGB{int(np.log2(int(r)))-3}/FusedBiasActivation/bias:0']).squeeze() + 1, #done
        r'synthesis\.b(\d+)\.skip\.weight',                 lambda r:   np.array(tf_params[f'synthesis/{r}x{r}/Skip/weight'])[::-1, ::-1].transpose(3, 2, 0, 1),
        r'.*\.resample_filter',                             None,
    )
    return G

#----------------------------------------------------------------------------

def convert_tf_discriminator(tf_D):
    if tf_D.version < 4:
        raise ValueError('TensorFlow pickle version too low')

    # Collect kwargs.
    tf_kwargs = tf_D.static_kwargs
    known_kwargs = set()
    def kwarg(tf_name, default=None):
        known_kwargs.add(tf_name)
        return tf_kwargs.get(tf_name, default)

    # Convert kwargs.
    kwargs = dnnlib.EasyDict(
        c_dim                   = kwarg('label_size',           2), #from 0 to 2
        img_resolution          = kwarg('resolution',           256), #from 1024 to 256
        img_channels            = kwarg('num_channels',         3),
        architecture            = kwarg('architecture',         'resnet'),
        channel_base            = kwarg('fmap_base',            16384) * 2,
        channel_max             = kwarg('fmap_max',             512),
        num_fp16_res            = kwarg('num_fp16_res',         0),
        conv_clamp              = kwarg('conv_clamp',           None),
        cmap_dim                = kwarg('mapping_fmaps',        None),
        block_kwargs = dnnlib.EasyDict(
            activation          = kwarg('nonlinearity',         'lrelu'),
            resample_filter     = kwarg('resample_kernel',      [1,3,3,1]),
            freeze_layers       = kwarg('freeze_layers',        0),
        ),
        mapping_kwargs = dnnlib.EasyDict(
            num_layers          = kwarg('mapping_layers',       0),
            embed_features      = kwarg('mapping_fmaps',        None),
            layer_features      = kwarg('mapping_fmaps',        None),
            activation          = kwarg('nonlinearity',         'lrelu'),
            lr_multiplier       = kwarg('mapping_lrmul',        0.1),
        ),
        epilogue_kwargs = dnnlib.EasyDict(
            mbstd_group_size    = kwarg('mbstd_group_size',     None),
            mbstd_num_channels  = kwarg('mbstd_num_features',   1),
            activation          = kwarg('nonlinearity',         'lrelu'),
        ),
    )

    # Check for unknown kwargs.
    kwarg('structure')
    unknown_kwargs = list(set(tf_kwargs.keys()) - known_kwargs)
    if len(unknown_kwargs) > 0:
        raise ValueError('Unknown TensorFlow kwarg', unknown_kwargs[0])

    # Collect params.
    tf_params = _collect_tf_params(tf_D)
    for name, value in list(tf_params.items()):
        match = re.fullmatch(r'FromRGB_lod(\d+)/(.*)', name)
        if match:
            r = kwargs.img_resolution // (2 ** int(match.group(1)))
            tf_params[f'{r}x{r}/FromRGB/{match.group(2)}'] = value
            kwargs.architecture = 'orig'
    #for name, value in tf_params.items(): print(f'{name:<50s}{list(value.shape)}')

    # Convert params.
    D = networks.Discriminator(**kwargs).eval().requires_grad_(False)
    # pylint: disable=unnecessary-lambda
    _populate_module_params(D,
        r'b(\d+)\.fromrgb\.weight',     lambda r:       tf_params[f'{r}x{r}/FromRGB/weight'].transpose(3, 2, 0, 1),
        r'b(\d+)\.fromrgb\.bias',       lambda r:       tf_params[f'{r}x{r}/FromRGB/bias'],
        r'b(\d+)\.conv(\d+)\.weight',   lambda r, i:    tf_params[f'{r}x{r}/Conv{i}{["","_down"][int(i)]}/weight'].transpose(3, 2, 0, 1),
        r'b(\d+)\.conv(\d+)\.bias',     lambda r, i:    tf_params[f'{r}x{r}/Conv{i}{["","_down"][int(i)]}/bias'],
        r'b(\d+)\.skip\.weight',        lambda r:       tf_params[f'{r}x{r}/Skip/weight'].transpose(3, 2, 0, 1),
        r'mapping\.embed\.weight',      lambda:         tf_params[f'LabelEmbed/weight'].transpose(),
        r'mapping\.embed\.bias',        lambda:         tf_params[f'LabelEmbed/bias'],
        r'mapping\.fc(\d+)\.weight',    lambda i:       tf_params[f'Mapping{i}/weight'].transpose(),
        r'mapping\.fc(\d+)\.bias',      lambda i:       tf_params[f'Mapping{i}/bias'],
        r'b4\.conv\.weight',            lambda:         tf_params[f'4x4/Conv/weight'].transpose(3, 2, 0, 1),
        r'b4\.conv\.bias',              lambda:         tf_params[f'4x4/Conv/bias'],
        r'b4\.fc\.weight',              lambda:         tf_params[f'4x4/Dense0/weight'].transpose(),
        r'b4\.fc\.bias',                lambda:         tf_params[f'4x4/Dense0/bias'],
        r'b4\.out\.weight',             lambda:         tf_params[f'Output/weight'].transpose(),
        r'b4\.out\.bias',               lambda:         tf_params[f'Output/bias'],
        r'.*\.resample_filter',         None,
    )
    return D

#----------------------------------------------------------------------------

@click.command()
@click.option('--source', help='Input pickle', default='saved_g_dictionary.pkl', required=True, metavar='PATH')
@click.option('--dest', help='Output pickle', default='generator.pth', required=True, metavar='PATH')
@click.option('--force-fp16', help='Force the networks to use FP16', type=bool, default=False, metavar='BOOL', show_default=True)
def convert_network_pickle(source, dest, force_fp16):
    """Convert legacy network pickle into the native PyTorch format.

    The tool is able to load the main network configurations exported using the TensorFlow version of StyleGAN2 or StyleGAN2-ADA.
    It does not support e.g. StyleGAN2-ADA comparison methods, StyleGAN2 configs A-D, or StyleGAN1 networks.

    Example:

    \b
    python legacy.py \\
        --source=https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkl \\
        --dest=stylegan2-cat-config-f.pkl
    """
    if not path.exists(dest):
        G = convert_tfpkl_to_pytorch(source,dest)
        
    G = load_torch_generator()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_network_pickle() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------