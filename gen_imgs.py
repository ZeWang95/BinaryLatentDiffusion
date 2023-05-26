import torch
import numpy as np
import copy
import os
from models.binaryae import  Generator
from hparams import get_sampler_hparams
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts,\
     get_sampler, get_samples_test
from utils.train_utils import EMA
from utils.log_utils import load_model, save_results




def main(H):

    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )

    embedding_weight = quanitzer_and_generator_state_dict.pop(
        'embed.weight')
    if H.deepspeed:
        embedding_weight = embedding_weight.half()
    embedding_weight = embedding_weight.cuda()

    generator = Generator(H)
    generator.load_state_dict(quanitzer_and_generator_state_dict, strict=False)
    generator = generator.cuda()
    
    del quanitzer_and_generator_state_dict
    
    sampler = get_sampler(H, embedding_weight).cuda()

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler)


    if H.load_step == -1:
        fs = os.listdir(os.path.join(H.load_dir, 'saved_models'))
        fs = [f for f in fs if f.startswith('bld_ema')]
        fs = [int(f.split('.')[0].split('_')[-1]) for f in fs]
        load_step = np.max(fs)
        print('Overriding loadstep with %d' %load_step)
        H.load_step = load_step


    if H.load_step > 0:
        device = torch.device("cuda:0")

        ema_sampler = load_model(
                ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir, device=device)

    print(f'sampling with temp {H.temp}')
    images = get_samples_test(H, generator, ema_sampler if H.ema else sampler, t=H.temp, n_samples=H.batch_size)
    save_results(images, 'results', H.load_step, H.log_dir, H.temp, False)

if __name__ == '__main__':
    H = get_sampler_hparams()
    main(H)
