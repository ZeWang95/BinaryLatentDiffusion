import torch
import numpy as np
import copy
import os
from models.binaryae import BinaryAutoEncoder
from hparams import get_sampler_hparams
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, get_sampler, get_t2i_samples_guidance_test
from utils.train_utils import EMA
from utils.log_utils import load_model, save_results_t2i
import misc
from transformers import T5Tokenizer, T5EncoderModel

import clip
import pdb

def main(H):
    misc.init_distributed_mode(H)

    ae_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['encoder', 'quantize', 'generator'],
        remove_component_from_key=False
    )

    binaryae = BinaryAutoEncoder(H)
    binaryae.load_state_dict(ae_state_dict, strict=True)
    binaryae = binaryae.cuda()

    if H.cross:
        T5tokenizer = T5Tokenizer.from_pretrained("t5-large", cache_dir=os.path.join(H.root_path, 'huggingface'))
        T5model = T5EncoderModel.from_pretrained("t5-large", cache_dir=os.path.join(H.root_path, 'huggingface')).cuda()
        H.text_emb = 1024
        max_length = 128

        clip_model, _ = clip.load(os.path.join(H.root_path, 'clip_models', 'ViT-L-14.pt'))
        clip_model = clip_model.cuda()
    else:
        clip_model, _ = clip.load(os.path.join(H.root_path, 'clip_models', 'ViT-L-14.pt'))
        clip_model = clip_model.cuda()

    
    sampler = get_sampler(H, binaryae.quantize.embed.weight).cuda()

    if H.ema:
        ema_sampler = copy.deepcopy(sampler)


    if H.distributed:
        sampler = torch.nn.parallel.DistributedDataParallel(sampler, device_ids=[H.gpu], find_unused_parameters=True)

    if H.load_step == -1:
        fs = os.listdir(os.path.join(H.load_dir, 'saved_models'))
        fs = [f for f in fs if f.startswith(f'{H.sampler}_ema')]
        fs = [int(f.split('.')[0].split('_')[-1]) for f in fs]
        load_step = np.max(fs)
        print('Overriding loadstep with %d' %load_step)
        H.load_step = load_step

    if H.load_step > 0:
        device = sampler.device

        sampler = None
        if H.ema:
            ema_sampler = load_model(
                ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir, device=device)

    
    if H.omega == 0.0:
        guidance = None
    else:
        guidance = H.omega

    bs = H.batch_size

    i=0
    while True:
        try:
            text = str(input('Enter your prompt: '))
            if H.cross:
                t5_tkn = T5tokenizer(text, max_length=max_length, padding='longest', truncation=True, return_tensors="pt").input_ids.cuda()
                label = T5model(t5_tkn).last_hidden_state.detach()#.float()

                clip_ckn = clip.tokenize(text).cuda()
                labelc = clip_model.encode_text(clip_ckn).detach().float()

                label = label.repeat(bs, 1, 1)
                labelc = labelc.repeat(bs, 1)
                label = [label, labelc.unsqueeze(1)]

            else:
                t = clip.tokenize(text).cuda()
                label = clip_model.encode_text(t).detach().float()
                label = label.repeat(bs, 1)

            images = get_t2i_samples_guidance_test(H, binaryae, ema_sampler if H.ema else sampler, label=label, g=guidance, t=H.temp)
            save_results_t2i(images, i, H.log_dir, H.temp, H.sample_steps, guidance, '_'.join(text.split(' ')[:10]), save_individually=False)
        except:
            print('skipping')
            continue

if __name__ == '__main__':

    H = get_sampler_hparams()
    print('---------------------------------')
    print(f'Setting up training for {H.sampler}')
    main(H)
