import torch
import numpy as np
import copy
import time
import os
import pdb

from tqdm import tqdm
from models.binaryae import BinaryAutoEncoder, Generator
from hparams import get_sampler_hparams
from utils.data_utils import get_data_loaders
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts,\
    get_sampler, get_online_samples, get_online_samples_guidance
from utils.train_utils import EMA, NativeScalerWithGradNormCount
from utils.log_utils import log, log_stats, config_log, start_training_log, \
    save_stats, load_stats, save_model, load_model, save_images, \
    MovingAverage
import misc
import torch.distributed as dist
from utils.lr_sched import adjust_lr, lr_scheduler

def main(H, vis):

    misc.init_distributed_mode(H)

    ae_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['encoder', 'quantize', 'generator'],
        remove_component_from_key=False
    )

    
    bergan = BinaryAutoEncoder(H)
    bergan.load_state_dict(ae_state_dict, strict=True)
    bergan = bergan.cuda()
    del ae_state_dict

    sampler = get_sampler(H, bergan.quantize.embed.weight).cuda()

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler)

    if H.distributed:
        find_unused = H.guidance
        sampler = torch.nn.parallel.DistributedDataParallel(sampler, device_ids=[H.gpu], find_unused_parameters=find_unused)
        sampler_without_ddp = sampler.module

    optim_eps = H.optim_eps
    optim = torch.optim.AdamW(sampler_without_ddp.parameters(), lr=H.lr, weight_decay=H.weight_decay, betas=(0.9, 0.95), eps=optim_eps)

    losses = np.array([])
    val_losses = np.array([])
    elbo = np.array([])
    val_elbos = np.array([])
    mean_losses = np.array([])
    start_step = 0
    log_start_step = 0

    loss_ma = MovingAverage(100)

    if H.load_model_step > 0:
        device = sampler.device
        sampler = load_model(sampler, H.sampler, H.load_model_step, H.load_model_dir, device=device).cuda()

        
    scaler = NativeScalerWithGradNormCount(H.amp, H.init_scale)

    if H.load_step > 0:
        start_step = H.load_step + 1

        device = sampler.device

        allow_mismatch = H.allow_mismatch
        sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch).cuda()
        if H.ema:
            # if EMA has not been generated previously, recopy newly loaded model
            try:
                ema_sampler = load_model(
                    ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch)
            except Exception:
                ema_sampler = copy.deepcopy(sampler_without_ddp)
        
        if not allow_mismatch:
            if H.load_optim:
                optim = load_model(
                    optim, f'{H.sampler}_optim', H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch)
                for param_group in optim.param_groups:
                    param_group['lr'] = H.lr
        try:
            train_stats = load_stats(H, H.load_step)
        except Exception:
            train_stats = None

        if not H.reset_step:
            if not H.reset_scaler:
                try:
                    scaler.load_state_dict(torch.load(os.path.join(H.load_dir, 'saved_models', f'absorbingbnl_scaler_{H.load_step}.th')))
                except Exception:
                    print('Failing to load scaler.')
        else:
            H.load_step = 0

        
        if train_stats is not None:
            losses, mean_losses, val_losses, elbo, H.steps_per_log

            losses = train_stats["losses"],
            mean_losses = train_stats["mean_losses"],
            val_losses = train_stats["val_losses"],
            val_elbos = train_stats["val_elbos"]
            log_start_step = 0

            losses = losses[0]
            mean_losses = mean_losses[0]
            val_losses = val_losses[0]
            val_elbos = torch.Tensor([0])

        else:
            log('No stats file found for loaded model, displaying stats from load step only.')
            log_start_step = start_step

        if H.reset_step:
            start_step = 0
    
    train_loader, val_loader = get_data_loaders(
        H.dataset,
        H.img_size,
        H.batch_size,
        get_val_dataloader=False,
        custom_dataset_path=H.path_to_data,
        num_workers=4,
        distributed=H.distributed,
        random=True,
        args=H, 
    )

    log(f"Sampler params total: {(sum(p.numel() for p in sampler.parameters())/1e6)}M")

    # for step in range(start_step, H.train_steps):
    H.train_steps = H.train_steps * H.update_freq
    H.warmup_iters = H.warmup_iters * H.update_freq
    H.steps_per_log = H.steps_per_log * H.update_freq
    lr_sched = lr_scheduler(base_value=H.lr, final_value=1e-6, iters=H.train_steps+1, warmup_steps=H.warmup_iters,
                     start_warmup_value=1e-6, lr_type='constant')
    print(lr_sched)
    step = start_step - 1
    epoch = -1

    optim.zero_grad()
    while True:
        epoch += 1
        train_loader.sampler.set_epoch(epoch)
        for data in train_loader:
            step += 1

            adjust_lr(optim, lr_sched, step)
            step_start_time = time.time()

            img = data[0].cuda()
            label = data[1].cuda()

            with torch.no_grad():
                code = bergan(img, code_only=True).detach()
                b,c,h,w = code.shape
                x = code.view(b,c,-1).permute(0,2,1).contiguous()

            with torch.cuda.amp.autocast(enabled=H.amp):
                if H.dataset.startswith('imagenet'):
                    stats = sampler(x, label)
                else:
                    stats = sampler(x)
                loss = stats['loss']
                loss = loss / H.update_freq

            if step == 0 and dist.get_rank() == 0:
                images = get_online_samples(H, bergan, ema_sampler if H.ema else sampler, x=x)
                save_images(images, 'samples', 999999999, H.log_dir, H.save_individually)
                # save to test the reconstruction quality

            grad_norm = scaler(loss, optim, clip_grad=H.grad_norm,
                                parameters=sampler_without_ddp.parameters(), create_graph=False,
                                update_grad=(step + 1) % H.update_freq == 0)

            if (step + 1) % H.update_freq == 0:
                optim.zero_grad()
            loss_ma.update(loss.item())
            if H.ema and step % (H.steps_per_update_ema * H.update_freq) == 0 and step > 0:
                ema.update_model_average(ema_sampler, sampler)

            torch.cuda.synchronize()

            if dist.get_rank() == 0:
                if step % H.steps_per_log == 0:

                    stats['lr'] = optim.param_groups[0]['lr']
                    step_time_taken = time.time() - step_start_time
                    stats['step_time'] = step_time_taken
                    mean_loss = np.mean(losses)
                    stats['mean_loss'] = loss_ma.avg()

                    if "scale" in scaler.state_dict().keys():
                        stats['loss scale'] = scaler.state_dict()["scale"]
                    mean_losses = np.append(mean_losses, mean_loss)
                    losses = np.array([])

                    log_stats(step, stats)

                if step % H.steps_per_save_output == 0:
                    if H.guidance:
                        images = get_online_samples_guidance(H, bergan, ema_sampler if H.ema else sampler)
                    else:
                        images = get_online_samples(H, bergan, ema_sampler if H.ema else sampler)
                    save_images(images, 'samples', step, H.log_dir, H.save_individually)


                if step % H.steps_per_checkpoint == 0 and step > H.load_step:
                    save_model(sampler, H.sampler, step, H.log_dir)
                    save_model(optim, f'{H.sampler}_optim', step, H.log_dir)
                    save_model(scaler, f'{H.sampler}_scaler', step, H.log_dir)

                    if H.ema:
                        save_model(ema_sampler, f'{H.sampler}_ema', step, H.log_dir)

                    train_stats = {
                        'losses': losses,
                        'mean_losses': mean_losses,
                        'val_losses': val_losses,
                        'elbo': elbo,
                        'val_elbos': val_elbos,
                        'steps_per_log': H.steps_per_log,
                        'steps_per_eval': H.steps_per_eval,
                    }
                    save_stats(H, train_stats, step)
            
            if step == H.train_steps:
                exit()


if __name__ == '__main__':
    H = get_sampler_hparams()
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    start_training_log(H)
    main(H, None)
