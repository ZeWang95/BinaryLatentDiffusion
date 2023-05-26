# Binary Autoencoder


In the original paper, we used a dedicated binary autoencoder trained specifically for each experiment. To promote more general usage, we now release two binary autoencoders trained using over 600M images from the [LAION](https://laion.ai/projects/) dataset.

- **C64**
The 64-latent-channel binary autoencoder can be downloaded [here](https://drive.google.com/drive/folders/1CFlixXLnEHZ0jaRLXLS4d_bJMZwG-Iih?usp=sharing). It represents a 256x256 image as a 16x16x64 binary tensor. 

- **C128**
The 128-latent-channel binary autoencoder can be downloaded [here](https://drive.google.com/drive/folders/1rlWjd5iDOydTrxJyfvLZ8oDogRB3HiMe?usp=sharing). It represents a 256x256 image as a 16x16x128 binary tensor. 

The C64 binary autoencoder should be able to reconstruct any images with fair quality. The C128 one offers a better reconstruction quality but poses additional challenges for the training of the diffusion sampler. 

If you would like to train your own binary autoencoder, you can try running:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train_ae_dist.py --dataset custom --amp --ema --steps_per_save_output 5000 --codebook_size 32 --steps_per_log 200 --steps_per_checkpoint 10000 --img_size 256 --batch_size 4 --latent_shape 1 16 16 --path_to_data /path-to-your-data --log_dir logs/binaryae_custom --disc_start_step 400001 --norm_first
```
where `/path-to-your-data` points to a folder or folders containing your image data. 

Or you can try finetuning the provided pretrained binary autoencoders by:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train_ae_dist.py --dataset custom --amp --ema --steps_per_save_output 5000 --codebook_size 64 --steps_per_log 200 --steps_per_checkpoint 10000 --img_size 256 --batch_size 4 --latent_shape 1 16 16 --path_to_data /path-to-your-data --log_dir logs/binaryae_custom --disc_start_step 400001 --norm_first --load_dir logs/BAE_C64/ --load_step 8100000
```
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train_ae_dist.py --dataset custom --amp --ema --steps_per_save_output 5000 --codebook_size 128 --steps_per_log 200 --steps_per_checkpoint 10000 --img_size 256 --batch_size 4 --latent_shape 1 16 16 --path_to_data /path-to-your-data --log_dir logs/binaryae_custom --disc_start_step 400001 --norm_first --load_dir logs/BAE_C128/ --load_step 10200000 --gen_mul 3
```
