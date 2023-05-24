# Demo for text to image generation

The current version is a simple demonstration of using binary latent diffusion for text to image generation. 

We use a pretrained (frozen) CLIP model to extract a single vector representating the text prompt, and feed it to the denoising transformer as an additional token. 
This model was trained using a small (~11M) subset of LAION with high aesthetic scores. 
We are currently training a larger model with more data and cross attention (text -> image) layers. 


![test_img](text-to-image-demo/figs/a_huge_metal_castle.jpg)