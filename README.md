# Inpainting-Structural-Explorer
This project only contains ablation studies for ONE-STAGE inpainting models in pytorch.

1. model_type
>EC: EC is the model used in EdgeConnect (most common in inpainting)

>UNET: UNET is used widely in segmentation and some inpainting tasks (PEN)

2. norm_type
>InstanceNorm

>BatchNorm

3. dis_spectral_norm, gen_spectral_norm
> whether to use sn in D and G

4. econv_type, dconv_type, dis_conv_type
> control the type of convolution used in various processes of inpainting 