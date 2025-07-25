
from flowvae.base_model.conv import convEncoder, convDecoder, convEncoder_Unet, convDecoder_Unet
from flowvae.base_model.mlp import mlpDecoder
from flowvae.vae import EncoderDecoder, Unet, BranchUnet, BranchEncoderDecoder


def load_encoder_decoder(_id):

    # if _id == 0:
    #     _encoder = bm.mlpEncoder(in_channels=802, hidden_dims=[256, 128, 64])
    #     _decoder = bm.mlpDecoder(in_channels=latent_dim, out_channels=401, hidden_dims=[64, 128, 256])
    if _id == 1:
        _encoder = convEncoder(in_channels=2, last_size=[5], hidden_dims=[64, 128, 256])
        _decoder = convDecoder(out_channels=1, last_size=[5], hidden_dims=[256, 512, 256, 128], sizes=[26, 101, 401])
    elif _id == 2:
        _encoder = convEncoder(in_channels=2, last_size=[5], hidden_dims=[64, 128, 256])
        _decoder = convDecoder(out_channels=1, last_size=[5], hidden_dims=[256, 256, 128, 64], sizes=[26, 101, 401])
    elif _id == 3:
        _encoder = convEncoder(in_channels=2, last_size=[5], hidden_dims=[64, 128, 256])
        _decoder = convDecoder(out_channels=1, last_size=[5], hidden_dims=[512, 256, 256, 128], sizes=[26, 101, 401])

    elif _id == 4:
        _encoder = convEncoder(in_channels=2, last_size=[6], hidden_dims=[64, 128, 128, 256], pool_kernels=[3,0,3,0], pool_strides=[2,0,2,0])
        _decoder = convDecoder(out_channels=1, last_size=[5], hidden_dims=[512, 256, 256, 128], sizes=[26, 101, 401])
    elif _id == 5:
        _encoder = convEncoder(in_channels=2, last_size=[5], hidden_dims=[64, 128, 256])
        _decoder = convDecoder(out_channels=1, last_size=[5], hidden_dims=[256, 128, 64, 64], sizes=[26, 101, 401])
    elif _id == 50:
        _encoder = convEncoder(in_channels=2, last_size=[5], hidden_dims=[64, 128, 256])
        _decoder = [convDecoder(out_channels=1, last_size=[5], hidden_dims=[256, 128, 64, 64], sizes = [24, 100, 401]),
                    convDecoder(out_channels=1, last_size=[5], hidden_dims=[256, 128, 64, 64], sizes = [24, 100, 401])]
    elif _id == 51:
        _encoder = convEncoder_Unet(in_channels=2, last_size=[5], hidden_dims=[64, 128, 256])
        _decoder = convDecoder_Unet(out_channels=1, last_size=[5], hidden_dims=[256, 128, 64, 64], 
                                            sizes = [24, 100, 401], encoder_hidden_dims=[256, 128, 64, 2])
    elif _id == 52:
        _encoder = convEncoder_Unet(in_channels=3, last_size=[5], hidden_dims=[64, 128, 256])
        _decoder = convDecoder_Unet(out_channels=2, last_size=[5], hidden_dims=[256, 128, 64, 64], 
                                            sizes = [24, 100, 401], encoder_hidden_dims=[256, 128, 64, 3])
    elif _id == 82:
        _encoder = convEncoder_Unet(in_channels=3, last_size=[6], hidden_dims=[64, 128, 128, 256], pool_kernels=[3, 3, 0, 0])
        _decoder = convDecoder_Unet(out_channels=2, last_size=[6], hidden_dims=[256, 128, 128, 64, 64], 
                                            sizes = [12, 24, 100, 401], encoder_hidden_dims=[256, 128, 128, 64, 3])
    elif _id == 92:
        _encoder = convEncoder_Unet(in_channels=3, last_size=[7], hidden_dims=[64, 128, 128, 256, 256], pool_kernels=[3, 0, 0, 0, 0])
        _decoder = convDecoder_Unet(out_channels=2, last_size=[7], hidden_dims=[256, 256, 128, 128, 64, 64], 
                                            sizes = [13, 25, 50, 100, 401], encoder_hidden_dims=[256, 256, 128, 128, 64, 3])
    
    elif _id == 6:
        _encoder = convEncoder(in_channels=2, last_size=[5], hidden_dims=[32, 64, 128])
        _decoder = convDecoder(out_channels=1, last_size=[5], hidden_dims=[128, 64, 32, 32], sizes=[26, 101, 401])
    elif _id == 61:
        _encoder = convEncoder_Unet(in_channels=2, last_size=[5], hidden_dims=[32, 64, 128])
        _decoder = convDecoder_Unet(out_channels=1, last_size=[5], hidden_dims=[128, 64, 32, 32],
                                            sizes = [24, 100, 401], encoder_hidden_dims=[128, 64, 32, 2])
    elif _id == 72:
        _encoder = convEncoder_Unet(in_channels=3, last_size=[5], hidden_dims=[128, 256, 512])
        _decoder = convDecoder_Unet(out_channels=2, last_size=[5], hidden_dims=[512, 256, 128, 128], 
                                            sizes = [24, 100, 401], encoder_hidden_dims=[512, 256, 128, 3])

    if _id < 20:
        _model_type = EncoderDecoder

    elif _id in [50]:
        _model_type = BranchEncoderDecoder

    elif _id > 95 and _id <= 105:
        _model_type = EncoderDecoder

        if _id in [100, 105, 104]: hidden_dims = [128, 256, 128]
        elif _id == 102: hidden_dims = [512, 512, 256]
        elif _id == 101: hidden_dims = [256, 512, 256]
        elif _id == 99: hidden_dims = [64, 128, 64]
        elif _id == 98: hidden_dims = [32, 64, 32]
        elif _id == 97: hidden_dims = [32, 32]
        elif _id == 96: hidden_dims = [32]

        if _id in [105]:
            i_p = 2
        elif _id in [104]:
            i_p = 4
        else:
            i_p = 3
        _encoder = convEncoder(in_channels=i_p, last_size=[5], hidden_dims=[64, 128, 256])
        _decoder = mlpDecoder(out_sizes=[2], hidden_dims=hidden_dims)
    
    
    elif _id in [224, 30, 33, 34, 50, 53, 54, 550, 553, 554, 63, 660, 663, 664]:


        if _id in [30, 33, 34]: hidden_dims = [512, 256, 128, 128]
        elif _id in [224]: hidden_dims = [128, 128, 64, 32]
        elif _id in [50, 53, 54]: hidden_dims = [256, 128, 64, 64]
        elif _id in [550, 553, 554]: hidden_dims = [256, 256, 128, 64]
        elif _id in [63]: hidden_dims = [512, 512, 256, 128]
        elif _id in [660, 663, 664]: hidden_dims = [512, 256, 256, 128]
        
        if _id % 10 == 3 or _id % 10 == 4: 
            print('model is BranchUnet')
            _model_type = BranchUnet
            ip = int(_id % 10)
            _encoder = convEncoder_Unet(in_channels=ip, last_size=[5], hidden_dims=[64, 128, 256])
            _decoder = [convDecoder_Unet(out_channels=1, last_size=[5], hidden_dims=hidden_dims, 
                                            sizes=[24, 100, 401], encoder_hidden_dims=[256, 128, 64, ip]),
                        convDecoder_Unet(out_channels=1, last_size=[5], hidden_dims=hidden_dims, 
                                            sizes=[24, 100, 401], encoder_hidden_dims=[256, 128, 64, ip])]
        else: 
            print('model is BranchEncoderDecoder')
            _model_type = BranchEncoderDecoder
            _encoder = convEncoder(in_channels=2, last_size=[5], hidden_dims=[64, 128, 256])
            _decoder = [convDecoder(out_channels=1, last_size=[5], hidden_dims=hidden_dims, sizes = [24, 100, 401]),
                        convDecoder(out_channels=1, last_size=[5], hidden_dims=hidden_dims, sizes = [24, 100, 401])]

    else:
        _model_type = Unet

    return _encoder, _decoder, _model_type