from flowvae.vae import frameVAE, UEncoderDecoder
from flowvae.ml_operator.operator import AEOperator
from flowvae.dataset import MCFlowDataset
from flowvae.base_model.conv import convDecoder_Unet, convEncoder_Unet
from torch import nn

def warmup_lr(epoch):

    if epoch < 20:
        lr =  1 + 0.5 * epoch
    else:
        lr =  10 * 0.95**(epoch - 20) + 1e-8
    
    print(epoch, 'lr set to ', lr)

    return lr

def step_lr(epoch):

    if epoch > 100:
        return 0.1 ** int((epoch - 100) / 50 + 1)
    else:
        return 1.0

if __name__ == '__main__':

    device = "cuda:0"

    split_paras = {
        'method': 'load',  
        'indexFileNum': 0,  
    }
    fldata = MCFlowDataset('300n36n', d_c=4, split_paras=split_paras, getindex_type='normal')

    latent_dim = 12

    for run in range(0,3):

        _encoder = convEncoder_Unet(in_channels=2, last_size=[7], hidden_dims=[64, 128, 256], pool_strides=[2, 2, 0], 
                                        basic_layers={'actv': nn.GELU})
        _decoder = convDecoder_Unet(out_channels=2, last_size=[7], hidden_dims=[256, 128, 64, 64], 
                                            sizes = [14, 58, 234], encoder_hidden_dims=[256, 128, 64, 2],
                                            basic_layers={'actv': nn.GELU})
        vae_model = UEncoderDecoder(latent_dim=latent_dim,
                            encoder=_encoder,
                            decoder=_decoder,
                            code_mode='ed',
                            code_dim=4,
                            code_layer=[],
                            decoder_input_layer=1, 
                        )

        print("network have {} paramerters in total".format(sum(x.numel() for x in vae_model.parameters())))
        
        op = AEOperator('0728_43n1_Run' + str(run), vae_model, fldata, shuffle=True, ref=True, num_workers=0,
                                                        num_epochs=300,
                                                        batch_size=16, 
                                                        recon_channels=(None, None)#, 
                                                        )
                                                                
        print("network have {} paramerters in total".format(sum(x.numel() for x in vae_model.parameters())))

        op.set_optimizer('Adam', lr=0.0001)
        op.set_scheduler('LambdaLR', lr_lambda=warmup_lr)
        op.train_model(save_check=300, v_tqdm=True)    

        print('=============================================')
        print('Run %d   Over' % run)
        