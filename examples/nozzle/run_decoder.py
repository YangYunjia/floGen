from flowvae.ml_operator import ModelOperator
from flowvae.dataset import FlowDataset
import matplotlib.pyplot as plt
from torch import nn
from flowvae.vae import DecoderModel
from flowvae.base_model.conv import convDecoder

def warmup_lr(epoch):

    if epoch < 20:
        lr =  1 + 0.5 * epoch
    else:
        lr =  10 * 0.95**(epoch - 20) + 1e-8
    
    print(epoch, 'lr set to ', lr)

    return lr


if __name__ == '__main__':

    device = "cuda:0"
    split_paras = {
        'method': 'load',   
        'indexFileName': 'D:\\jetSERN\\training\\data\\300n36_0dataindex.txt', 
    }
    fldata = FlowDataset(['300n36ndata', '300n36n2index'], split_paras=split_paras, input_channel_take=[3,4,5,6,15,16,17,18])

    for run in range(0,3):
        
        decoder = convDecoder(out_channels=2, last_size=[7], hidden_dims=[256, 128, 64, 64], 
                                            sizes = [14, 58, 234], basic_layers={'actv': nn.GELU})
        
        vae_model = DecoderModel(input_channels=8, decoder=decoder, device=device, decoder_input_layer=1) # 53

        print("network have {} paramerters in total".format(sum(x.numel() for x in vae_model.parameters())))

        op = ModelOperator('0728_53n1_Run' + str(run), vae_model, fldata, shuffle=True, num_workers=0,
                                                        batch_size=16, 
                                                        num_epochs=300)

        op.set_optimizer('Adam', lr=0.0001)
        op.set_scheduler('LambdaLR', lr_lambda=warmup_lr)
        
        op.train_model(save_check=300, v_tqdm=True)    

        print('=============================================')
        print('Run %d   Over' % run)
        