from flowvae.vae import EncoderDecoder
from flowvae.ml_operator.operator import AEOperator
from flowvae.dataset import ConditionDataset
from flowvae.base_model.conv import convEncoder, convDecoder
from flowvae.utils import warmup_lr


if __name__ == '__main__':

    #* define running devices
    device = "cuda:0"

    #* load dataset
    fldata = ConditionDataset('500', d_c=1, n_c=20, c_mtd='random', c_no=0, test=100)

    #* define some model parameters
    latent_dim = 12
    code_mode  = 'ed'
    ref        = True

    #* multiple runs for cross-validation
    for run in range(0,3):

        # define encoder and decoder
        _enc = convEncoder(in_channels=2, last_size=[5], hidden_dims=[32, 64, 128], dimension=1)
        _dec = convDecoder(out_channels=1, last_size=[5], hidden_dims=[64, 128, 256, 128], sizes=[26, 101, 401], dimension=1)

        # define the model
        vae_model = EncoderDecoder(latent_dim=latent_dim, encoder=_enc, decoder=_dec, decoder_input_layer=1, code_mode=code_mode, device=device)
        print("network have {} paramerters in total".format(sum(x.numel() for x in vae_model.parameters())))
        
        # define the operator and training parameters
        op = AEOperator('testrun', vae_model, fldata, init_lr=0.0001, num_epochs=300, batch_size=16, ref=ref, output_folder='.')    
        op.set_scheduler('LambdaLR', lr_lambda=warmup_lr)
        op.set_lossparas(sm_mode='1d', sm_epoch=0, sm_weight=0.0, aero_weight=0.0)
        
        # train the model
        op.train_model(save_check=100, v_tqdm=True)    

        print('=============================================')
        print('Run %d   Over' % run)
        
