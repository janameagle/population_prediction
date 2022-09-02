from train.train_GRUs_FullValid import train_ConvGRU_FullValid
from train.train_ensemble_ViT_FullValid import train_ensemble_ViT_FullValid
from model.v_convlstm import ConvLSTM
from model.v_convgru import ConvGRU

from transformer.ViT_ensemble import ensemble_ViT
import torch
from train.options import get_args

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    args = get_args()

    pred_sequence_list = ['forecasting'] #'backcasting'                        # what is backcasting? why do it?

    for pred_sequence in pred_sequence_list:
        factor_list = ['with_factors']                                         # what factors?

        for factor in factor_list:

            bias_status = True #False                                          # ?
            beta = 0                                                           # ?

            if factor == 'with_factors':
                input_channel = 19                                             # why 19?

            else:
                pass

            model_list = ['v_convlstm'] # the models to train ## change to ViT?

            for i in range(len(model_list)):
                model_n = model_list[i]
                print('{} training {} with lr {} and factor as {}...'.format(pred_sequence, model_n,
                                                                             args.learn_rate, factor))
                if seed is not None:
                    torch.manual_seed(seed)                                    # should I set a seed for model improvement? 
                else:
                    seed = 'noSeed'
                print('seed: ', seed)


                if model_n == 'v_convlstm':
                    net = ConvLSTM(input_dim=input_channel,
                                   hidden_dim=[32, 16, args.n_features],
                                   kernel_size=(3, 3), num_layers=args.n_layer,
                                   batch_first=True, bias=bias_status, return_all_layers=False)
                    net.to(device)
                    train_ConvGRU_FullValid(net=net, device=torch.device('cuda'),
                                  epochs=args.epoch, batch_size=args.batch_size, lr=args.learn_rate,
                                  save_cp=True, save_csv=True, factor_option=factor,
                                  pred_seq=pred_sequence, model_n=str(seed) + '_' + model_n)

                elif model_n == 'v_convgru':
                    net = ConvGRU(input_dim=input_channel,
                                   hidden_dim=[32, 16, args.n_features],
                                   kernel_size=(3, 3), num_layers=args.n_layer,
                                   batch_first=True, bias=bias_status, return_all_layers=False)
                    net.to(device)
                    train_ConvGRU_FullValid(net=net, device=torch.device('cuda'),
                                  epochs=args.epoch, batch_size=args.batch_size, lr=args.learn_rate,
                                  save_cp=True, save_csv=True, factor_option=factor,
                                  pred_seq=pred_sequence, model_n=str(seed) + '_' + model_n)


                elif model_n == 'ensemble_ViT':
                    if factor == 'with_factors':
                        net = ensemble_ViT(n_classes=7)
                        net.to(device)

                    train_ensemble_ViT_FullValid(net=net, device=torch.device('cuda'),
                                        epochs=31, batch_size=2, lr=args.learn_rate,
                                        save_cp=True, save_csv=True, factor_option=factor,
                                        pred_seq=pred_sequence,
                                        model_n=str(seed) + '_' + model_n, changemap=False)


                else:
                    print('error in model list!')

                print('completed training {}...'.format(model_n))
                print('=============================================================================================')

