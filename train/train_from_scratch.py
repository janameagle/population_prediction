from model.v_convlstm import ConvLSTM
import torch
from train.options import get_args
from utilis.dataset import MyDataset
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

proj_dir = "H:/Masterarbeit/Code/population_prediction/"



# functions
def train_ConvGRU_FullValid(net = ConvLSTM, device = torch.device('cuda'),
                  epochs=1, batch_size=1,lr=0.1,
                  save_cp=False, save_csv=True, factor_option='with_factors',
                  pred_seq='forward', model_n='No_seed_convLSTM'):
    
    dataset_dir = proj_dir + "data/"
    train_dir = dataset_dir + "train/"
    pred = 'lulc_pred_6y_6c_no_na/'
    train_data = MyDataset(imgs_dir = train_dir + pred + 'input/',masks_dir = train_dir + pred +'target/')
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers= 2)
    
    








# define some arguments
args = get_args()

bias_status = True #False                                          # ?
beta = 0                                                           # ?
input_channel = 6                                            # 19 driving factors
factor = 'with_factors'
pred_sequence = 'forward'
model_n = 'No_seed_convLSTM_no_na_scratch'

# define the network
net = ConvLSTM(input_dim=input_channel,
               hidden_dim=[32, 16, args.n_features], # hidden_dim = [32, 16, args.n_features]
               kernel_size=(3, 3), num_layers=args.n_layer,
               batch_first=True, bias=bias_status, return_all_layers=False)

# define the device
net.to(device)

# train the network by using the train function
train_ConvGRU_FullValid(net=net, device=device,
               epochs=5, batch_size=args.batch_size, lr=args.learn_rate,
               save_cp=False, save_csv=True, factor_option=factor,
               pred_seq=pred_sequence, model_n=model_n)