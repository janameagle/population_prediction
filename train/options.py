import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train ConvLSTM Models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-epoch', '--epoch', default=151, type=int, dest='epoch')
    parser.add_argument('-lr', '--learn_rate', default=8e-4, type=float, dest='learn_rate')
    parser.add_argument('-f', '--n_features', default=4, type=int, dest='n_features')
    parser.add_argument('-b', '--batch_size', default=12, type=int, nargs='?', help='Batch size', dest='batch_size') #5
    parser.add_argument('-n', '--n_layer', default=3, type=int, dest='n_layer')
    parser.add_argument('-l', '--seq_len', default=4, type=int, dest='seq_len')
    parser.add_argument('-is', '--input_shape', default=(256, 256), type=tuple, dest='input_shape')

    return parser.parse_args()