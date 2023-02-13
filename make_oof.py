import argparse
import pandas as pd
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from utils_cv_ import Model


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--name', type=str, required=True)
    p.add_argument('--model_fn', type=str, default='model_save')
    p.add_argument('--pretrained_model_name', type=str, default="microsoft/deberta-v3-base")
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_length', type=int, default=512)

    p.add_argument('--Pooling', type=str, required=True)
    config = p.parse_args()

    return config

def main(config):

    for i in range(5):
        globals()['path_{}'.format(i)] = os.path.join(config.model_fn,
                                            config.pretrained_model_name,
                                            config.name,
                                            'fold_{}'.format(i),
                                            )
    path_list = [path_0, path_1, path_2, path_3, path_4]
    test_data = pd.read_csv('data/train_fold.csv', index_col=0, encoding='utf-8')[['text_id', 'full_text', 'fold']]
    num = -1
    device = 'cuda:0'
    index_lump = []
    y_df = None
    with torch.no_grad():
        for path in path_list:
            num += 1

            texts = test_data[test_data['fold'] == num]      # target fold
            index_lump += texts.index.to_list()
            texts = texts['full_text'].to_list()

            saved_data = torch.load(os.path.join(path, 'best_.pt'), map_location='cuda:0')
            model_config = saved_data['config']

            model_config.MeanPooling = False
            model_config.MultilayerCLSpooling = False
            model_config.MultilayerMeanpooling = False
            model_config.MultilayerWeightpooling = False
            model_config.ConcatenatePooling = False
            model_config.AttentionPooling = False
            if config.Pooling == 'MeanPooling':
                model_config.MeanPooling = True
            elif config.Pooling == 'MultilayerCLSpooling':
                model_config.MultilayerCLSpooling = True
            elif config.Pooling == 'MultilayerMeanpooling':
                model_config.MultilayerMeanpooling = True
            elif config.Pooling == 'MultilayerWeightpooling':
                model_config.MultilayerWeightpooling = True
            elif config.Pooling == 'ConcatenatePooling':
                model_config.ConcatenatePooling = True
            elif config.Pooling == 'AttentionPooling':
                model_config.AttentionPooling = True
            else:
                print(5 / 0)

            tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
            model = Model(model_config).cuda()
            model.load_state_dict(saved_data['model'])
            model.eval()

            y_hats = None
            for index in tqdm(range(0, len(texts), config.batch_size)):
                mini_batch = tokenizer(
                    texts[index:index+config.batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=config.max_length,
                )

                x, mask = mini_batch['input_ids'], mini_batch['attention_mask']
                x, mask = x.to(device),  mask.to(device)
                y_hat = model(x, mask)   # (bs, 6) : tensor
                y_hats = y_hat if isinstance(y_hats, type(None)) else torch.cat([y_hats, y_hat], dim=0)

            y_hats = y_hats.detach().cpu().numpy()
            y_df = y_hats if isinstance(y_df, type(None)) else np.concatenate((y_df, y_hats), axis=0)

        y_df = pd.DataFrame(y_df)  # (3911, 6) -> numpy
        y_df.columns = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        y_df['idx'] = index_lump
        y_df = y_df.sort_values(by='idx', ascending=True)

        if os.path.isdir(os.path.join('predict', config.name)):
            pass
        else:
            os.makedirs(os.path.join('predict', config.name))

        y_df.reset_index(drop=True, inplace=True)
        y_df.to_csv(os.path.join('predict', config.name, 'oof_data.csv'), encoding='utf-8-sig')


if __name__ == '__main__':
    config = define_argparser()
    main(config)