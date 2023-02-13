from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os
import random
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import torch_optimizer as custom_optim
from transformers import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
import wandb
import torch.optim as optim
from pytorch_lightning.callbacks import TQDMProgressBar

# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from lightgbm import LGBMRegressor
# from sklearn.multioutput import MultiOutputRegressor
# from xgboost import XGBRegressor
#
# from torch.utils.checkpoint import checkpoint

class Pooling(nn.Module):

    def __init__(self, config, cfg):
        super().__init__()
        self.config = config

        if (config.name == 'Meanpool_mse_1024') | (config.name == 'Attention') | (config.name == 'MultilayerMeanpool') | (config.AttentionPooling):
            self.attention = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.GELU(),
                nn.Linear(cfg.hidden_size, 1),
            ).to('cuda')

    def forward(self, last_hidden_state, attention_mask):

        if self.config.MeanPooling:

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            return mean_embeddings

        elif self.config.MultilayerCLSpooling:

            layers = torch.stack(list(last_hidden_state), dim=0)[:, :, 0]  # (12, bs, n, 768) -> (12, bs, 768)
            layer_len = layers.shape[0]      # 12 : base / 24 : large
            layers = torch.sum(layers, dim=0) / layer_len # (bs, 768)

            return layers

        elif self.config.MultilayerMeanpooling:
            layers = torch.stack(list(last_hidden_state[-4:]), dim=0)  # (4, bs, n, 768)
            input_mask_expanded = attention_mask.unsqueeze(-1).unsqueeze(0).expand(layers.size()).float() # (4, bs, n, 768)
            weight_factor = torch.tensor([.25, .25, .25, .25]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                input_mask_expanded.size()).to('cuda')    # (4, bs, n, 768)
            sum_embeddings = torch.sum((layers * input_mask_expanded) * weight_factor, 2) # (4, bs, 768)
            sum_mask = input_mask_expanded.sum(2) # (4, bs, 768)

            weight_embeddings = sum_embeddings / sum_mask  # (4, bs, 768)
            weight_embeddings = weight_embeddings.sum(0)  # (bs, 768)

            return weight_embeddings


        elif self.config.MultilayerWeightpooling:
            layers = torch.stack(list(last_hidden_state[-4:]), dim=0)  # (4, bs, n, 768)
            input_mask_expanded = attention_mask.unsqueeze(-1).unsqueeze(0).expand(layers.size()).float() # (4, bs, n, 768)
            weight_factor = torch.tensor([.1, .2, .3, .4]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                input_mask_expanded.size()).to('cuda')    # (4, bs, n, 768)
            sum_embeddings = torch.sum((layers * input_mask_expanded) * weight_factor, 2) # (4, bs, 768)
            sum_mask = input_mask_expanded.sum(2) # (4, bs, 768)

            weight_embeddings = sum_embeddings / sum_mask  # (4, bs, 768)
            weight_embeddings = weight_embeddings.sum(0)  # (bs, 768)

            return weight_embeddings


        elif self.config.ConcatenatePooling:

            layers = torch.stack(list(last_hidden_state[-4:]), dim=0)  # (4, bs, n, 768)
            input_mask_expanded = attention_mask.unsqueeze(-1).unsqueeze(0).expand(layers.size()).float() # (4, bs, n, 768)
            sum_embeddings = torch.sum(layers * input_mask_expanded, 2) # (4, bs, 768)
            sum_mask = input_mask_expanded.sum(2) # (4, bs, 768)
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            concatenate_embeddings = sum_embeddings / sum_mask  # (4, bs, 768)
            concatenate_embeddings = torch.cat(
                (concatenate_embeddings[0], concatenate_embeddings[1], concatenate_embeddings[2], concatenate_embeddings[3]), -1)  # (bs, 768 * 4)

            return concatenate_embeddings

        elif self.config.AttentionPooling:

            w = self.attention(last_hidden_state).float()  #
            w[attention_mask == 0] = float('-inf')
            w = torch.softmax(w, 1)
            last_hidden_state = torch.sum(w * last_hidden_state, dim=1)

            return last_hidden_state

        else:
            return last_hidden_state[:, 0, :].squeeze(1)   # bs , 768

        # elif self.config.MaxPooling:
        #     return max_embeddings
        # elif self.config.MeanMaxPooling:
        #     concat = torch.cat([mean_embeddings, max_embeddings], dim=1)  # bs , 768 * 2
        #     return concat


class Model(pl.LightningModule):

    def __init__(self, cfg):
        self.cfg = cfg
        self.pretrained_model_name = cfg.pretrained_model_name
        self.config = AutoConfig.from_pretrained(cfg.pretrained_model_name, output_hidden_states=True)
        self.config.hidden_dropout = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.attention_dropout = 0.
        self.config.attention_probs_dropout_prob = 0.
        super().__init__()
        self.pool = Pooling(cfg, self.config)

        self.model = AutoModel.from_pretrained(cfg.pretrained_model_name, config=self.config)
        self.model.gradient_checkpointing_enable()

        dimension = self.config.hidden_size if not self.cfg.ConcatenatePooling else self.config.hidden_size*4

        self.linear = nn.Linear(dimension, 6)
        nn.init.orthogonal_(self.linear.weight.data)
        self.linear.bias.data.zero_()

        self.layernorm = nn.LayerNorm(dimension)
        self._init_weights(self.layernorm)

        self.crit = nn.SmoothL1Loss(reduction='mean') if not cfg.mse else nn.MSELoss(reduction='mean')
        self.crit = self.crit.cuda(cfg.gpu_id)

        self.best_loss = np.inf


    @property
    def automatic_optimization(self) -> bool:
        return False


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, mask):
        if (self.cfg.MeanPooling) | (self.cfg.AttentionPooling):
            last_hidden_state = self.model(x, attention_mask=mask)[0]
        else:
            last_hidden_state = self.model(x, attention_mask=mask).hidden_states[1:]
        embeddings = self.pool(last_hidden_state, mask) # (bs, 768) / (bs, 768*2)
        embeddings = self.layernorm(embeddings)
        y_hat = self.linear(embeddings)

        return y_hat

    def MCRMSE_cal(self, y_trues, y_preds):
        scores = []
        idxes = y_trues.shape[1]
        for i in range(idxes):
            y_true = y_trues[:, i]
            y_pred = y_preds[:, i]
            score = mean_squared_error(y_true, y_pred, squared=False)  # eval case : both tensor but no device
            scores.append(score)
        mcrmse_score = np.mean(scores)
        return mcrmse_score, scores

    def get_score(self, y_trues, y_preds):
        mcrmse_score, scores = self.MCRMSE_cal(y_trues, y_preds)
        return mcrmse_score, scores


    def training_step(self, train_batch, batch_idx):
        x, y, mask = train_batch['input_ids'], train_batch['labels'], train_batch['attention_mask']
        x, y, mask = x.to('cuda:{}'.format(self.cfg.gpu_id)), y.to('cuda:{}'.format(self.cfg.gpu_id)), mask.to('cuda:{}'.format(self.cfg.gpu_id))

        if batch_idx == 0:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            if self.cfg.awp:
                self.awp = AWP(self, self.optimizers(), self.crit, adv_lr=1, adv_eps=1e-3, start_epoch=2, scaler=self.scaler)

        with torch.cuda.amp.autocast(enabled=True):
            y_hat = self.forward(x, mask)
            loss = self.crit(y_hat, y)

        self.scaler.scale(loss).backward()

        if (self.cfg.iteration_per_update == 1) | (batch_idx % self.cfg.iteration_per_update == 1):
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1000)
            optimizer = self.optimizers()
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
            self.lr_schedulers().step()
            if batch_idx % 20 == 1:
                print('batch : {} / loss : {} / lr : {} / grad : {}'.format(batch_idx, loss,
                                                                            self.lr_schedulers().get_lr()[0],
                                                                            grad_norm))
        if self.cfg.awp:
            self.awp.attack_backward(x, y, mask, self.current_epoch)


        # self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return {'loss' : loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss)

        print(f'epoch {self.current_epoch} train loss {avg_loss}')

        # return {'train_loss': avg_loss}    # return None


    def validation_step(self, valid_batch, batch_idx):
        x, y, mask = valid_batch['input_ids'], valid_batch['labels'], valid_batch['attention_mask']
        x, y, mask = x.to('cuda:{}'.format(self.cfg.gpu_id)), y.to('cuda:{}'.format(self.cfg.gpu_id)), mask.to('cuda:{}'.format(self.cfg.gpu_id))
        y_hat = self.forward(x, mask)
        loss = self.crit(y_hat, y)

        if (batch_idx+1) % 20 == 0:
            print('batch : {} / loss : {} / lr : {}'.format(batch_idx, loss, self.lr_schedulers().get_lr()[0]))

        return {'val_loss' : loss, 'y_hat' : y_hat, 'y' : y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        output_val = torch.cat([x['y_hat'] for x in outputs], dim=0).cpu().detach().numpy()
        target_val = torch.cat([x['y'] for x in outputs], dim=0).cpu().detach().numpy()
        avg_score, _ = self.get_score(output_val, target_val)

        self.log('val_loss', avg_loss)
        self.log('val_score', avg_score)

        print('valid_loss : {}  /  valid_score :  {} '.format(avg_loss, avg_score))
        # wandb.log({'Valid_loss' : avg_score})
        if avg_score < self.best_loss:
            self.best_loss = avg_score
            self.best_model = deepcopy(self.state_dict())
            self.save_model(self.cfg, avg_score)
            print('saved')

        return {'val_loss' : avg_loss, 'val_mcrmse' : avg_score}


    def get_optimizer_params(self, encoder_lr, decoder_lr, weight_decay):
        if self.cfg.use_radam:
            optimizer = custom_optim.RAdam(self.parameters(), lr=self.cfg.lr)
            return optimizer
        else:
            param_optimizer = list(self.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'lr': encoder_lr, 'weight_decay': weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'lr': encoder_lr, 'weight_decay': 0.0},
                {'params': [p for n, p in self.named_parameters() if "model" not in n],
                 'lr': decoder_lr, 'weight_decay': 0.0}
            ]

            return optimizer_parameters

    def configure_optimizers(self):
        optimizer_parameters = self.get_optimizer_params(
                                         encoder_lr=self.cfg.encoder_lr,
                                         decoder_lr=self.cfg.decoder_lr,
                                         weight_decay=self.cfg.weight_decay)

        optimizer = optim.AdamW(optimizer_parameters, lr=self.cfg.encoder_lr, eps=self.cfg.adam_epsilon,
                                betas=self.cfg.betas)

        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.n_warmup_steps,
            num_training_steps=self.cfg.n_total_iterations
        ) if not self.cfg.cosine else get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.n_warmup_steps,
            num_training_steps=self.cfg.n_total_iterations,
            num_cycles=.5,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     scheduler.step()

    # if self.multi_sample:            y_hat = torch.mean(                         # bs , 768
        #             torch.stack(
        #                 [self.linear_1(embeddings),
        #                  self.linear_2(embeddings),
        #                  self.linear_3(embeddings),
        #                  self.linear_4(embeddings),
        #                  self.linear_5(embeddings)],
        #                 dim=0,
        #             ),
        #             dim=0,
        #         )
        #     return y_hat

        # else:

    def save_model(self, config, loss):
        path = os.path.join(config.model_fn,
                            config.pretrained_model_name,
                            config.name,
                            'fold_{}'.format(config.valid_fold),
                            )

        if not os.path.isdir(path):
            if 'working' not in os.getcwd():
                os.makedirs(path)
            else:
                pass
        else:
            pass

        torch.save(
            {
                'model': self.state_dict(),
                'config' : config,
                'loss' : round(loss, 4)
            }, os.path.join(path, 'best_.pt') if 'working' not in os.getcwd() else 'best_{}_{}.pt'.format(str(round(loss, 4)), config.valid_fold)
        )


def read_text(fn, valid_fold, config):
    data = os.path.join('data', str(fn) + '.csv') if 'working' not in os.getcwd() else os.path.join('../input/colab-dataset', 'data', str(fn) + '.csv')
    data = pd.read_csv(data, encoding='UTF-8')
    eval_list = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

    train_data = data[data['fold'] != valid_fold]
    valid_data = data[data['fold'] == valid_fold]
    del data

    train_texts = train_data['full_text'].values
    valid_texts = valid_data['full_text'].values
    train_labels = train_data[eval_list].values
    valid_labels = valid_data[eval_list].values

    # with open(fn, 'r') as f
    #     lines = f.readlines()
    #
    #     labels, texts = [], []
    #     for line in lines:
    #         if line.strip() != '':
    #             # The file should have tab delimited two columns.
    #             # First column indicates label field,
    #             # and second column indicates text field.
    #             label, text = line.strip().split('\t')
    #             labels += [label]
    #             texts += [text]

    return train_texts, valid_texts, train_labels, valid_labels


def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def getMaskedLabels(input_ids):
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    #
    # special_tokens = tokenizer.encode_plus('[CLS] [SEP] [MASK] [PAD]',
    #                                        add_special_tokens=False,
    #                                        return_tensors='pt')
    # special_tokens = torch.flatten(special_tokens["input_ids"])
    special_tokens = [1, 2, 128000, 0]
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < .1).to('cuda:0')
    # Preventing special tokens to get replace by the [MASK] token
    for special_token in special_tokens:
        token = special_token
        mask_arr *= (input_ids != token)

    for i in range(len(mask_arr)):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()  # [0, 1]
        input_ids[i][selection] = 128000

    return input_ids

def getDataset(fold_data_name, profile_data_name, fold):
    fold_data = pd.read_csv(os.path.join('data', fold_data_name+'.csv'))
    fold_data.rename({'Unnamed: 0': 'Index'}, axis=1, inplace=True)
    fold_data.set_index('Index', inplace=True)
    train_data = fold_data[fold_data['fold'] != fold]
    predict_data = fold_data[fold_data['fold'] == fold]

    profile_data = pd.read_csv(os.path.join('data', profile_data_name+'.csv'))
    profile_data['grammar_check_score'] = profile_data['grammar_check_score'] / profile_data['count_words']  # 1. grammar_check_score scaling by count_words : -0.44
    profile_data.drop(['Unnamed: 0', 'full_text'], axis=1, inplace=True)
    profile_train_data = profile_data.loc[train_data.index.to_list()]  # drop full_text
    profile_valid_data = profile_data.loc[predict_data.index.to_list()]  # drop full_text

    return train_data, predict_data, profile_train_data, profile_valid_data


class BoostingModel():  # basic ; input -> profile_data
    def __init__(self, train_data, train_predict, profile_train_data, config, prediction='cohesion', boosting='LGBM',):

        self.prediction = prediction
        self.boosting = boosting
        self.config = config

        train_x = profile_train_data.copy()

        self.full_column_list = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        column_list = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        column_list.remove(prediction)
        self.column_list = deepcopy(column_list)

        if config.pseudo_label:
            train_predict = pd.DataFrame(deepcopy(train_predict))
            train_predict.columns = self.full_column_list

            for i in self.column_list:
                train_x[i] = train_predict[i]
        else:
            pass
        train_y = train_data[[prediction]].to_numpy()

        numeric_features = train_x.columns.to_list()
        numeric_transformer = StandardScaler()
        categorical_features = []
        categorical_transformer = OneHotEncoder(
            categories='auto')  # categories='auto' : just for ignoring warning messages
        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, numeric_features),
                          ('passthrough', 'passthrough', categorical_features)])
        preprocessor_pipe = Pipeline(steps=[('preprocessor', preprocessor)])
        preprocessor_pipe.fit(train_x)
        self.preprocessor_pipe = preprocessor_pipe
        train_x_transformed = preprocessor_pipe.transform(train_x)
        if boosting == 'LGBM':
            self.model = LGBMRegressor()
        elif boosting == 'XGBoost':
            self.model = XGBRegressor()
        else:
            ValueError('no boosting model')

        self.model.fit(train_x_transformed, train_y)

    def predict(self, predict_data, profile_valid_data):
        predict_x = profile_valid_data.copy()

        if self.config.pseudo_label:
            if isinstance(predict_data, pd.DataFrame):  # -->  validation by 'actual data'
                for i in self.column_list:
                    predict_x[i] = predict_data[i]

            else:  # --> validation by 'inference data' / predict_data : numpy.array
                predict_data = pd.DataFrame(predict_data)
                predict_data.columns = self.full_column_list
                predict_data = predict_data[self.column_list]
                for i in self.column_list:  # sequence same
                    predict_x[i] = predict_data[i]

        else:
            pass

        transformed_data = self.preprocessor_pipe.transform(predict_x)
        predict_y = self.model.predict(transformed_data)

        return predict_y

class AWP:
    def __init__(
            self,
            model,
            optimizer,
            crit,
            adv_param="weight",
            adv_lr=1,
            adv_eps=0.2,
            start_epoch=0,
            adv_step=1,
            scaler=None

    ):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, x, y, attention_mask, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast():
                y_hat = self.model(x, attention_mask)
                adv_loss = self.crit(y_hat, y)

            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward() if self.scaler else adv_loss.backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self, ):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
