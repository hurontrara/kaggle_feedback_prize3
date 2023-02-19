# kaggle_feedback_prize3
Feedback Prize - English Language Learning(Kaggle Competition) 115th (SLIVER MEDAL)
# Usages
* Can use bash.ipynb in colab ( not tested in any local )
* You can use it by giving options like this.
```bash
!python3 train_cv.py --name Concatenate_512_mse --model_fn model_save --train_data_name train_fold --pretrained_model_name microsoft/deberta-v3-base --encoder_lr 2e-5 --decoder_lr 2e-5 --batch_size 8 --iteration_per_update 1 --max_length 512 --valid_fold 0 --n_epochs 5 --ConcatenatePooling --cosine --warmup_ratio 0 
```

* Available options are defined in train_cv.py -> define_argparser()
```python
def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--name', type=str, required=True)
    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_data_name', default='train_fold') # train_data
    p.add_argument('--pretrained_model_name', type=str, default="microsoft/deberta-v3-base")
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--adam_epsilon', type=float, default=1e-6)
    p.add_argument('--valid_fold', type=int, default=0)
    p.add_argument('--use_radam', action='store_true')
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--encoder_lr', type=float, default=2e-5)
    p.add_argument('--decoder_lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=.01)
    p.add_argument('--betas', type=float, default=(0.9, 0.999))
    p.add_argument('--max_grad_norm', type=float, default=1000)
    p.add_argument('--iteration_per_update', type=int, default=32)

    p.add_argument('--cosine', action='store_true')
    p.add_argument('--mse', action='store_true')

    p.add_argument('--MeanPooling', action='store_true')
    p.add_argument('--MultilayerCLSpooling', action='store_true')
    p.add_argument('--MultilayerMeanpooling', action='store_true')
    p.add_argument('--MultilayerWeightpooling', action='store_true')
    p.add_argument('--ConcatenatePooling', action='store_true')
    p.add_argument('--AttentionPooling', action='store_true')

    p.add_argument('--awp', action='store_true')

    config = p.parse_args()

    return config
```


* Also you can make oof-file by make_oof.py
```bash
!python3 make_oof.py --name Attn_512 --model_fn model_save --pretrained_model_name microsoft/deberta-v3-base --batch_size 32 --max_length 512 --Pooling AttentionPooling
```

# KeyIdea for solution
* There are many options available.
* -> a result of the ensemble of options
* https://wandb.ai/hurontrara/feedback3?workspace=user-hurontrara
* -> I've organized the results of the experiment. 
* The final ensemble was performed using (sklearn's linear regression + optuna optimization)  method.
* You can check ML.ipynb
