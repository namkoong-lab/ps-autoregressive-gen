import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Post-processing MIND data: remove articles with too few clicks + transform click probabilities')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir

    news_data = torch.load(f'{data_dir}/news_data_all.pt')
    click_data = torch.load(f'{data_dir}/click_data_all.pt')

    # Load data
    def get_num_obs_and_mean(click_data, key):
        obs = [x[1] for x in click_data[key]]
        return {'article_id': key, 'num_obs':len(obs), 'success_p': np.array(obs).mean()}

    article_and_click_stats = [get_num_obs_and_mean(click_data, k) for k in click_data.keys()]

    df = pd.DataFrame(article_and_click_stats)

    # Filter data
    df100 = df.loc[df.num_obs >= 100]

    print('Total number of articles', len(df.article_id.values) )
    print('Total number of articles with more than 100 obs', len(df100.article_id.values) )

    # Transform probabilities
    from scipy.special import expit, logit
    import copy
    def transform_success_p(raw_p, alpha):
        logits = logit(raw_p)
        
        # handle zero and one success probabilities
        mask = (raw_p > 0) * (raw_p < 1)
        logits_mean = logit(raw_p[mask]).mean() # only take mean of non zero or one probs
        
        probs = (1-mask) * raw_p + mask * expit((logits - alpha*logits_mean))
        return probs

    # Make new data frames
    df100_alpha1 = copy.deepcopy(df100)
    df100_alpha1['success_p'] = transform_success_p(df100.success_p, 1)

    # Make histogram and save
    import matplotlib.pyplot as plt
    plt.hist(df100.success_p, alpha=1, bins=40, label='alpha=0')
    plt.hist(df100_alpha1.success_p, alpha=0.8, bins=40, label='alpha=1');
    plt.legend()
    plt.savefig('click_hist.png')

    # We re-split into train and eval sets because the original MIND dataset split by click date, we split by article
    np.random.seed(44)
    permute_idx = np.random.permutation( len(df100.article_id) )

    article_ids100 = [x for x in df100.article_id]
    permuted_ids100 = [ article_ids100[idx] for idx in permute_idx ]

    eval_size100 = int( np.round( len(df100.article_id)*0.2 ) )

    eval_articles100 = permuted_ids100[:eval_size100]
    train_articles100 = permuted_ids100[eval_size100:]

    print(len(train_articles100), len(eval_articles100))

    # News Data Train / Eval Split
    news_data_train100 = {k:v for k,v in news_data.items() if k in train_articles100}
    news_data_eval100 = {k:v for k,v in news_data.items() if k in eval_articles100}
    print( len(news_data_train100), len(news_data_eval100) )

    # Click Data Train / Eval Split
    df_train100_pd = df100.loc[ [ True if x in train_articles100 else False for x in df100.article_id ] ]
    df_eval100_pd = df100.loc[ [ True if x in eval_articles100 else False for x in df100.article_id ] ]

    df_train100 = df_train100_pd.set_index('article_id').to_dict(orient='index')
    df_eval100 = df_eval100_pd.set_index('article_id').to_dict(orient='index')

    print( len(df_train100), len(df_eval100) )

    # Click Data Train / Eval Split
    df_train100_alpha1_pd = df100_alpha1.loc[ [ True if x in train_articles100 else False for x in df100_alpha1.article_id ] ]
    df_eval100_alpha1_pd = df100_alpha1.loc[ [ True if x in eval_articles100 else False for x in df100_alpha1.article_id ] ]

    df_train100_alpha1 = df_train100_alpha1_pd.set_index('article_id').to_dict(orient='index')
    df_eval100_alpha1 = df_eval100_alpha1_pd.set_index('article_id').to_dict(orient='index')

    print( len(df_train100_alpha1), len(df_eval100_alpha1) )

    # Make dirs
    Path(save_dir + '/train/').mkdir(parents=True, exist_ok=True)
    Path(save_dir + '/eval/').mkdir(parents=True, exist_ok=True)
    
    # Save
    torch.save(news_data_train100, f'{save_dir}/train/news_data.pt')
    torch.save(news_data_eval100, f'{save_dir}/eval/news_data.pt')

    torch.save(df_train100_alpha1, f'{save_dir}/train/click_data_alpha=1.pt')
    torch.save(df_eval100_alpha1, f'{save_dir}/eval/click_data_alpha=1.pt')

    # Check lengths are the same
    assert len(df_eval100) == len(df_eval100_alpha1)
    assert len(df_train100) == len(df_train100_alpha1)

    print('Train and eval dataset lengths: ', len(news_data_train100), len(news_data_eval100) )

    # Check dinstinct articles in train vs eval split of articles + click datasets
    assert len( [ idx for idx in news_data_train100.keys() if idx in news_data_eval100.keys() ] ) == 0
    assert len( [ idx for idx in df_eval100.keys() if str(idx) in df_train100.keys() ] ) == 0


if __name__ == "__main__":
    main()
