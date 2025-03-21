from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, Subset
import logging
import transformers
from typing import Sequence, Dict
from collections import Counter
from transformers import DistilBertTokenizer
from util import set_seed


def get_split_complement(split_start, split_percentage):
    if split_start == 'first':
        complement_split_start = 'last'
    elif split_start == 'last':
        complement_split_start = 'first'
    else:
        raise ValueError('Argument split_start must be "first" or "last"')
    return complement_split_start, 100-split_percentage


def get_token_dict(batch):
    """Input: batch dictionary; Output: text token dict (or category ids)"""
    return {'input_ids':batch['text_token_ids'], 'attention_mask':batch['text_length_mask']}


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> list:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    return tokenized_list
    

@dataclass
class MIND_DataCollator_Sample(object):
    """
    Samples observations with replacement
    """
    tokenizer: transformers.PreTrainedTokenizer
    num_clicks: 500

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        text_token_ids, click_rates, category_ids = \
                tuple([instance[key] for instance in instances] \
                for key in ("text_token_ids", "click_rate", "category_ids"))
        
        text_token_ids = torch.nn.utils.rnn.pad_sequence(
            text_token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        expanded_click_rates = torch.tensor(click_rates).unsqueeze(1).expand(len(click_rates), self.num_clicks)
        click_obs = torch.bernoulli(expanded_click_rates)
        
        return dict(
            text_token_ids = text_token_ids,
            click_obs = click_obs,
            text_length_mask = text_token_ids.ne(self.tokenizer.pad_token_id),
            click_length_mask = torch.ones_like(click_obs),
            click_rates = torch.tensor(click_rates),
            category_ids = torch.tensor(category_ids)
        )


@dataclass
class MIND_DataCollator_Fixed(object):
    """
    Uses fixed observation click sequence
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        text_token_ids, list_of_obs, click_rates, category_ids = \
                tuple([instance[key] for instance in instances] \
                for key in ("text_token_ids", "click_obs", "click_rate", "category_ids"))
        text_token_ids = torch.nn.utils.rnn.pad_sequence(
            text_token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        click_obs = torch.concatenate([x.unsqueeze(0) for x in list_of_obs], 0)
        return dict(
            text_token_ids = text_token_ids,
            click_obs = click_obs,
            text_length_mask = text_token_ids.ne(self.tokenizer.pad_token_id),
            click_length_mask = torch.ones_like(click_obs),
            click_rates = torch.tensor(click_rates),
            category_ids = torch.tensor(category_ids)
        )


class TextAndClickRateDataset(Dataset):
    """
    Dataset object that makes data loaders (can be used for train and/or eval)
    """
    def __init__(self, news_path: str, clicks_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, sample_frac=1.0, 
                 num_loader_obs=500, 
                 min_obs_length=None,
                 category2id = None,
                 split_start = None,
                 split_percentage = None,
                 generator_seed=230498,
                 bootstrap_seed=None):
        
        logging.info("Loading data...")
        self.tokenizer = tokenizer
        click_data = torch.load(clicks_path)
        news_data = torch.load(news_path)
    
        article_ids = list(click_data.keys())

        if sample_frac < 1:
            # keep only a fraction of the article_ids (for debugging)
            old_len = len(article_ids)
            id_subset = torch.randperm(old_len)[:int(sample_frac * old_len)]
            article_ids = [x for (x,b) in zip(article_ids, id_subset) if b]            
            logging.info(f'Subsampling dataset by rows: before {old_len}, after {len(article_ids)}')

        elif split_start is not None:
            self.split_start = split_start
            self.split_percentage = split_percentage
            self.orig_len = len(article_ids)
            new_len = int(len(article_ids)*split_percentage/100)
            logging.info(f'Training on {split_start} {split_percentage}% of examples')

            generator = torch.Generator()
            generator.manual_seed(generator_seed+10380)

            all_ids_shuffled = torch.randperm(self.orig_len, generator=generator)
             
            if bootstrap_seed is not None:
                boot_generator = torch.Generator()
                boot_generator.manual_seed(bootstrap_seed+238954)
                dataset_size = len(all_ids_shuffled)
                boot_idx = torch.randint(high=dataset_size, size=(dataset_size,), dtype=torch.int64, generator=boot_generator)
                all_ids_shuffled = all_ids_shuffled[boot_idx]
                logging.info(f'Bootstrap resampling with seed {bootstrap_seed}')

            if self.split_start == "first":
                id_subset = all_ids_shuffled[:new_len]
            elif self.split_start == "last":
                id_subset = all_ids_shuffled[-new_len:]
            else:
                raise ValueError('Argument split_start must be "first" or "last"')
            article_ids = [article_id for idx, article_id in enumerate(article_ids) if idx in id_subset]  


        # Consolidate data from articles into lists ===================
        click_rates = []
        num_obs = []
        categories = []
        texts = []
        filtered_ids = []
        
        for k in article_ids:
            #if 'num_obs' in click_data[k]:
                # click data contains click rate
            num_obs_tmp = click_data[k]['num_obs']
            success_p = click_data[k]['success_p']

            #else:
                # click data contains sequence of clicks; get click rate by averaging click sequence
            #    num_obs_tmp = torch.tensor([x[1] for x in click_data[k]]).float()
            #    success_p = row_obs.mean()
            
            # Record selected articles
            click_rates.append(success_p)
            num_obs.append(num_obs_tmp)
            texts.append(news_data[k]['title']+': '+news_data[k]['abstract'])
            categories.append(news_data[k]['category'])
            filtered_ids.append(k)
       
        # Save article data ======================================= 
        self.article_ids = filtered_ids
        self.click_rates = torch.tensor(click_rates)
        self.num_obs = num_obs  # number of observations in MIND dataset
        self.categories = categories
        self.num_loader_obs = num_loader_obs # number of observations we are training with / generating

        if category2id is None:
            cat_counter = Counter(categories)
            cat_values = [cat for cat, cnt in cat_counter.items() if cnt >= 50]
            cat_values += ["other"]
            self.category2id = { cat : i for i, cat in enumerate( cat_values ) }
        else:
            self.category2id = category2id
        self.category_ids = [ self.category2id[cat] if cat in self.category2id \
                                else self.category2id['other'] for cat in categories ]

        logging.info(f"Total rows: {len(filtered_ids)}")
        logging.info("Tokenizing inputs. This may take some time...")
        token_list = _tokenize_fn(texts, tokenizer)
        self.text_token_ids = [x['input_ids'][0] for x in token_list]
        
        # Generate a fixed sequence of observations (can be used for eval) =======================
        generator = torch.Generator()
        generator.manual_seed(generator_seed)
        self.loader_obs = []
        for click_rate in self.click_rates:
            self.loader_obs.append( torch.bernoulli(click_rate.repeat(self.num_loader_obs),
                                                    generator=generator) )

        # Make a fixed subset of the dataset (if necessary)
        # by first permuting the order of the articles, and then choosing the first rows
        self.fixed_article_subset_order = torch.randperm(len(filtered_ids), generator=generator)
    

    def __len__(self):
        return len(self.article_ids)


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(text_token_ids = self.text_token_ids[i],
                    click_rate = self.click_rates[i],
                    click_obs = self.loader_obs[i],
                    category_ids = self.category_ids[i])


    def make_loader(self, batch_size, train=False, num_subset_rows=None, train_deterministic_row_order=False):
        if num_subset_rows is not None:
            # Take a subset of the number of rows
            idxs = self.fixed_article_subset_order[:num_subset_rows]
            ds = Subset(self, idxs)
        else:
            ds = self

        if train:
            collate_fn = MIND_DataCollator_Sample(self.tokenizer, 
                                                  num_clicks=self.num_loader_obs)
        else:
            collate_fn = MIND_DataCollator_Fixed(self.tokenizer)

        if train and train_deterministic_row_order:
            dl = torch.utils.data.DataLoader(ds,
                   batch_size=batch_size,
                   collate_fn=collate_fn, shuffle=False)
        else:
            dl = torch.utils.data.DataLoader(ds,
                   batch_size=batch_size,
                   collate_fn=collate_fn, shuffle=train)
        return dl



def get_loaders_MIND(config, train_deterministic_row_order=False, extras=False):

    # Use tokenizer even if not using text, used in shared dataloader
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_data_dir = config.data_dir + '/train/' 
    eval_data_dir = config.data_dir + '/eval/'

    if config.click_data_suffix is None or len(config.click_data_suffix) == 0:
        click_filename = f'click_data_alpha={config.transform_success_p_alpha}.pt'
    else:
        click_filename = f'click_data_alpha={config.transform_success_p_alpha}_{config.click_data_suffix}.pt'

    # Data split to train
    split_percentage = None
    split_start = None
    if not hasattr(config, 'datasplit'):
        setattr(config, 'datasplit', None)
    if config.datasplit is not None:
        split_start, split_percentage = config.datasplit.split("_")
        split_percentage = float(split_percentage)
    
    set_seed(config.seed)
    logging.info('Making train dataset')
    train_newspath = train_data_dir + 'news_data.pt'
    train_clickspath = train_data_dir + click_filename
    if hasattr(config, 'num_loader_obs_train'):
        num_loader_obs = config.num_loader_obs_train
    else:
        num_loader_obs = config.num_loader_obs
    bootstrap_seed = None
    if hasattr(config, 'bootstrap_seed'):
        bootstrap_seed = config.bootstrap_seed
    train_dataset = TextAndClickRateDataset(train_newspath, train_clickspath, 
            tokenizer, sample_frac=config.sample_frac, 
            num_loader_obs = num_loader_obs,
            split_start = split_start, split_percentage = split_percentage,
            bootstrap_seed = bootstrap_seed)

    train_loader = train_dataset.make_loader(
        batch_size=config.batch_size, train=True, train_deterministic_row_order=train_deterministic_row_order)

    set_seed(config.seed)
    logging.info('Making eval dataset')
    eval_newspath = eval_data_dir + 'news_data.pt'
    eval_clickspath = eval_data_dir + click_filename
    eval_dataset = TextAndClickRateDataset(eval_newspath, eval_clickspath, 
            tokenizer, sample_frac=config.sample_frac, 
            num_loader_obs = config.num_loader_obs,
            category2id = train_dataset.category2id) 
    val_loader = eval_dataset.make_loader(
            batch_size=config.eval_batch_size, train=False)

    print("train dataset size: {}".format(len(train_dataset)))
    print("eval dataset size: {}".format(len(eval_dataset)))

    # at every epoch, evaluate not only on the val set, but also a fixed subset of the training set
    # this is to measure overfitting
    
    train_subset_rows = len(eval_dataset)
    train_fixed_subset_loader = train_dataset.make_loader(
            batch_size=config.batch_size, 
            train=False, 
            num_subset_rows=train_subset_rows)

    res = {'tokenizer':tokenizer, 'train_loader':train_loader, 'val_loader':val_loader, 
            'train_fixed_subset_loader':train_fixed_subset_loader,
            'train_dataset': train_dataset,
            'val_dataset':eval_dataset, 
          'category2id': train_dataset.category2id}

    if extras:
        if config.datasplit is not None:
            full_train_dataset = TextAndClickRateDataset(train_newspath, train_clickspath, 
                    tokenizer, sample_frac=config.sample_frac, 
                    num_loader_obs = config.num_loader_obs)
            full_train_loader = full_train_dataset.make_loader(
                batch_size=config.batch_size, train=False, train_deterministic_row_order=True)

            complement_split_start, complement_split_percentage = get_split_complement(split_start, split_percentage)
            if complement_split_percentage != 0:
                complement_train_dataset = TextAndClickRateDataset(train_newspath, train_clickspath, 
                        tokenizer, sample_frac=config.sample_frac, 
                        num_loader_obs = config.num_loader_obs,
                        split_start = complement_split_start, split_percentage = complement_split_percentage)
                complement_train_loader = complement_train_dataset.make_loader(
                    batch_size=config.batch_size, train=False, train_deterministic_row_order=True)
                res['complement_train_dataset'] = complement_train_dataset
                res['complement_train_loader'] = complement_train_loader
        
            res['full_train_dataset'] = full_train_dataset
            res['full_train_loader'] = full_train_loader

    return res
