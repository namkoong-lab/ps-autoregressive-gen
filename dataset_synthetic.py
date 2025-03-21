from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, Subset
import logging
from typing import Sequence, Dict

from util import set_seed


def get_split_complement(split_start, split_percentage):
    if split_start == 'first':
        complement_split_start = 'last'
    elif split_start == 'last':
        complement_split_start = 'first'
    else:
        raise ValueError('Argument split_start must be "first" or "last"')
    return complement_split_start, 100-split_percentage


@dataclass
class Synthetic_DataCollator_Sample(object):
    """
    Samples observations with replacement
    """
    num_clicks: 500

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        Z, click_rates = \
                tuple([instance[key] for instance in instances] \
                for key in ("Z", "click_rate"))
        
        Z = torch.concatenate([x.unsqueeze(0) for x in Z], 0)
        expanded_click_rates = torch.tensor(click_rates).unsqueeze(1).expand(len(click_rates), self.num_clicks)
        click_obs = torch.bernoulli(expanded_click_rates)
        
        return dict(
            Z = Z,
            click_obs = click_obs,
            click_rates = torch.tensor(click_rates),
            click_length_mask = torch.ones_like(click_obs)
        )


@dataclass
class Synthetic_DataCollator_Fixed(object):
    """
    Uses fixed observation click sequence
    """

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        Z, list_of_obs, click_rates = \
                tuple([instance[key] for instance in instances] \
                for key in ("Z", "click_obs", "click_rate"))
        Z = torch.concatenate([x.unsqueeze(0) for x in Z], 0)
        click_obs = torch.concatenate([x.unsqueeze(0) for x in list_of_obs], 0)
        return dict(
            Z = Z,
            click_obs = click_obs,
            click_rates = torch.tensor(click_rates),
            click_length_mask = torch.ones_like(click_obs)
        )


class VectorAndClickDataset(Dataset):
    """
    Dataset object that makes data loaders (can be used for train and/or eval)
    No resampling clicks, though we do permute rows when training
    """
    def __init__(self, dataset_file):
        dataset_data = torch.load(dataset_file)
        self.clicks = dataset_data['Y']
        self.click_rates = dataset_data['click_rate'].flatten().unsqueeze(1)
        self.Z = dataset_data['Z']
        assert len(self.Z) == len(self.clicks)
        logging.info(f"Total rows: {len(self.Z)}")
    def __len__(self):
        return len(self.click_rates)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
                click_obs = self.clicks[i],
                click_rates = self.click_rates[i],
                click_length_mask = torch.ones_like(self.clicks[i]),
                Z = self.Z[i])
    def make_loader(self, batch_size, train=False):
        dl = torch.utils.data.DataLoader(self,
               batch_size=batch_size,
               shuffle=train)
        return dl



class VectorAndClickRateDatasetFromDict(Dataset):
    """
    Dataset object that makes data loaders (can be used for train and/or eval)
    With resampling: takes click rates as input, and if we need deterministic 
        clicks for evals, those clicks are then generated during __init__
    """
    def __init__(self, click_rates, Z, 
                 split_start = None,
                 split_percentage = None,
                 num_loader_obs=500, 
                 generator_seed=230498,
                 bootstrap_seed=None):
        self.click_rates = click_rates
        self.Z = Z
        self.num_loader_obs = num_loader_obs
        logging.info(f"Total rows: {len(self.Z)}")
        
        # Generate a fixed sequence of observations (can be used for eval) =======================
        generator = torch.Generator()
        generator.manual_seed(generator_seed)
        self.loader_obs = []
        for click_rate in self.click_rates:
            self.loader_obs.append( torch.bernoulli(click_rate.repeat(self.num_loader_obs),
                                                    generator=generator) )

        if split_start is not None:
            self.split_start = split_start
            self.split_percentage = split_percentage
            self.orig_len = len(self.click_rates)
            new_len = int(len(self.click_rates)*split_percentage/100)
            logging.info(f'Training on {split_start} {split_percentage}% of examples')

            generator = torch.Generator()
            generator.manual_seed(generator_seed+10380)
            self.orig_click_rates = self.click_rates
            self.orig_Z = self.Z
            
            if self.split_start == "first":
                self.click_rates = self.click_rates[:new_len]
                self.Z = self.Z[:new_len]
            elif self.split_start == "last":
                self.click_rates = self.click_rates[-new_len:]
                self.Z = self.Z[-new_len:]
            else:
                raise ValueError('Argument split_start must be "first" or "last"')

        num_rows = len(self.Z)
        self.bootstrap_idxs = torch.arange(0, num_rows)
        if bootstrap_seed is not None:
            boot_generator = torch.Generator()
            boot_generator.manual_seed(bootstrap_seed+238954)
            self.bootstrap_idxs = torch.randint(high=num_rows, size=(num_rows,),
                    dtype=torch.int64, generator=boot_generator)
        # Make a fixed subset of the dataset (if necessary)
        # by first permuting the order of the articles, and then choosing the first rows
        # this is not fixed across different bootstrap seeds
        self.fixed_article_subset_order = torch.randperm(num_rows, generator=generator)

        
    def __len__(self):
        return len(self.click_rates)


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
                    click_rate = self.click_rates[self.bootstrap_idxs[i]],
                    click_obs = self.loader_obs[self.bootstrap_idxs[i]],
                    Z = self.Z[self.bootstrap_idxs[i]])


    def make_loader(self, batch_size, train=False, num_subset_rows=None, train_deterministic_row_order=False):
        if num_subset_rows is not None:
            # Take a subset of the number of rows
            idxs = self.fixed_article_subset_order[:num_subset_rows]
            ds = Subset(self, idxs)
        else:
            ds = self

        if train:
            collate_fn = Synthetic_DataCollator_Sample(
                                                  num_clicks=self.num_loader_obs)
        else:
            collate_fn = Synthetic_DataCollator_Fixed()

        if train and train_deterministic_row_order:
            dl = torch.utils.data.DataLoader(ds,
                   batch_size=batch_size,
                   collate_fn=collate_fn, shuffle=False)
        else:
            dl = torch.utils.data.DataLoader(ds,
                   batch_size=batch_size,
                   collate_fn=collate_fn, shuffle=train)
        return dl


class VectorAndClickRateDataset(VectorAndClickRateDatasetFromDict):
    def __init__(self, dataset_file, 
                 split_start = None,
                 split_percentage = None,
                 num_loader_obs=500,
                 generator_seed=230498,
                 bootstrap_seed=None):
        
        dataset_data = torch.load(dataset_file)
        # Consolidate data from articles into lists ===================
        # make sure click_rates is D x 1
        click_rates = dataset_data['click_rate'].flatten().unsqueeze(1)
        Z = dataset_data['Z']
        assert len(Z) == len(click_rates)
        super().__init__(click_rates, Z, split_start, split_percentage, num_loader_obs, generator_seed, bootstrap_seed)

def get_dataset_from_embed_dumps(model_dir, dataset_name='train', kwargs={}):
    model_path = model_dir + "/best_loss_row_embeds.pt"
    embeds = torch.load(model_path, map_location=torch.device('cpu'))
    model_path = model_dir + '/best_loss_predictions.pt'
    preds = torch.load(model_path, map_location=torch.device('cpu'))
    click_rates = preds[dataset_name]['click_rates']
    Z = embeds[dataset_name]
    ds = VectorAndClickRateDatasetFromDict(click_rates, Z, **kwargs)
    return ds

def get_loaders_synthetic(config, train_deterministic_row_order=False, extras=True):

    # Data split to train
    split_percentage = None
    split_start = None
    if not hasattr(config, 'datasplit'):
        setattr(config, 'datasplit', None)
    if not hasattr(config, 'bootstrap_seed'):
        setattr(config, 'bootstrap_seed', None)
    if config.datasplit is not None:
        split_start, split_percentage = config.datasplit.split("_")
        split_percentage = float(split_percentage)
    
    set_seed(config.seed)
    logging.info('Making train dataset')

    train_kwargs = {'split_start':split_start, 
            'split_percentage':split_percentage,
            'bootstrap_seed': config.bootstrap_seed}
    if not config.embed_data_dir:
        train_path = config.data_dir + '/train_data.pt'
        print(train_path)
        train_dataset = VectorAndClickRateDataset(train_path, **train_kwargs)
    else:
        train_dataset = get_dataset_from_embed_dumps(config.data_dir, 'train', train_kwargs)
    train_loader = train_dataset.make_loader(
        batch_size=config.batch_size, train=True, train_deterministic_row_order=train_deterministic_row_order)

    set_seed(config.seed)
    logging.info('Making eval dataset')
    if not config.embed_data_dir:
        eval_path = config.data_dir + '/eval_data.pt'
        eval_dataset = VectorAndClickRateDataset(eval_path)
    else:
        eval_dataset = get_dataset_from_embed_dumps(config.data_dir, 'val')

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

    res = {'train_loader':train_loader, 'val_loader':val_loader, 
            'train_fixed_subset_loader':train_fixed_subset_loader,
            'train_dataset': train_dataset,
            'val_dataset':eval_dataset}


    if config.extra_eval_data is not None:
        assert not config.embed_data_dir # not implemented
        extra_eval_dataset = VectorAndClickDataset(config.extra_eval_data)
        extra_eval_loader = extra_eval_dataset.make_loader(batch_size=config.batch_size, train=False)
        res['extra_eval_dataset'] = extra_eval_dataset
        res['extra_eval_loader'] = extra_eval_loader
    
    if extras:
        if config.datasplit is not None:
            if config.embed_data_dir:
                # no bootstrap seed here
                full_train_dataset = get_dataset_from_embed_dumps(config.data_dir, 'train')
            else:
                full_train_dataset = VectorAndClickRateDataset(train_path)
            full_train_loader = train_dataset.make_loader(
                batch_size=config.batch_size, train=False, train_deterministic_row_order=True)

            complement_split_start, complement_split_percentage = get_split_complement(split_start, split_percentage)
            if complement_split_percentage != 0:
                complement_kwargs = {'split_start':complement_split_start, 
                        'bootstrap_seed': config.bootstrap_seed,
                        'split_percentage':complement_split_percentage}

                if config.embed_data_dir:
                    complement_train_dataset = get_dataset_from_embed_dumps(config.data_dir, 'train', complement_kwargs)
                else:
                    complement_train_dataset = VectorAndClickRateDataset(train_path,
                            **complement_kwargs)
                complement_train_loader = complement_train_dataset.make_loader(
                    batch_size=config.batch_size, train=False, train_deterministic_row_order=True)
                res['complement_train_dataset'] = complement_train_dataset
                res['complement_train_loader'] = complement_train_loader
        
            res['full_train_dataset'] = full_train_dataset
            res['full_train_loader'] = full_train_loader

    return res


