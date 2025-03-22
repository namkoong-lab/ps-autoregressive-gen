import torch
from dataset_MIND import get_token_dict
from torch.utils.data import Dataset, DataLoader

class DictOfListsDataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary where values are lists of equal length.
        """
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.length = len(next(iter(data_dict.values())))  # Length of the lists

    def __len__(self):
        # The length of the dataset is the length of the lists in the dict
        return self.length

    def __getitem__(self, idx):
        # Get the ith elements of each list in the dict
        item = {key: self.data_dict[key][idx] for key in self.keys}
        return item

def get_dpt_history_loader(loc, batch_size, is_train=True):
    data = torch.load(loc)
    dataset = DictOfListsDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return dataset, loader

def eval_all_Zreps(model, loader, device, click_rates=False):
    model.eval()
    with torch.no_grad():
        all_Zreps_list = []
        all_click_rates = []
        for batch in loader:
            for k,v in batch.items():
                batch[k] = v.to(device)
            zreps = model.z_encoder(get_token_dict(batch)).detach()
            all_Zreps_list.append(zreps)
            all_click_rates.append(batch['click_rates'])
    if click_rates:
        return torch.cat(all_Zreps_list,0), torch.cat(all_click_rates)
    else: return torch.cat(all_Zreps_list,0)
    
def get_val_loss_dpt(model, hist_loader, device, num_arms, all_Zreps=None):
    if all_Zreps is not None:
        all_Zreps.to(device)
    model.eval()
    all_preds = []
    got_best_arm_count = 0
    total_envs = 0
    total_loss = 0.0
    total_obs = 0
    with torch.no_grad():
        for batch in hist_loader:
            # currently works only on CPU, need to change datasets a bit...
            #for k,v in batch.items():
            #    batch[k] = v.to(device)       
            prev_Ys = batch['env']['Y'].to(device)
            num_envs, batch_size, T = prev_Ys.shape
            prev_Ys = prev_Ys.view(num_envs * num_arms, -1)

            # generate / simulate arm counts, using dirichlet thing from DPT
            arm_counts = torch.zeros(num_envs, batch_size)

            if all_Zreps is not None:
                these_arms = batch['env']['arm_idxs'].to(device)
                encoded_Z = all_Zreps[these_arms.flatten()].view(num_envs * num_arms, -1)
            else:
                # get preds
                encoded_Z = model.z_encoder(batch['env']['Z'].to(device)).view(num_envs * num_arms, -1)
            H = batch['hist'].shape[1]
            # go through history
            for h in range(H):
                # Create a range tensor of shape (T) which will be broadcasted
                t_values = torch.arange(T).unsqueeze(0).unsqueeze(0)

                # Compare each entry in arm_counts with the range of t_values
                # arm_counts[..., None] adds a dimension to match shape for broadcasting
                click_mask = (arm_counts.unsqueeze(-1) > t_values).int().view(num_envs * num_arms, -1).to(device)
                pred = model.get_p_pred(encoded_Z, prev_Ys, click_mask)

                # shape into num_envs x num_arms 
                preds_reshaped = pred.view(num_envs, num_arms)
                targets = batch['env']['best_arm'].to(device)

                total_loss += torch.nn.functional.cross_entropy(preds_reshaped,targets,reduction='sum').item()
                total_obs += num_envs
                
                # update history
                next_arms = batch['hist'][:,h].flatten().type(torch.int64)
                next_arms_onehot = torch.zeros_like(arm_counts)
                next_arms_onehot.scatter_(1, next_arms.unsqueeze(1), 1)
                arm_counts += next_arms_onehot
                all_preds.append(preds_reshaped.cpu())
            total_envs += num_envs
            last_selected_arms = preds_reshaped.argmax(1).cpu()
            if 'best_arm_onehot' in batch['env']:
                got_best_arm_count += (last_selected_arms == batch['env']['best_arm']).float().sum()
            else:
                got_best_arm_count += (last_selected_arms == batch['env']['best_arm'].argmax(1)).float().sum()
    
    return {'loss': total_loss / total_obs, 'total_loss': total_loss, 'total_obs': total_obs, 'all_preds': all_preds, 'num_eventually_correct': got_best_arm_count, 'total_envs': total_envs, 'proportion_eventually_correct': got_best_arm_count / total_envs}
             
