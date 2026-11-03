"""Reference-feature preparation shared by the paper timing benchmark."""
import torch

def make_ref_input(group):
    data = {k + '1': torch.from_numpy(v[()]).float().unsqueeze(0).cuda() for k, v in group.items() if k != 'image_size'}
    size = group['image_size'][()]
    data['image1'] = torch.empty((1, 1, int(size[1]), int(size[0])), device='cuda')
    return data
