import os
import torch
from torch.optim import Adam
from core.models import predrnn, predrnn_memory_decoupling

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn': predrnn.RNN,
            'predrnn_memory_decoupling': predrnn_memory_decoupling.RNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            ##  Use DataParallel
            # self.network = torch.nn.DataParallel(Network(self.num_layers, self.num_hidden, configs)).cuda().to(configs.device)
            ##  No DataParallel
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        ##  Use DataParallel
        # stats = torch.load(checkpoint_path)
        # self.network.load_state_dict(stats['net_param'])

        ##  No DataParallel
        stats = torch.load(checkpoint_path, map_location='cuda:0')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in stats['net_param'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.network.load_state_dict(new_state_dict)

    def train(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        next_frames, loss = self.network(frames_tensor, mask_tensor)
        loss.backward(loss.clone().detach())
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames, _ = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()