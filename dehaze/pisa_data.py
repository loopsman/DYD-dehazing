"""Pair-uniform shuffled epochs with fully resumable sample order."""
import torch
from .mixed_haze import MixedHaze


class ShuffledPairs(MixedHaze):
    def __init__(self,sources,resolution,seed):
        super().__init__(sources,resolution)
        self.lookup=[(i,j) for i,d in enumerate(self.datasets) for j in range(len(d))]
        self.generator=torch.Generator().manual_seed(seed)
        self.order=[]
        self.cursor=0
        self.epoch=0

    def sample(self):
        if self.cursor==len(self.order):
            self.order=torch.randperm(len(self.lookup),generator=self.generator).tolist()
            self.cursor=0;self.epoch+=1
        i,j=self.lookup[self.order[self.cursor]]
        self.cursor+=1
        result=self.datasets[i][j]
        result['source']=self.sources[i]['name']
        return result

    def state_dict(self):
        return dict(order=list(self.order),cursor=self.cursor,epoch=self.epoch,generator=self.generator.get_state())

    def load_state_dict(self,state):
        self.order=list(state['order']);self.cursor=state['cursor'];self.epoch=state['epoch']
        self.generator.set_state(state['generator'])
