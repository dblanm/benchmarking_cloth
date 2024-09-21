import os
from argparse import Namespace

import torch

from .raft.raft import RAFT
from .raft.utils.utils import forward_interpolate, InputPadder
from .utils import get_default_device


class Model:
    def __init__(self):
        args = Namespace(
            model=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "checkpoints",
                "raft-things.pth",
            ),
            small=False,
            mixed_precision=False,
        )
        self._device = get_default_device()
        self._model = torch.nn.DataParallel(RAFT(args))
        self._model.load_state_dict(torch.load(args.model))
        self._model.eval()
        self._model.to(self._device)
        self.iters = 32
        self.prev_flow = None

    def reset_flow(self):
        self.flow_init = None

    def __call__(self, image1, image2):
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(
            image1[None].to(self._device), image2[None].to(self._device)
        )

        with torch.no_grad():
            flow_low, flow_pr = self._model(
                image1,
                image2,
                flow_init=self.prev_flow,
                iters=self.iters,
                test_mode=True,
            )
            self.prev_flow = forward_interpolate(flow_low[0])[None]
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
        return flow
