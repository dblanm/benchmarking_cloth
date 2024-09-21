import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, LSTM, ReLU
from torch_geometric.nn import DeepGCNLayer, TransformerConv

from .models import BaseModel


def rigid_icp(pc0, pc1):
    c0 = np.mean(pc0, axis=0)
    c1 = np.mean(pc1, axis=0)
    H = (pc0 - c0).transpose() @ (pc1 - c1)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t = c1 - R @ c0
    return R, t


class OcclusionFusion(BaseModel):
    def __init__(self, opt):
        super().__init__(self, opt)
        self._model = MotionCompleteNet()
        self.historical_motion = None
        self.historical_max_len = 16
        self.std_curr = None
        self.std_prev = None
        self.rigid_motion_curr = None

    def preprocess(self, frame_id):
        node_feature = np.load(
            os.path.join(self.input_path_node, "{:04d}.npy".format(frame_id))
        )
        node_pos = node_feature[:, :3]
        node_motion = node_feature[:, 3:6]
        visible = node_feature[:, -1] > 0.5

        pyd = np.load(
            os.path.join(self.input_path_graph, "{:04d}.npz".format(frame_id))
        )
        down_sample_idx1 = pyd["down_sample_idx1"]
        down_sample_idx2 = pyd["down_sample_idx2"]
        down_sample_idx3 = pyd["down_sample_idx3"]
        up_sample_idx1 = pyd["up_sample_idx1"]
        up_sample_idx2 = pyd["up_sample_idx2"]
        up_sample_idx3 = pyd["up_sample_idx3"]
        nn_index_l0 = pyd["nn_index_l0"]
        nn_index_l1 = pyd["nn_index_l1"]
        nn_index_l2 = pyd["nn_index_l2"]
        nn_index_l3 = pyd["nn_index_l3"]

        node_num_l0 = node_pos.shape[0]

        # extract rigid motion
        rigid_R, rigid_t = rigid_icp(
            node_pos[visible, :], node_pos[visible, :] + node_motion[visible, :]
        )
        self.rigid_motion_curr = (
            np.dot(node_pos, rigid_R.transpose()) + rigid_t - node_pos
        )
        nonrigid_motion = node_motion - self.rigid_motion_curr

        curr_motion = np.zeros(shape=(node_num_l0, 4))
        # motion in centimeter
        curr_motion[visible, :3] = nonrigid_motion[visible, :] * 100.0

        # normalize the motion
        self.curr_std = np.mean(np.std(curr_motion[visible, :3], axis=0)) + 0.1
        curr_motion[visible, :3] = curr_motion[visible, :3] / self.curr_std
        curr_motion[:, -1] = visible

        # init the mu of new nodes as 0.0, and the sigma of new nodes as a larger value (1.0)
        prev_motion = np.zeros(shape=(node_num_l0, 4))
        prev_motion[:, -1] = 1.0

        # for the first frame, set historical motion
        # using node position change between consequent frames as historical motion
        if frame_id > 1:
            node_feature_prev = np.load(
                os.path.join(self.input_path_node, "{:04d}.npy".format(frame_id - 1))
            )
            node_pos_prev = node_feature_prev[:, :3]
            visible_prev = node_feature_prev[:, -1] > 0.5
            prev_node_num = node_pos_prev.shape[0]

            # node num of current frame could be larger than the previous frame, and new nodes will be add to the end of the node array
            node_motion_prev = node_pos[: node_pos_prev.shape[0]] - node_pos_prev

            rigid_R, rigid_t = rigid_icp(
                node_pos_prev[visible_prev, :],
                node_pos_prev[visible_prev, :] + node_motion_prev[visible_prev, :],
            )
            rigid_motion_prev = (
                np.dot(node_pos_prev, rigid_R.transpose()) + rigid_t - node_pos_prev
            )
            prev_motion[:prev_node_num, :3] = (
                node_motion_prev - rigid_motion_prev
            ) * 100.0

        if self.historical_motion is None:
            self.historical_motion = np.zeros(shape=(1, node_num_l0, 4))
        else:
            seq_len = self.historical_motion.shape[0]
            prev_node_num = self.historical_motion.shape[1]
            drop = (seq_len == self.historical_max_len) * 1
            seq_len = min(seq_len + 1, self.historical_max_len)
            temp = np.zeros(shape=(seq_len, node_num_l0, 4))
            temp[:-1, :prev_node_num, :] = (
                self.historical_motion[drop:, :, :] * self.std_prev / self.curr_std
            )
            temp[-1, :prev_node_num, :] = prev_motion[:prev_node_num, :] / self.curr_std
            self.historical_motion = temp

        self.std_prev = self.curr_std

        node_pos = node_pos - np.mean(node_pos, axis=0)

        node_pos_torch = torch.from_numpy(node_pos.astype(np.float32)).to(self._device)
        curr_motion_torch = torch.from_numpy(curr_motion.astype(np.float32)).to(
            self._device
        )
        historical_motion_torch = torch.from_numpy(
            self.historical_motion.astype(np.float32)
        ).to(self._device)

        node_num, nn_num = nn_index_l0.shape
        edge_index_l0 = np.zeros(shape=(2, node_num * nn_num), dtype=np.int64)
        edge_index_l0[0:] = np.repeat(np.arange(node_num), nn_num)
        edge_index_l0[1:] = nn_index_l0.reshape(-1)

        node_num, nn_num = nn_index_l1.shape
        edge_index_l1 = np.zeros(shape=(2, node_num * nn_num), dtype=np.int64)
        edge_index_l1[0:] = np.repeat(np.arange(node_num), nn_num)
        edge_index_l1[1:] = nn_index_l1.reshape(-1)

        node_num, nn_num = nn_index_l2.shape
        edge_index_l2 = np.zeros(shape=(2, node_num * nn_num), dtype=np.int64)
        edge_index_l2[0:] = np.repeat(np.arange(node_num), nn_num)
        edge_index_l2[1:] = nn_index_l2.reshape(-1)

        node_num, nn_num = nn_index_l3.shape
        edge_index_l3 = np.zeros(shape=(2, node_num * nn_num), dtype=np.int64)
        edge_index_l3[0:] = np.repeat(np.arange(node_num), nn_num)
        edge_index_l3[1:] = nn_index_l3.reshape(-1)

        edge_index_l0 = torch.from_numpy(edge_index_l0).to(self._device)
        edge_index_l1 = torch.from_numpy(edge_index_l1).to(self._device)
        edge_index_l2 = torch.from_numpy(edge_index_l2).to(self._device)
        edge_index_l3 = torch.from_numpy(edge_index_l3).to(self._device)
        down_sample_idx1 = torch.from_numpy(
            np.array(down_sample_idx1).astype(np.int64)
        ).to(self._device)
        down_sample_idx2 = torch.from_numpy(
            np.array(down_sample_idx2).astype(np.int64)
        ).to(self._device)
        down_sample_idx3 = torch.from_numpy(
            np.array(down_sample_idx3).astype(np.int64)
        ).to(self._device)
        up_sample_idx1 = torch.from_numpy(np.array(up_sample_idx1).astype(np.int64)).to(
            self._device
        )
        up_sample_idx2 = torch.from_numpy(np.array(up_sample_idx2).astype(np.int64)).to(
            self._device
        )
        up_sample_idx3 = torch.from_numpy(np.array(up_sample_idx3).astype(np.int64)).to(
            self._device
        )

        return (
            node_pos_torch,
            curr_motion_torch,
            historical_motion_torch,
            [edge_index_l0, edge_index_l1, edge_index_l2, edge_index_l3],
            [down_sample_idx1, down_sample_idx2, down_sample_idx3],
            [up_sample_idx1, up_sample_idx2, up_sample_idx3],
        )

    def run_single_frame(self, frame_id):
        (
            node_pos,
            curr_motion,
            historical_motion,
            edge_indices,
            down_sample_indices,
            up_sample_indices,
        ) = self.preprocess(frame_id)

        with torch.no_grad():
            outputs = self.model(
                node_pos,
                curr_motion,
                historical_motion,
                edge_indices,
                down_sample_indices,
                up_sample_indices,
            )
        outputs = outputs.detach().cpu().numpy()
        mu = outputs[:, :3]
        sigma = outputs[:, -1]

        # eq.7 in the paper
        motion_scale = np.sqrt(np.sum(np.square(mu), axis=1))
        confidence = np.exp(-4 * np.square(sigma / (motion_scale + 1.0)))

        mu = mu * self.curr_std
        sigma = sigma * self.curr_std

        pred_motion = mu / 100.0
        node_motion = pred_motion + self.rigid_motion_curr

        return node_motion, confidence

    def __call__(self):
        frame_id = 0  # TODO: Remove, Temporal fix
        motion, confidence = self.run_single_frame(frame_id)


class MotionCompleteNet(torch.nn.Module):
    def __init__(self):
        super(MotionCompleteNet, self).__init__()
        feature_dim = 11
        hidden_channels = 32
        output_dim = 4
        self.hidden_channels = hidden_channels

        self.node_encoder = Linear(feature_dim, hidden_channels)

        self.lstm_layer_num = 2
        self.lstm_hidden_dim = 32
        self.lstm_output_dim = 4
        self.seq_encoder = LSTM(
            input_size=4,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_layer_num,
            batch_first=False,
        )
        self.seq_linear = Linear(self.lstm_hidden_dim, self.lstm_output_dim)

        self.conv0 = TransformerConv(hidden_channels, hidden_channels)

        self.layer11 = self.build_layer(hidden_channels)
        self.layer12 = self.build_layer(hidden_channels)
        self.layer21 = self.build_layer(hidden_channels)
        self.layer22 = self.build_layer(hidden_channels)
        self.layer31 = self.build_layer(hidden_channels)
        self.layer32 = self.build_layer(hidden_channels)
        self.layer41 = self.build_layer(hidden_channels)
        self.layer42 = self.build_layer(hidden_channels)
        self.layer51 = self.build_layer(hidden_channels * 2)
        self.layer52 = self.build_layer(hidden_channels * 2)
        self.layer61 = self.build_layer(hidden_channels * 3)
        self.layer62 = self.build_layer(hidden_channels * 3)
        self.layer71 = self.build_layer(hidden_channels * 4)
        self.layer72 = self.build_layer(hidden_channels * 4)

        self.norm_out = LayerNorm(hidden_channels * 4, elementwise_affine=True)
        self.act_out = ReLU(inplace=True)

        self.lin = Linear(hidden_channels * 4, output_dim)

    def build_layer(self, ch):
        conv = TransformerConv(ch, ch)
        norm = LayerNorm(ch, elementwise_affine=True)
        act = ReLU(inplace=True)
        layer = DeepGCNLayer(
            conv, norm, act, block="res+", dropout=0.1, ckpt_grad=False
        )
        return layer

    def forward(
        self,
        curr_pos,
        curr_motion,
        prev_motion,
        edge_indexes,
        down_sample_maps,
        up_sample_maps,
    ):
        node_num = curr_pos.shape[0]

        seq_feature, _ = self.seq_encoder(prev_motion.view(-1, node_num, 4), None)

        seq_pred = self.seq_linear(seq_feature[-1]).view(-1, self.lstm_output_dim)

        # the input feature of nodes
        x = self.node_encoder(torch.cat([curr_pos, seq_pred, curr_motion], dim=-1))

        feature0 = self.conv0(x, edge_indexes[0])
        feature1 = self.layer11(feature0, edge_indexes[0])
        feature1 = self.layer12(feature1, edge_indexes[0])

        feature2 = feature1[down_sample_maps[0]]
        feature2 = self.layer21(feature2, edge_indexes[1])
        feature2 = self.layer22(feature2, edge_indexes[1])

        feature3 = feature2[down_sample_maps[1]]
        feature3 = self.layer31(feature3, edge_indexes[2])
        feature3 = self.layer32(feature3, edge_indexes[2])

        feature4 = feature3[down_sample_maps[2]]
        feature4 = self.layer41(feature4, edge_indexes[3])
        feature4 = self.layer42(feature4, edge_indexes[3])

        feature5 = feature4[up_sample_maps[2]]
        feature5 = self.layer51(
            torch.cat([feature5, feature3], dim=-1), edge_indexes[2]
        )
        feature5 = self.layer52(feature5, edge_indexes[2])

        feature6 = feature5[up_sample_maps[1]]
        feature6 = self.layer61(
            torch.cat([feature6, feature2], dim=-1), edge_indexes[1]
        )
        feature6 = self.layer62(feature6, edge_indexes[1])

        feature7 = feature6[up_sample_maps[0]]
        feature7 = self.layer71(
            torch.cat([feature7, feature1], dim=-1), edge_indexes[0]
        )
        feature7 = self.layer72(feature7, edge_indexes[0])

        out = self.act_out(self.norm_out(feature7))
        out = F.dropout(out, p=0.1, training=self.training)

        pred = self.lin(out)

        # use softplus to make sigma positive
        pred[:, -1] = F.softplus(pred[:, -1])

        return pred
