""" Differential geometry properties. Implements the computation of the 1st and
2nd order differential quantities of the 3D points given a UV coordinates and a
mapping f: R^{2} -> R^{3}, which takes a UV 2D point and maps it to a xyz 3D
point. The differential quantities are computed using analytical formulas
involving derivatives d_f/d_uv which are practically computed using Torch's
autograd mechanism. The computation graph is still built and it is possible to
backprop through the diff. quantities computation. The computed per-point
quantities are the following: normals, mean curvature, gauss. curvature.

Author: Jan Bednarik, jan.bednarik@epfl.ch
Date: 7.2.2020
"""

import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation


def loss_euc(output, target):
    return torch.mean(torch.linalg.vector_norm(output - target, dim=1))


def loss_frob(output, target, min_value=1e-10):
    return torch.mean(
        torch.linalg.matrix_norm((output - target).clamp_(min_value), dim=(1, 2))
    )


def metric(model, inputs):
    """calculate a jacobian tensor along a batch of inputs. returns something of size
    `batch_size` x `output_dim` x `input_dim`"""

    def _func_sum(inputs):
        return model(inputs).sum(dim=0)

    jac = ag.functional.jacobian(
        _func_sum, inputs, create_graph=True, vectorize=True
    ).permute(1, 0, 2)
    return torch.bmm(jac.transpose(1, 2), jac)


class DiffGeomProps(nn.Module):
    """Computes the differential geometry properties including normals,
    mean curvature, gaussian curvature, first fundamental form.
    Args:
        normals (bool): Whether to compute normals.
        curv_mean (bool): Whether to compute mean curvature.
        curv_gauss (bool): Whether to compute gaussian curvature.
        fff (bool): Whether to compute first fundamental form.
        gpu (bool): Whether to use GPU.
    """

    def __init__(
        self, device, normals=True, curv_mean=True, curv_gauss=True, fff=False
    ):
        nn.Module.__init__(self)

        self.device = device
        self._comp_normals = normals
        self._comp_cmean = curv_mean
        self._comp_cgauss = curv_gauss
        self._comp_fff = fff

    def forward(self, xyz, uv):
        """Computes the 1st and 2nd order derivative quantities, namely
        normals, mean curvature, gaussian curvature, first fundamental form.
        Args:
            xyz (torch.Tensor): 3D points, output 3D space (B, M, 3).
            uv (torch.Tensor): 2D points, parameter space, shape (B, M, 2).
        Returns:
            dict: Depending on `normals`, `curv_mean`, `curv_gauss`, `fff`
                includes normals, mean curvature, gauss. curvature and first
                fundamental form as torch.Tensor.
        """

        # Return values.
        ret = {}

        if not (
            self._comp_normals
            or self._comp_cmean
            or self._comp_cgauss
            or self._comp_fff
        ):
            return ret

        # Data shape.
        B, M = xyz.shape[:2]

        # 1st order derivatives d_fx/d_uv, d_fy/d_uv, d_fz/d_uv.
        dxyz_duv = []
        for o in range(3):
            derivs = self.df(xyz[:, :, o], uv)  # (B, M, 2)
            assert derivs.shape == (B, M, 2)
            dxyz_duv.append(derivs)

        # Jacobian, d_xyz / d_uv.
        J_f_uv = torch.cat(dxyz_duv, dim=2).reshape((B, M, 3, 2))

        # normals
        normals = F.normalize(
            torch.cross(J_f_uv[..., 0], J_f_uv[..., 1], dim=2), p=2, dim=2
        )  # (B, M, 3)
        assert normals.shape == (B, M, 3)

        # Save normals.
        if self._comp_normals:
            ret["normals"] = normals

        if self._comp_fff or self._comp_cmean or self._comp_cgauss:
            # 1st fundamental form (g)
            g = torch.matmul(J_f_uv.transpose(2, 3), J_f_uv)
            assert g.shape == (B, M, 2, 2)

            # Save first fundamental form, only E, F, G terms, instead of
            # the whole matrix [E F; F G].
            if self._comp_fff:
                ret["fff"] = g.reshape((B, M, 4))[:, :, [0, 1, 3]]  # (B, M, 3)

        if self._comp_cmean or self._comp_cgauss:
            # determinant of g.
            detg = g[:, :, 0, 0] * g[:, :, 1, 1] - g[:, :, 0, 1] * g[:, :, 1, 0]
            assert detg.shape == (B, M)

            # 2nd order derivatives, d^2f/du^2, d^2f/dudv, d^2f/dv^2
            d2xyz_duv2 = []
            for o in range(3):
                for i in range(2):
                    deriv = self.df(dxyz_duv[o][:, :, i], uv)  # (B, M, 2)
                    assert deriv.shape == (B, M, 2)
                    d2xyz_duv2.append(deriv)

            d2xyz_du2 = torch.stack(
                [d2xyz_duv2[0][..., 0], d2xyz_duv2[2][..., 0], d2xyz_duv2[4][..., 0]],
                dim=2,
            )  # (B, M, 3)
            d2xyz_dudv = torch.stack(
                [d2xyz_duv2[0][..., 1], d2xyz_duv2[2][..., 1], d2xyz_duv2[4][..., 1]],
                dim=2,
            )  # (B, M, 3)
            d2xyz_dv2 = torch.stack(
                [d2xyz_duv2[1][..., 1], d2xyz_duv2[3][..., 1], d2xyz_duv2[5][..., 1]],
                dim=2,
            )  # (B, M, 3)
            assert d2xyz_du2.shape == (B, M, 3)
            assert d2xyz_dudv.shape == (B, M, 3)
            assert d2xyz_dv2.shape == (B, M, 3)

            # Each (B, M)
            gE, gF, _, gG = g.reshape((B, M, 4)).permute(2, 0, 1)
            assert gE.shape == (B, M)

        # Compute mean curvature.
        if self._comp_cmean:
            cmean = (
                torch.sum(
                    (-normals / detg[..., None])
                    * (
                        d2xyz_du2 * gG[..., None]
                        - 2.0 * d2xyz_dudv * gF[..., None]
                        + d2xyz_dv2 * gE[..., None]
                    ),
                    dim=2,
                )
                * 0.5
            )
            ret["cmean"] = cmean

        # Compute gaussian curvature.
        if self._comp_cgauss:
            iiL = torch.sum(d2xyz_du2 * normals, dim=2)
            iiM = torch.sum(d2xyz_dudv * normals, dim=2)
            iiN = torch.sum(d2xyz_dv2 * normals, dim=2)
            cgauss = (iiL * iiN - iiM.pow(2)) / (gE * gF - gG.pow(2))
            ret["cgauss"] = cgauss

        return ret

    def df(self, x, wrt):
        B, M = x.shape
        return ag.grad(
            x.flatten(),
            wrt,
            grad_outputs=torch.ones(B * M, dtype=torch.float32).to(self.device),
            create_graph=True,
            retain_graph=True,
        )[0]


class Mapping2Dto3D(nn.Module):
    """
    Core Atlasnet Function.
    Takes batched points as input and run them through an MLP.
    Note : the MLP is implemented as a torch.nn.Conv1d with kernels of size 1 for speed.
    Note : The latent vector is added as a bias after the first layer.
        Note that this is strictly identical as concatenating each input point with the
        latent vector but saves memory and speeed.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(
        self,
        bottleneck_size=1024,
        dim_template=2,
        hidden_neurons=512,
        num_layers=2,
        activation="softplus",
    ):
        self.bottleneck_size = bottleneck_size
        self.input_size = dim_template
        self.dim_output = 3
        self.hidden_neurons = hidden_neurons
        self.num_layers = num_layers
        super(Mapping2Dto3D, self).__init__()
        print(
            f"New MLP decoder : hidden size {hidden_neurons}, num_layers {num_layers},"
            f" activation {activation}"
        )

        self.conv1 = torch.nn.Conv1d(self.input_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)

        self.conv_list = nn.ModuleList(
            [
                torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1)
                for i in range(self.num_layers)
            ]
        )

        self.last_conv = torch.nn.Conv1d(self.hidden_neurons, self.dim_output, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_neurons)

        self.bn_list = nn.ModuleList(
            [torch.nn.BatchNorm1d(self.hidden_neurons) for i in range(self.num_layers)]
        )

        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(self.bn1(x))
        x = self.activation(self.bn2(self.conv2(x)))
        for i in range(self.num_layers):
            x = self.activation(self.bn_list[i](self.conv_list[i](x)))
        return self.last_conv(x)


class Atlasnet(nn.Module):
    def __init__(self, device):
        """
        Core Atlasnet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and
        returns a pointcloud or a mesh
        Author : Thibault Groueix 01.11.2019
        """
        super(Atlasnet, self).__init__()
        # Intialize deformation networks
        self.decoder = Mapping2Dto3D().to(device)

        # Geom props from Yan Bednarik
        self.dgp = DiffGeomProps(
            device, normals=False, curv_mean=False, curv_gauss=False, fff=True
        )

    def forward(self, input_points, compute_fff=False):
        # Input has shape (# batch, # sampled points, 2)
        if compute_fff:
            # Require grad if 1st fundamental form has to be computed
            input_points.requires_grad = True

        # Decoder takes (# batch, 2, # sampled points)
        output_points = self.decoder(input_points.transpose(1, 2)).transpose(1, 2)

        fff = self.dgp(output_points, input_points)["fff"] if compute_fff else None

        return output_points, fff


class MLP(nn.Module):
    def __init__(self, hidden_size1=128, hidden_size2=256, hidden_size3=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, hidden_size1)
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = torch.nn.Linear(hidden_size3, 3)
        self.activation_fn = nn.Softplus()

    def forward(self, x):
        hidden1 = self.activation_fn(self.fc1(x))
        hidden2 = self.activation_fn(self.fc2(hidden1))
        hidden3 = self.activation_fn(self.fc3(hidden2))
        output = self.fc4(hidden3)
        return output
