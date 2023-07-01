from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_fast
import numpy as np
import scipy
from sklearn.cross_decomposition import PLSRegression

from .compute_u import compute_u
from .compute_v_dama import compute_v_dama
from .rome_hparams import DAMAHyperParams
from .rome_main import get_context_templates



def apply_dama_on_module(module, P, mu_in, mu_out):


    # Apply DAME on the module
    new_module = deepcopy(module)

    # TODO - apply DAME on the module
    raise NotImplementedError
    return new_module

# TODO - INLP functions store together with INLP code
def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T)  # orthogonal basis

    w_basis * np.sign(w_basis[0][0])  # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace

    return P_W

# TODO: this is just for refernce shouldn't be used here
def project_with_pls(x: np.ndarray, P: np.ndarray, mu: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    :param x: ndarray, input vector
    :param P: ndarray, projection matrix
    :param mu: ndarray, mean of training data
    :param s: ndarray, std of training data
    :return: ndarray, projected vector
    """
    # TODO: are operations involving s necessary?
    x = (x - mu) / s
    x = np.dot(x, P)
    x = (x * s) + mu

    return x


def apply_dama(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DAMAHyperParams,
    copy=False,
    return_orig_module=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:

    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    if copy:
        model = deepcopy(model)

    module_copy = {}

    # Dama is applied for all requests together to improve generatlzation
    projections = execute_dama(model, tok, requests, hparams)

    with torch.no_grad():
        for m_name, (P, mu_in, mu_out) in projections.items():

            orig_module = nethook.get_module(model, m_name)
            new_module = apply_dama_on_module(orig_module, P, mu_in, mu_out)

            if return_orig_module and m_name not in module_copy:
                module_copy[m_name] = orig_module.detach().clone()

            nethook.set_module(model, m_name, new_module)

        print(f"New weights successfully inserted into {list(projections.keys())}")

    return model, module_copy


def execute_dama(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: Dict,
        hparams: DAMAHyperParams,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

    # # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # # Save old weights for future restoration
    # weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute u and v
    # Update loop: sequentially intervene at each specified layer
    projections = {}

    l_vec_list = []
    r_vec_list = []
    for layer in sorted(hparams.layers):
        for request in requests:
            # Compute rank-1 update matrix
            left_vector: torch.Tensor = compute_u(
                model,
                tok,
                request,
                hparams,
                layer,
                get_context_templates(model, tok, hparams.context_template_length_params),
            )
            print("Left vector shape:", left_vector.shape)
            l_vec_list.append(left_vector)

            # compute v vectors for each PRONOUN option
            right_contrast_vector: torch.Tensor = compute_v_dama(
                model,
                tok,
                request,
                hparams,
                layer,
                left_vector,
                get_context_templates(model, tok, hparams.context_template_length_params),
            )
            print("Right vector shape:", right_contrast_vector.shape)
            r_vec_list.append(right_contrast_vector)


        with torch.no_grad():
            module_name = f"{hparams.rewrite_module_tmp.format(layer)}"
            # detach the weights from the grapha and convert to numpy
            U = torch.stack(l_vec_list, dim=0).cpu().numpy()
            V = torch.stack(r_vec_list, dim=0).cpu().numpy()

            W = weights[f"{hparams.rewrite_module_tmp.format(layer)}"].cpu().numpy()

            u_dim = U.shape[1]
            # multiply V by pseudo inverse of W
            U_hat = np.matmul(V, np.linalg.pinv(W))


            # compute PLS mapping between U and U_hat
            pls = PLSRegression(n_components=hparams.nullspace_dimension, scale=False)
            pls.fit(U, U_hat)

            B = pls.x_weights_[:,:hparams.projection_components] # not needed but maybe useful to get some statistics
            P = np.eye(u_dim, u_dim) - get_rowspace_projection(B.T)
            # TODO: maybe use global statistics to compute mu_s
            mu_in = pls._x_mean
            mu_out = W @ pls._x_mean

            # save as tensors
            if torch.cuda.is_available():
                P = torch.tensor(P, dtype=torch.float16, device='cuda')
                mu_in = torch.tensor(mu_in, dtype=torch.float16, device='cuda')
                mu_out = torch.tensor(mu_out, dtype=torch.float16, device='cuda')
            else:
                P = torch.tensor(P, dtype=torch.float32, device='cpu')
                mu_in = torch.tensor(mu_in, dtype=torch.float32, device='cpu')
                mu_out = torch.tensor(mu_out, dtype=torch.float32, device='cpu')

            projections[module_name] = (P, mu_in, mu_out)

        print(f"Projections successfully computed for layer {list(projections.keys())}")
    return projections



