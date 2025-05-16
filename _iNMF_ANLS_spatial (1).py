import lazy_loader as lazy
np = lazy.load("numpy", error_on_import=True)
from tqdm import tqdm
from ._utilities import nnlsm_blockpivot

def optimize_ALS(
    liger_object,
    k: int,
    value_lambda: float = 5.0,
    spatial_gamma: float = 0.0,
    bi_gamma: float = 0.0,
    L_list=None,
    thresh: float = 1e-6,
    max_iters: int = 30,
    nrep: int = 1,
    H_init=None,
    W_init=None,
    V_init=None,
    rand_seed: int = 1,
    print_obj: bool = False,
):
    """
    Perform iNMF with optional Laplacian and bi-Laplacian smoothing via ALS.
    bi_gamma: weight of the L^2 penalty Tr(H^T L^2 H) = ||L H||_F^2
    """
    N = liger_object.num_samples
    ns = [adata.shape[0] for adata in liger_object.adata_list]
    G = len(liger_object.var_genes)
    X = [adata.layers["scale_data"].toarray() for adata in liger_object.adata_list]

    if k >= min(ns):
        raise ValueError(f"k must be < smallest dataset size ({min(ns)})")
    if k >= G:
        raise ValueError(f"k must be < number of genes ({G})")

    best_obj = np.inf
    best_seed = None

    for rep in range(nrep):
        # seed for reproducibility
        np.random.seed(rand_seed + rep)

        # precompute sqrt penalty factors
        sqrt_lambda = np.sqrt(value_lambda)
        sqrt_s = np.sqrt(spatial_gamma)
        sqrt_b = np.sqrt(bi_gamma)

        # precompute half-Laplacians for spatial and bi-terms
        L_half_list  = [None]*N
        L2_half_list = [None]*N
        if L_list is not None:
            for i, Li in enumerate(L_list):
                # eigen-decompose Li = Q Λ Q^T
                w, Q = np.linalg.eigh(Li)
                # first-order half: L^{1/2}
                L_half_list[i]  = Q @ np.diag(np.sqrt(np.clip(w, 0, None))) @ Q.T
                # second-order half: sqrt(L^2) = Q Λ Q^T = Li
                # but to maintain symmetry, rebuild via Q,Λ
                L2_half_list[i] = Q @ np.diag(w) @ Q.T

        # initialize factors
        W = np.abs(np.random.uniform(0, 2, (k, G)))
        V = [np.abs(np.random.uniform(0, 2, (k, G))) for _ in range(N)]
        H = [np.abs(np.random.uniform(0, 2, (ns[i], k))) for i in range(N)]
        if W_init is not None: W = W_init
        if V_init is not None: V = V_init
        if H_init is not None: H = H_init

        # compute initial objective
        obj = 0.0
        for i in range(N):
            # reconstruction + V-penalty
            obj += np.linalg.norm(X[i] - H[i] @ (W + V[i]))**2
            obj += value_lambda * np.linalg.norm(H[i] @ V[i])**2
            # spatial Laplacian penalty
            if spatial_gamma > 0:
                obj += spatial_gamma * np.trace(H[i].T @ L_list[i] @ H[i])
            # bi-Laplacian penalty
            if bi_gamma > 0:
                obj += bi_gamma * np.linalg.norm(L_list[i] @ H[i])**2

        # ALS loop
        delta = np.inf
        for _ in tqdm(range(max_iters), desc=f"ALS rep {rep+1}"):
            if delta <= thresh:
                break

            # 1) update H
            for i in range(N):
                n_i = ns[i]
                # build base blocks
                A_blocks = [
                    (W + V[i]).T,                # (G × k)
                    (sqrt_lambda * V[i]).T       # (G × k)
                ]
                B_blocks = [
                    X[i].T,                      # (G × n_i)
                    np.zeros((G, n_i))           # (G × n_i)
                ]

                # add first-order Laplacian
                if spatial_gamma > 0:
                    A_blocks.append(sqrt_s * L_half_list[i][:, :k])   # (n_i × k)
                    B_blocks.append(np.zeros((n_i, n_i)))             # (n_i × n_i)

                # add second-order bi-Laplacian
                if bi_gamma > 0:
                    A_blocks.append(sqrt_b * L2_half_list[i][:, :k])  # (n_i × k)
                    B_blocks.append(np.zeros((n_i, n_i)))             # (n_i × n_i)

                # stack and solve NNLS
                A_h = np.vstack(A_blocks)  # ((2G + penalties*n_i) × k)
                B_h = np.vstack(B_blocks)  # ((2G + penalties*n_i) × n_i)
                H[i] = nnlsm_blockpivot(A_h, B_h)[0].T

            # 2) update V
            for i in range(N):
                A_v = np.vstack([H[i], sqrt_lambda * H[i]])           # ((2 n_i) × k)
                B_v = np.vstack([X[i] - H[i] @ W, np.zeros((ns[i], G))])  # ((2 n_i) × G)
                V[i] = nnlsm_blockpivot(A_v, B_v)[0]

            # 3) update W
            A_w = np.vstack(H)                                          # ((∑n_i) × k)
            B_w = np.vstack([X[i] - H[i] @ V[i] for i in range(N)])     # ((∑n_i) × G)
            W = nnlsm_blockpivot(A_w, B_w)[0]

            # recompute objective
            prev_obj = obj
            obj = 0.0
            for i in range(N):
                obj += np.linalg.norm(X[i] - H[i] @ (W + V[i]))**2
                obj += value_lambda * np.linalg.norm(H[i] @ V[i])**2
                if spatial_gamma > 0:
                    obj += spatial_gamma * np.trace(H[i].T @ L_list[i] @ H[i])
                if bi_gamma > 0:
                    obj += bi_gamma * np.linalg.norm(L_list[i] @ H[i])**2
            delta = abs(prev_obj - obj) / ((prev_obj + obj) / 2)

        # track best
        if obj < best_obj:
            best_obj = obj
            best_H, best_W, best_V = H, W, V
            best_seed = rand_seed + rep
        if print_obj:
            print(f"Rep {rep+1}: obj={best_obj:.4e}, seed={best_seed}")

    # save best factors
    for i in range(N):
        liger_object.adata_list[i].obsm["H"] = best_H[i]
        liger_object.adata_list[i].varm["W"] = best_W.T
        liger_object.adata_list[i].varm["V"] = best_V[i].T

    return liger_object
