import os
import torch


def build_text_basis(base_text_features, rank=128, rank_ref=8, eps=1e-6):
    """
    base_text_features: [K, D]
    return:
        B_T: [D, r]
        singular_values: [min(D, K)]
        rank: int
    """
    T = base_text_features.float()          # [K, D]
    U, S, Vh = torch.linalg.svd(T.t(), full_matrices=False)  # [D, K] = U diag(S) Vh
    rank = min(int((S > eps).sum().item()), rank)
    B_T = U[:, :rank].contiguous()          # [D, r]
    B_ref = U[:, :rank_ref].contiguous()          # [D, r]
    
    # singular-value energy
    S2 = S.pow(2)
    effective_rank = int((S > eps).sum().item())
    total_energy = S2.sum()
    effective_energy = S2[:effective_rank].sum()
    BT_energy = S2[:rank].sum()
    Bref_energy = S2[:rank_ref].sum()

    energy_stats = {
        "effective_rank": effective_rank,
        "rank_BT": rank,
        "rank_Bref": rank_ref,

        # absolute energies
        "total_energy": total_energy.item(),
        "BT_energy": BT_energy.item(),
        "Bref_energy": Bref_energy.item(),

        # ratios
        "BT_energy_ratio_total": (BT_energy / total_energy).item(),
        "Bref_energy_ratio_total": (Bref_energy / total_energy).item(),

        # relative to effective non-zero spectrum
        "BT_energy_ratio_effective": (BT_energy / effective_energy).item(),
        "Bref_energy_ratio_effective": (Bref_energy / effective_energy).item(),

        # Bref relative to B_T
        "Bref_energy_ratio_BT": (Bref_energy / BT_energy).item(),

        # useful spectrum diagnostics
        "top_singular_values": S[:min(10, len(S))].detach().cpu().tolist(),
        "cumulative_energy_top10": (
            torch.cumsum(S2, dim=0)[:min(10, len(S))] / total_energy
        ).detach().cpu().tolist(),
    }

    print("==== Text Basis Energy Report ====")
    print(f"Effective rank: {effective_rank}")
    print(f"B_T rank:       {rank}")
    print(f"B_ref rank:     {rank_ref}")
    print("----------------------------------")
    print(f"B_T energy / total:    {energy_stats['BT_energy_ratio_total'] * 100:.4f}%")
    print(f"B_ref energy / total:  {energy_stats['Bref_energy_ratio_total'] * 100:.4f}%")
    print(f"B_ref energy / B_T:    {energy_stats['Bref_energy_ratio_BT'] * 100:.4f}%")
    print("----------------------------------")
    print("Top singular values:")
    for i, s in enumerate(energy_stats["top_singular_values"]):
        print(f"  σ[{i+1}] = {s:.6f}")

    print("Cumulative energy of top components:")
    for i, e in enumerate(energy_stats["cumulative_energy_top10"]):
        print(f"  top-{i+1}: {e * 100:.4f}%")
            
    return B_T, S, rank, B_ref


def build_reference_text_basis(ref_text_features, rank=128, eps=1e-6, normalize=True, center=False):
    """
    ref_text_features: [N_ref, D]

    Return:
        B_ref: [D, r]
        singular_values: [min(D, N_ref)]
        selected_rank: int
    """
    X = ref_text_features.float()  # [N_ref, D]

    # if normalize:
    #     X = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # if center:
    #     X = X - X.mean(dim=0, keepdim=True)

    # Same convention as build_text_basis:
    # X.T: [D, N_ref]
    U, S, Vh = torch.linalg.svd(X.t(), full_matrices=False)

    numerical_rank = int((S > eps).sum().item())
    selected_rank = min(rank, numerical_rank, U.shape[1])

    B_ref = U[:, :selected_rank].contiguous()  # [D, selected_rank]

    # Optional numerical safety
    B_ref, _ = torch.linalg.qr(B_ref, mode="reduced")

    print(f">> Built B_ref with shape {B_ref.shape}")
    print(f">> Numerical rank = {numerical_rank}, selected rank = {selected_rank}")
    print(f">> Top-10 singular values: {S[:10].detach().cpu().numpy()}")
    
    # singular-value energy
    S2 = S.pow(2)
    effective_rank = int((S > eps).sum().item())
    total_energy = S2.sum()
    effective_energy = S2[:effective_rank].sum()
    Bref_energy = S2[:rank].sum()

    energy_stats = {
        "effective_rank": effective_rank,
        "rank_Bref": rank,

        # absolute energies
        "total_energy": total_energy.item(),
        "Bref_energy": Bref_energy.item(),

        # ratios
        "Bref_energy_ratio_total": (Bref_energy / total_energy).item(),

        # relative to effective non-zero spectrum
        "Bref_energy_ratio_effective": (Bref_energy / effective_energy).item(),

        # useful spectrum diagnostics
        "top_singular_values": S[:min(10, len(S))].detach().cpu().tolist(),
        "cumulative_energy_top10": (
            torch.cumsum(S2, dim=0)[:min(10, len(S))] / total_energy
        ).detach().cpu().tolist(),
    }

    print("==== Text Basis Energy Report ====")
    print(f"Effective rank: {effective_rank}")
    print(f"B_ref rank:     {rank}")
    print("----------------------------------")
    print(f"B_ref energy / total:  {energy_stats['Bref_energy_ratio_total'] * 100:.4f}%")
    print("----------------------------------")

    return B_ref.cpu(), S, selected_rank


def compute_probe_residual(probe_weight, B_T):
    """
    probe_weight: [K, D]
    B_T: [D, r]
    return:
        W_res: [K, D]
    """
    W = probe_weight.float()
    proj = W @ B_T @ B_T.t()
    W_res = W - proj
    return W_res


# def build_residual_basis(W_res, k):
#     """
#     W_res: [K, D]
#     k: int
#     return:
#         B_R: [D, k]
#         singular_values: [...]
#     """
#     U, S, Vh = torch.linalg.svd(W_res.t(), full_matrices=False)  # [D, K]
#     k = min(k, U.size(1))
#     B_R = U[:, :k].contiguous()                                  # [D, k]
#     return B_R, S

def build_residual_basis(
    W_res,
    k=None,
    energy_thresh=0.95,
    auto_rank=True,
    min_rank=4,
    max_rank=64
):
    """
    W_res: [K, D]
    return:
        B_R: [D, k]
        singular_values: [m]
        k: int
        cumulative_energy: [m]
    """
    U, S, Vh = torch.linalg.svd(W_res.t(), full_matrices=False)  # [D, K]

    if auto_rank:
        k, cumulative_energy = choose_rank_by_energy(
            S,
            energy_thresh=energy_thresh,
            min_rank=min_rank,
            max_rank=max_rank
        )
    else:
        assert k is not None
        k = min(k, U.size(1))
        cumulative_energy = torch.cumsum(S.float() ** 2, dim=0) / (S.float() ** 2).sum().clamp_min(1e-12)

    B_R = U[:, :k].contiguous()
    return B_R, S, k, cumulative_energy

def save_subspace_cache(save_path, B_T, B_R, text_rank, residual_singular_values, selected_rank, cumulative_energy, meta=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "B_T": B_T.cpu(),
        "B_R": B_R.cpu(),
        "text_rank": text_rank,
        "residual_singular_values": residual_singular_values.cpu(),
        "selected_rank": selected_rank,
        "cumulative_energy": cumulative_energy.cpu(),
        "meta": meta if meta is not None else {}
    }, save_path)

def load_subspace_cache(save_path):
    data = torch.load(save_path, map_location="cpu")
    return data

def choose_rank_by_energy(singular_values, energy_thresh=0.95, min_rank=1, max_rank=None):
    """
    singular_values: 1D tensor, shape [m]
    energy_thresh: float in (0, 1]
    return:
        k: int
        cumulative_energy: 1D tensor
    """
    s = singular_values.float()
    energy = s ** 2
    total_energy = energy.sum()

    if total_energy <= 0:
        return min_rank, torch.zeros_like(energy)

    cumulative_energy = torch.cumsum(energy, dim=0) / total_energy

    # 找到第一个达到阈值的位置
    k = int(torch.searchsorted(cumulative_energy, torch.tensor(energy_thresh, device=cumulative_energy.device)).item()) + 1

    if max_rank is not None:
        k = min(k, max_rank)

    k = max(k, min_rank)
    return k, cumulative_energy