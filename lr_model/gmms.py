from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


def _fit_gmm_bic(
    X: np.ndarray,
    k_min: int = 1,
    k_max: int = 16,
    covariance_type: str = "full",
    n_init: int = 3,
    max_iter: int = 500,
    random_state: Optional[int] = 42,
    reg_covar: float = 1e-6,
) -> Tuple[GaussianMixture, Dict[str, Any]]:
    """Fit multiple GMMs and pick the one with lowest BIC."""
    best_model = None
    best_bic = np.inf
    bic_curve = []
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            reg_covar=reg_covar,
            init_params="kmeans",
        ).fit(X)
        bic = gmm.bic(X)
        bic_curve.append((k, bic))
        if bic < best_bic:
            best_bic = bic
            best_model = gmm
    meta = {"bic": best_bic, "bic_curve": bic_curve, "selected_k": best_model.n_components}
    return best_model, meta

# Define a single GMM fitting function
def fit_single_gmm(embeddings: np.ndarray,
                   n_components: int,
                   covariance_type: str = "full",
                   n_init: int = 3,
                   max_iter: int = 500,
                   random_state: Optional[int] = 42,
                   reg_covar: float = 1e-6) -> GaussianMixture:
    """Fit a single Gaussian Mixture Model (GMM) to the provided embeddings."""
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        reg_covar=reg_covar,
        init_params="kmeans",
    ).fit(embeddings)
    return gmm


def fit_likelihood_models(match_embeddings: np.ndarray,
                          nonmatch_embeddings: np.ndarray,
                          args) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Fit Gaussian models for P(z | H1_match) and P(z | H2_nonmatch).

    Supports:
      - args.gmm_model in {'bayesian', 'standard'}
      - For 'standard', if args.n_components_gmm == 'auto', uses BIC to select K in [args.bic_k_min, args.bic_k_max].

    Returns:
      match_model, nonmatch_model, info_meta
        - info_meta: dict with helpful training metadata (selected Ks, active comps, BIC curves, etc.)
    """
    info = {
        "model_type": args.gmm_model,
        "covariance_type": args.covariance_type,
    }

    if args.gmm_model == 'bayesian':
        print("  -> Using BayesianGaussianMixture (Dirichlet process).")
        # Upper bound on components; the model will prune unused ones.
        match_model = BayesianGaussianMixture(
            n_components=args.n_components_gmm,
            covariance_type=args.covariance_type,
            weight_concentration_prior_type="dirichlet_process",
            init_params="kmeans",
            max_iter=getattr(args, "max_iter", 1000),
            random_state=getattr(args, "random_state", 42),
            reg_covar=getattr(args, "reg_covar", 1e-6),
        ).fit(match_embeddings)

        nonmatch_model = BayesianGaussianMixture(
            n_components=args.n_components_gmm,
            covariance_type=args.covariance_type,
            weight_concentration_prior_type="dirichlet_process",
            init_params="kmeans",
            max_iter=getattr(args, "max_iter", 1000),
            random_state=getattr(args, "random_state", 42),
            reg_covar=getattr(args, "reg_covar", 1e-6),
        ).fit(nonmatch_embeddings)

        # Effective (active) components
        match_active = int(np.sum(match_model.weights_ > 1e-3))
        nonmatch_active = int(np.sum(nonmatch_model.weights_ > 1e-3))
        info.update({
            "match_active_components": match_active,
            "nonmatch_active_components": nonmatch_active,
            "upper_bound_components": args.n_components_gmm,
        })
        print(f"     H1 active comps: {match_active} / {args.n_components_gmm}")
        print(f"     H2 active comps: {nonmatch_active} / {args.n_components_gmm}")

    else:
        print("  -> Using standard GaussianMixture.")
        # Allow automatic K via BIC if requested
        auto_k = (isinstance(args.n_components_gmm, str) and args.n_components_gmm.lower() == "auto")

        if auto_k:
            k_min = getattr(args, "bic_k_min", 1)
            k_max = getattr(args, "bic_k_max", 16)
            print(f"     Selecting K by BIC in [{k_min}, {k_max}] for H1 …")
            match_model, meta1 = _fit_gmm_bic(
                match_embeddings,
                k_min=k_min,
                k_max=k_max,
                covariance_type=args.covariance_type,
                n_init=getattr(args, "n_init", 3),
                max_iter=getattr(args, "max_iter", 500),
                random_state=getattr(args, "random_state", 42),
                reg_covar=getattr(args, "reg_covar", 1e-6),
            )
            print(f"       -> H1 selected K = {meta1['selected_k']} (BIC={meta1['bic']:.2f})")

            print(f"     Selecting K by BIC in [{k_min}, {k_max}] for H2 …")
            nonmatch_model, meta2 = _fit_gmm_bic(
                nonmatch_embeddings,
                k_min=k_min,
                k_max=k_max,
                covariance_type=args.covariance_type,
                n_init=getattr(args, "n_init", 3),
                max_iter=getattr(args, "max_iter", 500),
                random_state=getattr(args, "random_state", 42),
                reg_covar=getattr(args, "reg_covar", 1e-6),
            )
            print(f"       -> H2 selected K = {meta2['selected_k']} (BIC={meta2['bic']:.2f})")

            info.update({
                "bic_match_selected_k": meta1["selected_k"],
                "bic_nonmatch_selected_k": meta2["selected_k"],
                "bic_match_curve": meta1["bic_curve"],
                "bic_nonmatch_curve": meta2["bic_curve"],
            })

        else:
            # Fixed K provided
            k = int(args.n_components_gmm)
            match_model = GaussianMixture(
                n_components=k,
                covariance_type=args.covariance_type,
                n_init=getattr(args, "n_init", 3),
                max_iter=getattr(args, "max_iter", 500),
                random_state=getattr(args, "random_state", 42),
                reg_covar=getattr(args, "reg_covar", 1e-6),
                init_params="kmeans",
            ).fit(match_embeddings)

            nonmatch_model = GaussianMixture(
                n_components=k,
                covariance_type=args.covariance_type,
                n_init=getattr(args, "n_init", 3),
                max_iter=getattr(args, "max_iter", 500),
                random_state=getattr(args, "random_state", 42),
                reg_covar=getattr(args, "reg_covar", 1e-6),
                init_params="kmeans",
            ).fit(nonmatch_embeddings)

            info.update({"fixed_k": k})

    return match_model, nonmatch_model, info
