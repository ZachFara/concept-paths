import numpy as np
import torch

from src.capture import capture_activations, load_model_bundle


def test_capture_contract_distilgpt2() -> None:
    prompts = [
        "Overall, the day was great.",
        "The meal was okay.",
        "I found the movie dull.",
        "The service felt excellent.",
        "The app is bad.",
    ]
    bundle = load_model_bundle("distilgpt2", device=torch.device("cpu"))
    acts = capture_activations(
        bundle,
        prompts,
        batch_size=len(prompts),
        capture_sites=("residual", "mlp"),
    )
    residual = acts["residual"][:3]
    mlp = acts["mlp"][:3]

    assert residual.shape[0] == 3
    assert residual.shape[1] == 5
    assert mlp.shape[0] == 3
    assert mlp.shape[1] == 5
    assert np.isfinite(residual).all()
    assert np.isfinite(mlp).all()
