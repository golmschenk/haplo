import torch

from haplo.losses import norm_based_gradient_clip


def test_norm_based_gradient_clipping():
    gradient_tensor = torch.tensor([3, 4], dtype=torch.float32)
    # Simple 3**2 + 4**2 = 5**2 should result in the values divided by 5.
    expected_normalized_gradient_tensor = torch.tensor([3/5, 4/5], dtype=torch.float32)
    normalized_gradient_tensor = norm_based_gradient_clip(gradient_tensor)
    assert torch.allclose(normalized_gradient_tensor, expected_normalized_gradient_tensor)


def test_norm_based_gradient_clipping_when_norm_limit_is_not_reached():
    gradient_tensor = torch.tensor([3/6, 4/6], dtype=torch.float32)
    # Since the norm of 3 and 4 is 5, dividing by 6 puts the norm below 1.
    expected_normalized_gradient_tensor = torch.tensor([3/6, 4/6], dtype=torch.float32)
    normalized_gradient_tensor = norm_based_gradient_clip(gradient_tensor)
    assert torch.allclose(normalized_gradient_tensor, expected_normalized_gradient_tensor)
