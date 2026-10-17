import torch


def is_torchdynamo_exporting() -> bool:
    """
    Implementation from
    https://github.com/huggingface/transformers/blob/09281b8d56b08199ce9f397c694f38853cc45278/src/transformers/utils/import_utils.py#L1768
    """
    try:
        import torch

        return torch.compiler.is_exporting()
    except Exception:
        return False


def count_parameters(model: torch.nn.Module) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
