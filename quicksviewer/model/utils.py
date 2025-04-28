from transformers import AutoConfig
from typing import Dict
import transformers
from importlib import import_module
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F



def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg





def _get_unpad_data(attention_mask: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if hasattr(_get_unpad_data, "seqlens_in_batch"):
        seqlens_in_batch = _get_unpad_data.seqlens_in_batch
    else:
        seqlens_in_batch = torch.sum(attention_mask, dim=1)

    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def set_seqlens_in_batch(seqlens_in_batch: torch.Tensor) -> None:
    _get_unpad_data.seqlens_in_batch = seqlens_in_batch


def patch(model: nn.Module) -> None:
    if transformers.__version__ < "4.43.0":
        m = import_module(model.__module__)
        if not hasattr(m, "_get_unpad_data"):
            raise ValueError(f"Module {m} does not have function '_get_unpad_data' for packing")
        m._get_unpad_data = _get_unpad_data
    else:
        transformers.modeling_flash_attention_utils._get_unpad_data = _get_unpad_data


