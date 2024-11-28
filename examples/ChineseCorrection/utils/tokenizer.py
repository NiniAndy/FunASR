import re
from abc import ABC, abstractmethod, abstractproperty
from os import PathLike
from typing import Dict, List, Tuple, Union, Any
from typing import Optional

from examples.ChineseCorrection.utils import cfg2dict
from funasr.register import tables

T = Union[str, bytes]

def init_tokenizer(kwargs: dict):
    tokenizer = kwargs.get("tokenizer", None)
    if tokenizer is not None:
        tokenizer_class = tables.tokenizer_classes.get(tokenizer)
        tokenizer = tokenizer_class(**kwargs.get("tokenizer_conf", {}))
        kwargs["token_list"] = (tokenizer.token_list if hasattr(tokenizer, "token_list") else None)
        kwargs["token_list"] = (tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else kwargs["token_list"])
        vocab_size = len(kwargs["token_list"]) if kwargs["token_list"] is not None else -1
        if vocab_size == -1 and hasattr(tokenizer, "get_vocab_size"):
            vocab_size = tokenizer.get_vocab_size()
    else:
        vocab_size = -1
    kwargs["tokenizer"] = tokenizer
    kwargs["tokenizer_conf"]["vocab_size"] = vocab_size

    return kwargs


def init_pny_tokenizer(kwargs: dict):
    pny_tokenizer = kwargs.get("pny_tokenizer", None)
    if pny_tokenizer is not None:
        pny_tokenizer_class = tables.tokenizer_classes.get(pny_tokenizer)
        pny_tokenizer = pny_tokenizer_class(**kwargs.get("pny_tokenizer_conf", {}))
        kwargs["pny_token_list"] = (pny_tokenizer.token_list if hasattr(pny_tokenizer, "token_list") else None)
        kwargs["pny_token_list"] = (pny_tokenizer.get_vocab() if hasattr(pny_tokenizer, "get_vocab") else kwargs["pny_token_list"])
        pny_vocab_size = len(kwargs["pny_token_list"]) if kwargs["pny_token_list"] is not None else -1
        if pny_vocab_size == -1 and hasattr(pny_tokenizer, "get_vocab_size"):
            pny_vocab_size = pny_tokenizer.get_vocab_size()
    else:
        pny_vocab_size = -1
    kwargs["pny_tokenizer"] = pny_tokenizer
    kwargs["pny_tokenizer_conf"]["vocab_size"] = pny_vocab_size
    return kwargs

