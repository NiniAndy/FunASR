from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union
from typing import Optional
from typing import Dict
import warnings
import re
from os import PathLike

from funasr.tokenizer.abs_tokenizer import BaseTokenizer
from funasr.register import tables


@tables.register("tokenizer_classes", "BPETokenizer")
class BPETokenizer(BaseTokenizer):
    def __init__(
        self,
        bpe_model: Union[PathLike, str],
        non_lang_syms: Optional[Union[str, PathLike, List]] = None,
        split_with_space: bool = False,
        connect_symbol: str = '',
        unk_symbol='<unk>',
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.non_lang_syms_pattern = None
        if non_lang_syms is not None:
            self.non_lang_syms_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")

        if not isinstance(non_lang_syms, List):
            self.non_lang_syms = read_non_lang_symbols(non_lang_syms)
        else:
            # non_lang_syms=["{NOISE}"]
            self.non_lang_syms = non_lang_syms

        self.split_with_space = split_with_space
        self.connect_symbol = connect_symbol
        self.unk = unk_symbol


        self._model = bpe_model
        self.bpe_model = None

    def _build_sp(self):
        if self.bpe_model is None:
            import sentencepiece as spm
            self.bpe_model = spm.SentencePieceProcessor()
            self.bpe_model.load(self._model)

    def text2tokens(self, line: Union[str, list]) -> List[str]:
        self._build_sp()
        line = line.strip()
        if self.non_lang_syms_pattern is not None:
            parts = self.non_lang_syms_pattern.split(line.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [line]

        tokens = []
        for part in parts:
            if part in self.non_lang_syms:
                tokens.append(part)
            else:
                tokens.extend(tokenize_by_bpe_model(self.bpe_model, part))
        return tokens


    def tokens2text(self, tokens: Iterable[str]) -> str:
        self._build_sp()
        text = self.connect_symbol.join(tokens)
        return text.replace("▁", ' ').strip()



def tokenize_by_bpe_model(sp, txt, upper=True, seg_dict=None,):
    if sp is None:
        assert seg_dict is not None
    if seg_dict is None:
        assert sp is not None
    tokens = []
    # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    pattern = re.compile(r'([\u4e00-\u9fff])')
    # Example:
    #   txt   = "你好 ITS'S OKAY 的"
    #   chars = ["你", "好", " ITS'S OKAY ", "的"]
    chars = pattern.split(txt.upper() if upper else txt)
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for ch_or_w in mix_chars:
        # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
        if pattern.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
        # encode ch_or_w using bpe_model.
        else:
            if sp is not None:
                for p in sp.encode_as_pieces(ch_or_w):
                    tokens.append(p)
            else:
                for en_token in ch_or_w.split():
                    en_token = en_token.strip()
                    if en_token in seg_dict:
                        tokens.extend(seg_dict[en_token].split(' '))
                    else:
                        tokens.append(en_token)
    return tokens



def read_non_lang_symbols(non_lang_sym_path):
    """read non-linguistic symbol from file.

    The file format is like below:

    {NOISE}\n
    {BRK}\n
    ...

    Args:
        non_lang_sym_path: non-linguistic symbol file path, None means no any syms.

    """
    if non_lang_sym_path is None:
        return []
    else:
        syms = read_lists(non_lang_sym_path)
        non_lang_syms_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
        for sym in syms:
            if non_lang_syms_pattern.fullmatch(sym) is None:

                class BadSymbolFormat(Exception):
                    pass

                raise BadSymbolFormat(
                    "Non-linguistic symbols should be "
                    "formatted in {xxx}/<xxx>/[xxx], consider"
                    " modify '%s' to meet the requirment. "
                    "More details can be found in discussions here : "
                    "https://github.com/wenet-e2e/wenet/pull/819" % (sym))
        return syms


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


'''
tokenizer_conf = {
    "token_list": '/ssd/zhuang/code/FunASR/examples/librispeech/DATA/data/lang_char/train_960/unigram5000.txt',
    "bpe_model": '/ssd/zhuang/code/FunASR/examples/librispeech/DATA/data/lang_char/train_960/unigram5000.model',
    "non_lang_syms": None,
    "split_with_space": False,
}

tokenizer = BPETokenizer(**tokenizer_conf)
token = "IF HE'D RUN OUT OF TURNIP SEED HE WOULDN'T DRESS UP AND TAKE THE BUGGY TO GO FOR MORE"
id = tokenizer.encode(token)
restore = tokenizer.decode(id)
'''