from typing import Dict, List, Tuple
from funasr.tokenizer.g2p_tokenizer_utils import PhonemeBpeTokenizer
from funasr.register import tables

@tables.register("tokenizer_classes", "G2PTokenizer")
class G2PTokenizer(object):

    def __init__(self, symbol_table_path, language: str = 'zh', *args, **kwargs) -> None:
        # NOTE(Mddct): don't build here, pickle issues
        self.language = language
        self.tokenizer = PhonemeBpeTokenizer(symbol_table_path)
        self._vocab = self.tokenizer.vocab
        self.token2id = self._vocab
        self.id2token ={v: k for k, v in self.token2id.items()}
        self.token_list = list(self._vocab.keys())
        self._vocab_size = len(self._vocab)

    def tokenize(self, line: str) -> Tuple[List[str], List[int]]:
        text, ids = self.tokenizer.tokenize(line, self.language)
        return text, ids

    def detokenize(self, ids: List[int]) -> Tuple[str, List[str]]:
        raise ValueError("PhonemeBpeTokenizer does not support detokenize")


    def text2tokens(self, line: str) -> List[str]:
        return self.tokenize(line)[0]

    def tokens2text(self, tokens: List[str]) -> str:
        raise ValueError("PhonemeBpeTokenizer does not support detokenize")

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        ids = [self._vocab[t] for t in tokens]
        return ids

    def ids2tokens(self, ids: List[int]) -> List[str]:
        raise ValueError("PhonemeBpeTokenizer does not support detokenize")

    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def symbol_table(self) -> Dict[str, int]:
        return self._vocab


if __name__ == '__main__':
    # text = "你好"
    # language = 'zh'
    text = "hello"
    language = 'en'
    tokenizer = G2PTokenizer("./g2p_tokenizer_utils/vocab.json", language)
    text, ids = tokenizer.tokenize(text)
    vocab_size = tokenizer.vocab_size()
    print(text, ids)