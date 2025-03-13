from funasr.register import tables
from funasr.datasets.llm_datasets.preprocessor import TextPreprocessRemovePunctuation

if __name__ == "__main__":

    tokenizer_conf = {'unk_symbol': '<unk>', 'init_param_path': '/ssd/zhuang/code/LLM/Qwen1.5-7B-Chat/'}
    tokenizer = 'HuggingfaceTokenizer'

    tokenizer_class = tables.tokenizer_classes.get(tokenizer)
    tokenizer = tokenizer_class(**tokenizer_conf)
    # text_processor = TextPreprocessRemovePunctuation()

    target = "n|i↓↑|x|ɑʊ↓↑"
    # target = text_processor(target)
    target = target.lower()
    id = tokenizer.encode(target)
    re_target = tokenizer.decode(id)
    print (id)
    print (re_target)
    print (target)

