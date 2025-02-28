from transformers import HubertModel, HubertConfig

configuration = HubertConfig.from_pretrained("/ssd/zhuang/Pretraining_model/hubert-large-ls960-ft/")
model = HubertModel(configuration)
configuration = model.config