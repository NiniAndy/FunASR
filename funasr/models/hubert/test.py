from transformers import HubertModel, HubertConfig

if __name__ == "__main__":
    
    configuration = HubertConfig.from_pretrained("/ssd/zhuang/Pretraining_model/hubert-large-ls960-ft/")
    model = HubertModel(configuration)
    configuration = model.config