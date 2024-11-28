from yacs.config import CfgNode as CN
def cfg2dict(cfg_node):
    """
    Recursively converts a yacs CfgNode into a dictionary.
    """
    cfg_dict = dict(cfg_node)
    for key, value in cfg_dict.items():
        if isinstance(value, CN):
            cfg_dict[key] = cfg2dict(value)
    return cfg_dict