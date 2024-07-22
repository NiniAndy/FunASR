from dataclasses import fields

def load_cfg_from_dict(cfg, cfg_dict):
    cls = type(cfg)
    for f in fields(cls):
        try:
            value = cfg_dict[f.name]
            setattr(cfg, f.name, value)
        except KeyError:
            pass
    return cfg