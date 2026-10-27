from hemahier.models.heads.flat import FlatHead
from hemahier.models.heads.hemahier import HemaHierHead
from hemahier.models.heads.hmil import HMILHead

HEADS = {"flat": FlatHead, "hmil": HMILHead, "hemahier": HemaHierHead}


def build_head(model_cfg: dict, ont, dim_l: int, dim_m: int):
    cfg = dict(model_cfg)
    return HEADS[cfg.pop("name")](ont, dim_l, dim_m, **cfg)
