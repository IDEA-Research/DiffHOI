from models.DiffHOI_S.diffhoi_s import build_diffhoi_s
from models.DiffHOI_L.diffhoi_l import build_diffhoi_l


def build_model(args):
    if args.model_name=="diffhoi_s":
        return build_diffhoi_s(args)
    else:
        return build_diffhoi_l(args)
