from model.gsdecoder.gs_decoder_architecture import GSDecoder


def create_gsdecoder(cfg):
    decoder = GSDecoder(cfg=cfg)
    return decoder
