import sys
sys.path.append('ms_istft_vits')

import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

device = 'cpu' #cuda:0
config = f"ms_istft_vits/pretrained_model/kss.json"
model = f"ms_istft_vits/pretrained_model/kss.pth"


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


hps = utils.get_hparams_from_file(config)
SAMPLE_RATE = hps.data.sampling_rate

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint(model, net_g, None)

def generate_audio(text):
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        return audio
