import torch
import torchvision.transforms.functional as T
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_regular_timestamps
import math
import warnings

warnings.filterwarnings(
    "ignore",
    message="xFormers is available.*",
    category=UserWarning,
    module=r"dinov2\.layers\..*",
)

dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()

def preprocess(frames):
    frames = frames.float() / 255.0
    return torch.stack([T.resize(x, [224, 224], InterpolationMode.BICUBIC, antialias=True) for x in frames])

def decode_clip(video_path, T=24, window_sec=2.0, center_time=None):
    dec = VideoDecoder(video_path)
    dur = float(getattr(dec.metadata, "duration_seconds", 0.0))
    if dur <= 0:
        raise ValueError(f"Could not read video duration: {video_path}")
    if center_time is None:
        center_time = dur / 2.0

    half = window_sec / 2
    start, end = max(0.0, center_time - half), min(dur, center_time + half)
    delta = window_sec / max(T - 1, 1)

    clip = clips_at_regular_timestamps(
        dec,
        num_frames_per_clip=T,
        seconds_between_frames=delta,
        seconds_between_clip_starts=1e9,
        sampling_range_start=start,
        sampling_range_end=end,
        policy="repeat_last",
    ).data[0]

    if clip.shape[1] not in (1, 3):
        clip = clip.permute(0, 3, 1, 2)
    return clip

def extract_pixel_embeddings(video_paths, T=24, window_sec=2.0):
    outs = []
    for i, p in enumerate(video_paths):
        try:
            clip = decode_clip(p, T=T, window_sec=window_sec)
            preprocess_clip = preprocess(clip)
            outs.append(preprocess_clip.flatten(1))
        except Exception as e:
            print(f"Error decoding clip: {e}")
    return torch.stack(outs, dim=0)

def extract_dinov2_embeddings(video_paths, device=None, T=24, window_sec=2.0):
    device = device or torch.device("cpu")
    model = dinov2.to(device)
    
    outs = []
    for i, p in enumerate(video_paths):
        try:
            clip = decode_clip(p, T=T, window_sec=window_sec)
            preprocess_clip = preprocess(clip)
            outs.append(preprocess_clip)
        except Exception as e:
            print(f"Error decoding clip: {e}")

    batch = torch.cat(outs, dim=0).to(device)
    with torch.no_grad():
        feats = model.forward_features(batch)
        cls = feats["x_norm_clstoken"].unsqueeze(1)
        patches = feats["x_norm_patchtokens"]
        tokens = torch.cat([cls, patches], dim=1)
        Z_flat = tokens.flatten(1)
        Z = Z_flat.view(len(outs), outs[0].shape[0], -1)
        return Z

def compute_temporal_geometry(Z):
    delta = Z[:, 1:, :] - Z[:, :-1, :]                                 
    d = delta.norm(dim=-1)                                    
    cos = F.cosine_similarity(delta[:, :-1, :], delta[:, 1:,  :], dim=-1)  
    theta = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))      
    return d, theta

def moment4(x):
    mu  = x.mean(dim=-1)                   
    mn  = x.amin(dim=-1)                   
    mx  = x.amax(dim=-1)                   
    var = x.var(dim=-1, unbiased=False)    
    return mu, mn, mx, var

def features_from_Z(Z):
    d, t = compute_temporal_geometry(Z)           
    d7 = d[:, :7]                             
    t6 = t[:, :6]                            
    mu_d, mn_d, mx_d, var_d = moment4(d)
    mu_t, mn_t, mx_t, var_t = moment4(t)
    stats = torch.stack([mu_d, mn_d, mx_d, var_d, mu_t, mn_t, mx_t, var_t], dim=1) 
    return torch.cat([d7, t6, stats], dim=1)