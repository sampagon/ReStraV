import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import yt_dlp
import tempfile
import os
import dinov2_features as d2

#Load model
class MLP(nn.Module):
    def __init__(self, in_dim=21, h1=64, h2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1)
        )
    def forward(self, x):
        return self.net(x)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = MLP()
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.to(device)
model.eval()

mean = np.load("mean.npy")
std = np.load("std.npy")
best_tau = float(np.load("best_tau.npy"))

#Download video using yt-dlp and return path
def download_video(url):
    try:
        temp_dir = tempfile.mkdtemp()
        outfile = os.path.join(temp_dir, "video.mp4")

        ydl_opts = {
            "outtmpl": outfile,
            "format": "mp4/bestvideo+bestaudio/best",
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        return outfile
    except Exception as e:
        return gr.Error(f"Download failed: {e}")


#Perform classification
def classify(video_file):
    if video_file is None:
        return {"No video": 1.0}

    Z = d2.extract_dinov2_embeddings([video_file], device=device)
    features = d2.features_from_Z(Z)

    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    x = (features - mean.squeeze()) / std.squeeze()
    x = x.astype(np.float32)
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        prob = torch.sigmoid(logits).cpu().numpy().item()

    return {"REAL": prob, "FAKE": 1 - prob}

with gr.Blocks(title="ReStrav Classifier") as demo:

    gr.Markdown("### Upload a video OR paste a URL. If using a URL, the video will be downloaded and displayed below.")

    with gr.Row():
        with gr.Column():
            video_widget = gr.Video(label="Video")
            url_input = gr.Textbox(label="Paste video URL here")

            download_btn = gr.Button("Load Video from URL")
        with gr.Column():
            output = gr.Label(label="Prediction")
            classify_btn = gr.Button("Classify")

    def handle_download(url):
        if not url:
            return gr.Error("No URL provided"), None
        path = download_video(url)
        return path

    download_btn.click(handle_download, inputs=url_input, outputs=video_widget)

    classify_btn.click(classify, inputs=video_widget, outputs=output)

if __name__ == "__main__":
    demo.launch()
