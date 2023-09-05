import os
import glob
from pathlib import Path
import subprocess
import cog
from wavebeat.dstcn import dsTCNModel


class Predictor(cog.Predictor):
    def setup(self):
        # find the checkpoint path
        ckpts = glob.glob(os.path.join("checkpoints", "*.ckpt"))
        if len(ckpts) < 1:
            raise RuntimeError(f"No checkpoints found in checkpoints.")
        else:
            self.ckpt_path = ckpts[-1]
        # construct the model, and load weights from checkpoint
        self.model = dsTCNModel.load_from_checkpoint(self.ckpt_path)
        # set model to eval mode
        self.model.eval()

    @cog.input("input", type=Path, help="Path to the input music. Supports wav format.")
    def predict(self, audio_path):
        output_path = "/tmp/converted_audio.wav"

        # Use ffmpeg to convert the audio to .wav format
        command = ["ffmpeg", "-i", audio_path, output_path]
        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            return f"Error processing audio: {e.output.decode('utf-8')}"

        # Predict beats and downbeats
        beats, downbeats = self.model.predict_beats(str(audio_path))

        # Create results as string
        result = "Beats:\n"
        for beat in beats:
            result += f"{beat:0.2f}\n"
        result += "\nDownbeats:\n"
        for downbeat in downbeats:
            result += f"{downbeat:0.2f}\n"

        # Save to file and return
        out_path = audio_path.parent / "result.txt"
        with open(out_path, "w") as f:
            f.write(result)
        return out_path
