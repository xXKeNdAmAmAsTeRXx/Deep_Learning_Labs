import torch
import torchaudio.transforms as T
import torchaudio.functional as F

class AudioPitchShift:
    def __init__(
        self,
        n_steps: float = 8.0,
        sr: int = 8000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400
    ):
        self.n_steps = n_steps
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.rate = 2 ** (-n_steps / 12)

        self.spec_fn = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=None,
        )

        self.stretch_fn = T.TimeStretch(
            hop_length=hop_length,
            n_freq=n_fft // 2 + 1,
            fixed_rate=self.rate,
        )

        self.inv_spec_fn = T.InverseSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        rate = 2 ** (-self.n_steps / 12)

        spec_fn = T.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=None,
        )
        spec = spec_fn(waveform)

        stretch_fn = T.TimeStretch(
            hop_length=self.hop_length,
            n_freq=self.n_fft // 2 + 1,
            fixed_rate=rate,
        )
        stretched_spec = stretch_fn(spec)

        inv_spec_fn = T.InverseSpectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )
        stretched_wav = inv_spec_fn(stretched_spec)

        orig_len = waveform.shape[-1]
        stretched_len = stretched_wav.shape[-1]
        resampled = F.resample(stretched_wav, orig_freq=stretched_len, new_freq=orig_len)

        if resampled.shape[-1] > orig_len:
            resampled = resampled[..., :orig_len]
        elif resampled.shape[-1] < orig_len:
            pad = orig_len - resampled.shape[-1]
            resampled = torch.nn.functional.pad(resampled, (0, pad))

        return resampled

class AudioGaussianNoise:
    def __init__(self, snr_db: torch.Tensor = torch.tensor([20])):
        self.snr_db = snr_db

    def __call__(self, wave: torch.Tensor, snr_db: torch.Tensor = torch.tensor([20, 10, 3])) -> torch.Tensor:
        noise = torch.randn_like(wave)
        return F.add_noise(wave, noise, snr_db)

class AudioReverb:
    def __init__(self, sr: int = 8000, room_scale: float = 0.4):
        self.sr = sr
        self.room_scale = room_scale

    def __call__(self, waveform: torch.Tensor, sr: int = 8000, room_scale: float = 0.4) -> torch.Tensor:
        ir_len = int(sr * room_scale * 0.5)
        t_ir   = torch.linspace(0, 1, ir_len)
        decay  = torch.exp(-6 * t_ir)
        ir     = torch.randn(ir_len) * decay
        ir     = ir / ir.abs().max().clamp(min=1e-8)

        if waveform.ndim == 2:
            wav_3d = waveform.unsqueeze(0)
        else:
            wav_3d = waveform

        num_channels = wav_3d.shape[1]

        ir_3d  = ir.flip(0).unsqueeze(0).unsqueeze(0)
        ir_3d  = ir_3d.repeat(num_channels, 1, 1)
        padding = ir_len - 1

        reverbed = torch.nn.functional.conv1d(
            wav_3d, ir_3d, padding=padding, groups=num_channels
        )[..., :waveform.shape[-1]]

        if waveform.ndim == 2:
            reverbed = reverbed.squeeze(0)

        scale = waveform.abs().max() / reverbed.abs().max().clamp(min=1e-8)
        return reverbed * scale

class AudioBassBoost:
    def __init__(self, sr: int = 8000, gain_db: float = 12.0):
        self.sr = sr
        self.gain_db = gain_db

    def __call__(self, waveform: torch.Tensor, sr: int = 8000, gain_db: float = 12.0) -> torch.Tensor:
        filtered = F.lowpass_biquad(waveform, sample_rate=sr, cutoff_freq=300.0)
        gain_lin  = 10 ** (gain_db / 20)
        bass_only = filtered * gain_lin

        return waveform + (bass_only - waveform) * 0.6