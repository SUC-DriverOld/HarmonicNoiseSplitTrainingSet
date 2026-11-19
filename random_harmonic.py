import numpy as np
from numba import njit
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.interpolate import PchipInterpolator
import librosa
import soundfile as sf
from tools import sample_y_coords_with_weighted_intervals


def upsample(data, hopsize):
    """Interpolate frame controls without negative amplitude/F0 overshoot."""
    data = np.asarray(data)
    if data.ndim == 2:
        return np.column_stack([upsample(row, hopsize) for row in data])
    if data.ndim != 1 or data.size == 0:
        raise ValueError("data must be a non-empty 1D or 2D array")
    if len(data) == 1:
        return np.full(hopsize, data[0], dtype=float)
    frame_times = np.arange(len(data)) * hopsize
    sample_times = np.minimum(np.arange(len(data) * hopsize), frame_times[-1])
    return PchipInterpolator(frame_times, data)(sample_times)


@njit(cache=True)
def _sum_partials(phase, f0, coefficients, last_frame, offsets, hop, sr):
    """Evaluate cubic envelopes and sum audible partials without a samples x orders array."""
    sin_offset, cos_offset = np.sin(offsets), np.cos(offsets)
    out = np.empty(len(phase))
    n_intervals = coefficients.shape[1]
    nyquist, fade_width = sr * 0.5, sr * 0.02
    for frame in range((len(out) + hop - 1) // hop):
        start = frame * hop
        for i in range(start, min(start + hop, len(out))):
            t = (i - start) / hop
            s1, c1 = np.sin(phase[i]), np.cos(phase[i])
            s, c = s1, c1
            total = 0.0
            for k in range(min(len(offsets), int(nyquist / f0[i]))):
                if frame < n_intervals:
                    a = (
                        (coefficients[0, frame, k] * t + coefficients[1, frame, k]) * t
                        + coefficients[2, frame, k]
                    ) * t + coefficients[3, frame, k]
                else:
                    a = last_frame[k]
                taper = min(1.0, (nyquist - (k + 1) * f0[i]) / fade_width)
                total += (s * cos_offset[k] + c * sin_offset[k]) * (a * taper)
                # Rotate by one fundamental phase to obtain sin/cos of the next order.
                s, c = s * c1 + c * s1, c * c1 - s * s1
            out[i] = total
    return out


class GeneratorHarmonic:
    def __init__(
        self,
        sr=44100,
        hopsize=512,
        random_seed=42,
        # f0参数
        sin_amp_max=2,
        sin_amp_min=0,
        sin_period_max=0.5,
        sin_period_min=0.15,
        gaussian_smooth_sigma=2.0,
        # f0步长参数
        step_hop=4096,
        min_seconds=0.5,
        max_seconds=1.85,
        max_gap=6.0,
        min_note="G2",
        max_note="C6",
        # 掩码参数
        mask_min0=0.05,
        mask_max0=0.5,
        mask_min1=0.4,
        mask_max1=6,
    ):
        if not isinstance(sr, (int, np.integer)) or sr < 8000:
            raise ValueError("sr must be an integer >= 8000")
        if not isinstance(hopsize, (int, np.integer)) or hopsize < 1:
            raise ValueError("hopsize must be a positive integer")
        if step_hop < 1 or min_seconds <= 0 or max_seconds < min_seconds:
            raise ValueError("invalid step duration")
        if sin_period_min <= 0 or sin_period_max < sin_period_min:
            raise ValueError("invalid vibrato period")
        self.sr = sr
        self.hopsize = hopsize

        # F0 参数
        self.sin_amp_max = sin_amp_max
        self.sin_amp_min = sin_amp_min
        self.sin_period_max = sin_period_max
        self.sin_period_min = sin_period_min
        self.gaussian_smooth_sigma = gaussian_smooth_sigma

        # 步长参数
        self.step_hop = step_hop
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.max_gap = max_gap
        self.min_note = min_note
        self.max_note = max_note

        # 掩码参数
        self.mask_min0 = mask_min0
        self.mask_max0 = mask_max0
        self.mask_min1 = mask_min1
        self.mask_max1 = mask_max1

        # 计算步长参数
        self.step_min_length = max(1, int(self.min_seconds * self.sr / self.step_hop))
        self.step_max_length = max(
            self.step_min_length, int(self.max_seconds * self.sr / self.step_hop)
        )

        self.step_min_midi = librosa.note_to_midi(self.min_note)
        self.step_max_midi = librosa.note_to_midi(self.max_note)

        # 计算掩码参数
        self.min0 = max(1, int(round(self.mask_min0 * self.sr / self.hopsize)))
        self.max0 = max(self.min0, int(round(self.mask_max0 * self.sr / self.hopsize)))
        self.min1 = max(1, int(round(self.mask_min1 * self.sr / self.hopsize)))
        self.max1 = max(self.min1, int(round(self.mask_max1 * self.sr / self.hopsize)))

        # 初始化随机数生成器
        self.rng = np.random.default_rng(random_seed)

        f0_sin_max_f = self.sr / self.hopsize / 2

        if 1 / sin_period_min > f0_sin_max_f:
            raise ValueError(
                f"f0抖动频率太高，大于采样率（sr/hop/2）的一半，当前值：{f0_sin_max_f}"
            )

    def generate_harmonic(self, n_samples, apply_mask=True):
        """Return exactly n_samples; paired synthesis supplies its own voicing mask."""
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        f0 = self.generate_f0(n_samples)  # shape (spec_len,)

        spec_len = f0.shape[0]
        ampl_spec, x_coords, y_coords = self.generate_amplitudes(
            spec_len
        )  # shape (spec_len, sr//2)

        mask = self.generate_mask(spec_len) if apply_mask else np.ones(spec_len)
        mask = mask[:, np.newaxis]

        # 计算振幅, 从dB到linear
        ampl_spec = 10 ** (ampl_spec / 20)  # shape (spec_len, sr//2)

        ampl_spec *= mask  # shape (spec_len, sr//2)

        max_nhar = max(1, int((self.sr / 2) / f0.min()))
        freqs = f0[:, None] * np.arange(1, max_nhar + 1)
        # Query times are exactly on frames, so only frequency bins need interpolation.
        last_bin = ampl_spec.shape[1] - 1
        bins = np.minimum(freqs / (self.sr / 2), 1.0) * last_bin
        lo = np.minimum(bins.astype(np.intp), last_bin - 1)
        rows = np.arange(spec_len)[:, None]
        left = ampl_spec[rows, lo]
        frames = left + (bins - lo) * (ampl_spec[rows, lo + 1] - left)
        frames[freqs > self.sr / 2] = 0.0
        coefficients = PchipInterpolator(np.arange(spec_len), frames, axis=0).c
        f0_upsampled = upsample(f0, self.hopsize)[:n_samples]
        x = np.cumsum(f0_upsampled / self.sr)
        x = x - np.round(x)
        offsets = self.rng.uniform(0, 1, size=max_nhar) + self.rng.uniform(0, 2 * np.pi)
        return _sum_partials(
            2 * np.pi * x,
            f0_upsampled,
            coefficients,
            frames[-1],
            offsets,
            self.hopsize,
            self.sr,
        )

    def generate_f0(self, n_samples):
        # 生成F0阶梯
        num_steps = n_samples // self.step_hop + 1
        f0_step = self.generate_step(num_steps)

        # 上采样到hopsize级
        out_len = max(2, n_samples // self.hopsize + 1)
        f0_step = np.interp(
            np.arange(out_len) * self.hopsize,
            np.arange(num_steps) * self.step_hop,
            f0_step,
        )

        # 随机生成调幅
        sin_period = self.rng.uniform(self.sin_period_min, self.sin_period_max)
        omega = 2 * np.pi / sin_period
        thop = self.hopsize / self.sr
        x = np.arange(out_len) * thop

        sin_amp_range = (self.sin_amp_max - self.sin_amp_min) / 2
        sin_amp_center = (self.sin_amp_max + self.sin_amp_min) / 2

        sin_amp = np.sin(2 * np.pi / 2 * x) * sin_amp_range + sin_amp_center
        sin_wave = sin_amp * np.sin(omega * x)
        f0 = f0_step + sin_wave

        f0 = ndimage.gaussian_filter1d(f0, sigma=self.gaussian_smooth_sigma)

        return librosa.midi_to_hz(f0)

    def generate_step(self, total_length):
        f0_step = np.empty(total_length)
        current_position = 0

        # 初始midi
        current_value = self.rng.uniform(self.step_min_midi, self.step_max_midi)

        while current_position < total_length:
            step_length = self.rng.integers(
                self.step_min_length, self.step_max_length + 1
            )
            step_length = min(step_length, total_length - current_position)

            # 填充当前midi值
            f0_step[current_position : current_position + step_length] = current_value
            current_position += step_length

            if current_position >= total_length:
                break

            # 生成midi新值，确保与当前值的差距不超过max_gap
            min_next = max(self.step_min_midi, current_value - self.max_gap)
            max_next = min(self.step_max_midi, current_value + self.max_gap)

            current_value = self.rng.uniform(min_next, max_next)

        return f0_step

    def generate_amplitudes(self, spec_len):
        spec_len = max(2, spec_len)
        max_freq = 2048
        ampl_spec = np.zeros((spec_len, max_freq))

        ampl_contro_points_density = 16  # 控制点密度,一秒钟有多少个控制点
        seconds = spec_len * self.hopsize / self.sr
        num_random_points = int(
            ampl_contro_points_density * seconds
        )  # 注意修正mun_control_points为num_control_points

        x_min, x_max = 0, spec_len
        y_min, y_max = 0, max_freq

        candidates = np.arange(1, spec_len - 1)
        num_random_points = min(num_random_points, len(candidates))
        x_coords_rand = self.rng.choice(
            candidates, size=num_random_points, replace=False
        )
        y_coords_rand = sample_y_coords_with_weighted_intervals(
            self.rng, y_min, y_max, num_random_points
        )

        fixed_points = np.array(
            [
                [x_min, y_min],  # 左下
                [x_max - 1, y_min],  # 右下
                [x_min, y_max - 1],  # 左上
                [x_max - 1, y_max - 1],  # 右上
                [x_max // 2, y_max - 1],  # 上边中点
                [x_max // 2, 0],  # 下边中点
                [x_min, y_max // 2],  # 左边中点
                [x_max - 1, y_max // 2],  # 右边中点
            ]
        )
        x_coords = np.concatenate([x_coords_rand, fixed_points[:, 0]])
        y_coords = np.concatenate([y_coords_rand, fixed_points[:, 1]])
        points = np.column_stack((x_coords, y_coords / 10))

        num_control_points = num_random_points + 8
        ampl_values = self.rng.uniform(-50, -10, size=num_control_points)

        grid_x, grid_y = np.mgrid[0:spec_len, 0:max_freq]

        # 使用三次插值在网格上计算振幅值
        ampl_spec = griddata(
            points, ampl_values, (grid_x, grid_y / 10), method="linear"
        )

        ampl_spec = np.nan_to_num(ampl_spec)

        low_pass = np.linspace(0, self.rng.uniform(-50, -25), max_freq)
        ampl_spec = ampl_spec + low_pass[None, :]

        return ampl_spec, x_coords, y_coords

    def generate_mask(self, mask_len):
        mask = np.zeros(mask_len, dtype=float)
        current_index = 0
        current_state = self.rng.integers(0, 2)  # 随机选择起始状态（0或1）

        while current_index < mask_len:
            # 确定当前状态需要的片段长度
            if current_state == 0:
                seg_length = self.rng.integers(self.min0, self.max0 + 1)
            else:
                seg_length = self.rng.integers(self.min1, self.max1 + 1)

            # 确保片段不超过总长度
            seg_end = min(current_index + seg_length, mask_len)

            mask[current_index:seg_end] = current_state

            current_index = seg_end
            current_state = 1 - current_state

        mask = ndimage.gaussian_filter1d(mask, sigma=self.rng.uniform(0.5, 3))
        if np.all(mask == 0):
            mask[:] = 1

        return mask


def plot_multiple_f0_curves(
    generator: GeneratorHarmonic, duration: float = 5.0, num_curves: int = 3
):
    """
    生成并绘制多个 F0 曲线。

    参数:
        generator (GeneratorHarmonic): 已实例化的 GeneratorHarmonic 对象。
        duration (float): 要绘制的时长（秒）。默认为 5 秒。
        num_curves (int): 要生成和绘制的 F0 曲线数量。默认为 3。
        sr (int): 采样率。默认为 44100 Hz。
    """
    from matplotlib import pyplot as plt

    plt.figure(figsize=(12, 6))

    # 生成时间轴 (秒)
    # 注意: generate_f0 返回的是 hopsize 级别的序列
    hopsize = generator.hopsize
    sr = generator.sr
    time_axis = np.arange(int(duration * sr) // hopsize + 1) * (hopsize / sr)
    # 截断到指定时长
    time_axis = time_axis[time_axis <= duration]

    # 定义颜色
    colors = plt.cm.Set1(np.linspace(0, 1, num_curves))  # 使用 Set1 色彩映射

    for i in range(num_curves):
        # 生成 F0 曲线
        f0_curve = generator.generate_f0(int(duration * sr))
        # 截断到指定时长
        f0_curve = f0_curve[: len(time_axis)]

        # 绘制曲线
        plt.plot(
            time_axis, f0_curve, color=colors[i], linewidth=2, label=f"F0 曲线 {i + 1}"
        )

    plt.xlabel("时间 (秒)")
    plt.ylabel("F0 (Hz)")
    plt.title(f"{num_curves} 个生成的 F0 曲线 (叠加)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, duration)
    # 可以根据需要设置 y 轴范围，例如 plt.ylim(0, 500)
    plt.tight_layout()
    plt.show()


def plot_amplitude_spectrogram(generator: GeneratorHarmonic, duration: float = 5.0):
    from matplotlib import pyplot as plt

    hopsize = generator.hopsize
    sr = generator.sr
    time_axis = np.arange(int(duration * sr) // hopsize + 1) * (hopsize / sr)
    # 截断到指定时长
    time_axis = time_axis[time_axis <= duration]

    # 计算振幅谱
    spec_len = len(time_axis)
    max_freq = 2048
    ampl_spec, x_coords, y_coords = generator.generate_amplitudes(spec_len)

    # 绘制振幅谱
    plt.figure(figsize=(10, 6))
    # 使用 imshow 显示频谱，origin='lower' 保证频率从下往上增加
    cax = plt.imshow(
        ampl_spec.T,
        aspect="auto",
        origin="lower",
        extent=[0, spec_len, 0, max_freq],
        cmap="viridis",
    )

    plt.scatter(
        x_coords,
        y_coords,
        c="red",
        s=10,
        alpha=0.7,
        edgecolors="white",
        linewidth=0.5,
        label="Control Points",
    )
    plt.colorbar(cax, label="Amplitude")
    plt.title("Interpolated Amplitude Spectrogram")
    plt.xlabel("Time Frame")
    plt.ylabel("Frequency Bin")
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 创建 X, Y 网格（用于3D绘图）
    X = np.arange(0, spec_len)
    Y = np.arange(0, max_freq)
    X, Y = np.meshgrid(X, Y)
    Z = ampl_spec.T  # 注意转置，匹配 (freq, time)

    # 绘制3D曲面图
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.6, label="Amplitude")

    # 设置标签
    ax.set_title("3D Amplitude Spectrogram", fontsize=16)
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Frequency Bin")
    ax.set_zlabel("Amplitude")

    # 可选：调整视角
    ax.view_init(elev=25, azim=-45)  # elev: 仰角, azim: 方位角

    plt.tight_layout()
    plt.show()


# --- 使用示例 ---
if __name__ == "__main__":
    # 实例化 GeneratorHarmonic 对象
    gen = GeneratorHarmonic(
        sr=44100,
        hopsize=512,
        random_seed=42,
        # F0 参数
        sin_amp_max=1.0,
        sin_amp_min=0,
        sin_period_max=0.3,
        sin_period_min=0.15,
        gaussian_smooth_sigma=2,
        # 步长参数
        step_hop=4096,
        min_seconds=0.5,
        max_seconds=1.0,
        max_gap=12.0,
        min_note="G2",
        max_note="C6",
        # 掩码参数
        mask_min0=0.05,
        mask_max0=0.5,
        mask_min1=0.4,
        mask_max1=6,
    )

    # 调用绘图函数
    # plot_amplitude_spectrogram(gen, duration=5)

    # 调用生成函数10次
    for i in range(10):
        waveform = gen.generate_harmonic(n_samples=44100 * 5)
        sf.write(f"output{i}.wav", waveform, 44100, subtype="FLOAT")
