# Harmonic / Noise 合成训练数据

无需准备音频即可生成谐波与噪声的配对训练数据。保留原有 Dataset 类名、主要位置参数以及 `(X, y)`、`(1, n_samples)` 返回格式；另提供不导入、不依赖 torch 的 WAV 导出脚本。

## 安装和直接导出

```powershell
python -m pip install -r requirements.txt
python generate_wavs.py --output-dir pairs --count 100 --duration 5 --sr 44100 --target-lufs -16.5 --seed 42
```

已有项目虚拟环境时，可以直接运行：

```powershell
.\.venv\Scripts\python.exe generate_wavs.py --output-dir pairs --count 100 --duration 5 --target-lufs -18
```

导出使用 NumPy、SciPy、librosa、SoundFile、pyloudnorm，不需要 torch、GPU、Matplotlib 或 FFmpeg。FFmpeg 只用于额外的独立核验。

每组保存到独立目录，三路均为相同采样率、相同样本数的单声道 32-bit FLOAT WAV：

```text
pairs/
  000000/
    harmonic.wav
    noise.wav
    mixure.wav
    metadata.json
  000001/
    harmonic.wav
    noise.wav
    mixure.wav
    metadata.json
```

默认按需求使用文件名 `mixure.wav`。若希望使用常见拼写，可传入 `--mixture-name mixture.wav`。

### 响度和峰值

`--target-lufs`（简写 `--lufs`）控制**混合音频的综合响度**，默认 `-16.5 LUFS`。例如：

```powershell
python generate_wavs.py --output-dir pairs_18 --count 20 --target-lufs -18 --peak-db -1
python generate_wavs.py --output-dir pairs_15 --count 20 --lufs -15
```

- 使用 pyloudnorm 的 BS.1770 综合响度测量，采用 400 ms 门控块，不以 RMS 或采样峰值代替 LUFS。
- harmonic 与 noise 使用相同的全局增益。瞬态过高时，使用共享的平滑限幅增益曲线，随后重新测量并迭代调整响度。
- `--peak-db` 默认为 `-1`，约束两路源和混合音频的采样峰值及 4 倍过采样峰值估计。另保留 0.3 dB 余量，减小不同重建滤波器的估计差异。
- 混合文件最后由两个 float32 源相加生成。不能分别把三路各自归一化到同一个 LUFS，否则会改变监督标签对应关系。harmonic/noise 的独立响度自然可以不同。
- `metadata.json` 记录目标与实测 LUFS、峰值、限幅状态及实际 SNR。`status=ok` 表示与目标相差不超过 0.15 LU；若极端信号在峰值约束下无法达到目标，返回 `peak_limited` 并记录实际值，优先保证峰值安全。
- 导出脚本要求时长至少 0.4 秒。底层生成器支持更短的音频，但这类片段只作峰值保护，记录 `short_clip_peak_only`；静音不赋予虚假的 LUFS 值。

### 其他参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--count` | `10` | 生成组数 |
| `--duration` | `5` | 每组秒数；样本数为 `round(duration * sr)` |
| `--sr` | `44100` | 采样率，至少 8000 Hz |
| `--hopsize` | `512` | 基础谱包络和 F0 控制步长 |
| `--seed` | `42` | 基础随机种子 |
| `--start-index` | `0` | 起始目录编号，用于续生成 |
| `--snr-db MIN MAX` | `-5 20` | 限幅前 harmonic/noise 的全片 RMS 比值范围，噪声包括全部辅音成分 |
| `--consonants ...` | 全部 | 指定需要采样的辅音类型 |
| `--no-consonants` | 关闭 | 关闭辅音增强，便于对照实验 |
| `--overwrite` | 关闭 | 明确覆盖所选编号中已有的输出文件 |

每个样本的随机种子由基础种子与编号推导。在相同依赖版本下，分批生成与一次生成的相同编号可逐字节复现：

```powershell
python generate_wavs.py --output-dir pairs --count 100 --start-index 100 --seed 42
```

若目录已存在，默认报错，避免覆盖已有训练数据。`--no-consonants` 仍使用当前长度修复和响度控制，不保证与旧版本逐字节一致。

## 辅音生成的改动

原始 noise 只对白噪声施加缓慢变化的频谱包络，并统一衰减高频；harmonic 的有声/无声掩码则独立生成。这种方式没有显式的摩擦段、闭塞—爆破—送气段，也没有与有声部分对应的时序。

现在由 `consonants.py` 在采样级生成共同的辅音—元音事件计划：

| 类型 | 近似音类 | harmonic 目标 | noise 目标 |
| --- | --- | --- | --- |
| 清摩擦音 | `s sh f` | 该段关闭周期激励 | 各自不同的频带噪声与短时包络 |
| 送气音 | `h` | 该段关闭周期激励 | 带宽与共振峰状谱包络的气流噪声 |
| 浊摩擦音 | `z zh v` | 保留较弱周期激励 | 同时保留摩擦噪声 |
| 清塞音 | `p t k` | 闭塞、释放、送气段关闭周期激励 | 短爆破与随后送气，并带元音起始处的送气尾部 |
| 浊塞音 | `b d g` | 闭塞和送气阶段允许弱周期激励 | 爆破、较短送气 |
| 塞擦音 | `ts ch` | 闭塞、爆破、摩擦段关闭周期激励 | 爆破接持续摩擦 |
| 元音上下文 | 随机谐波段 | 周期成分 | 同步的轻微气声 |

辅音的时长、强度、谱形和包络随机变化，`s` 等高频摩擦成分不再被背景噪声的低通倾向统一压掉。所有摩擦、爆破、送气和元音气声均归入 noise；浊辅音的周期成分归入 harmonic。事件开始/结束样本、音类及增益信息保存在 metadata 中，结束位置为开区间。

这是用于增加训练分布覆盖的声学近似，不能当作可懂度自然语音或严格音素标注语料。实际辅音分离能力仍需通过训练和真实录音评估。

同时修复了基础生成器的长度漂移、短音频控制点不足、F0 抖动频率单位错误、零步长、整数掩码平滑、插值振幅过冲和滑音时高次谐波越过 Nyquist 的问题。Matplotlib 改为仅绘图函数内部导入。

## Python 中直接生成

```python
from paired_generator import PairedGenerator

synth = PairedGenerator(
    sr=44100,
    random_seed=42,
    target_lufs=-16.5,
    peak_db=-1.0,
    consonants=["s", "sh", "f", "t", "k", "ts"],
)
harmonic, noise, mixture = synth.generate(n_samples=44100 * 5)
# 均为 (220500,) 的 np.float32，mixture == harmonic + noise。
print(synth.last_metadata["loudness"])
```

`GeneratorNoise.generate_noise()` 单独使用时也默认加入辅音噪声，但它无法获知另一路 harmonic 的时序。训练与成对导出应使用 `PairedGenerator` 或已接入该逻辑的 Dataset。基础单源生成函数不单独做 LUFS 归一化。

## 现有训练接口

```python
from dataset import GeneratedHarmonicNoiseSplitTrainingSet
from random_harmonic import GeneratorHarmonic
from random_noise import GeneratorNoise

train = GeneratedHarmonicNoiseSplitTrainingSet(
    harmonic_generator=GeneratorHarmonic(random_seed=42),
    noise_generator=GeneratorNoise(random_seed=527),
    sr=44100,
    hop_length=512,
    cropsize=256,
    copy_rate=0.05,
    mixup_rate=0.2,
    mixup_alpha=0.4,
    mode="noise_as_inst",
    random_seed=42,
    target_lufs=-16.5,
    peak_db=-1.0,
)
X, y = train[0]
```

- 两种 `mode` 的监督语义保持一致；返回类型仍为 NumPy 数组。
- 默认启用辅音增强。设 `consonant_augmentation=False` 可关闭。
- 默认在 copy/mixup 完成后对最终混合音频控制响度。设 `target_lufs=None` 可恢复原来的随机整体音量增强。
- 验证集新增可选 `hop_length`，默认从 harmonic 生成器获取，修复原先把 `cropsize` 当作 hop length 的长度计算错误。
- 不安装 torch 时，Dataset 仍可按序列使用；安装 torch 后继承真正的 `torch.utils.data.Dataset`。导出脚本完全不导入该模块。
- 传入自定义基础生成器时，必须返回精确长度的有限单声道数组，且采样率一致。它们自己的随机状态仍由调用方管理；内置生成器的随机种子可在创建时设置。

## 单线程生成速度

现有导出命令自动使用优化后的算法，无需增加 worker、GPU 或修改音频参数。谐波合成按采样点递推各阶相位，共享 PCHIP 插值系数；峰值保护使用同一套 4 倍过采样滤波器，并在最终 float32 三路上核验。Numba 首次调用时编译单线程内核并缓存，因此批量生成和首次启动的耗时不同。

```powershell
.\.venv\Scripts\python.exe benchmark_generation.py --profile
```

基准固定数值库使用一个线程，默认种子 42、每组 5 秒、44.1 kHz、−16.5 LUFS，串行导出 4 组、重复 3 轮。计时包含合成、限幅、三路 WAV 和元数据写盘；启动和首组单独计时。当前开发机同配置的轮次中位数从 **1.620 秒/组降至 0.417 秒/组**（约 3.9 倍）；音高、限幅迭代次数和磁盘速度会影响结果。浮点运算顺序改变，不保证与旧版文件逐字节一致。

## 检验

```powershell
python -m unittest discover -s tests -v
```

测试包括清/浊辅音标签、闭塞/爆破/送气时序、高频摩擦噪声、短音频与多采样率、随机种子复现、训练/验证形状、copy/mixup、LUFS 与峰值限制、保存回读后的三路相加关系，以及阻止 torch/Matplotlib 导入时的 CLI 独立运行。

可用 FFmpeg 独立测量导出的混合文件：

```powershell
ffmpeg -hide_banner -nostats -i pairs/000000/mixure.wav -af ebur128=peak=true -f null NUL
```

试听文件不纳入版本控制。运行 `python generate_wavs.py --output-dir generated_audio --count 3 --duration 5 --sr 44100 --target-lufs -16.5` 可生成三组试听样本。
