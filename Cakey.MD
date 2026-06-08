CaKey 结构与 Kimodo 接入详细说明
面向计算机小白的逐步技术文档：为什么要加 CaKey、CaKey 内部怎么工作、如何接到 Kimodo root/body Transformer、如何训练与检查

适用背景：你已经尝试了只加入 SceneCo / Voxel ViT 的方案，但在 Stage 1 或路径生成阶段出现了路径不规则、轨迹抖动、关键帧之间过渡不自然的问题。因此本文件重点解释 SceneAdapt 中 CaKey 的结构，并给出它在 Kimodo 中的推荐接入方式。
一句话结论：CaKey 不是场景编码器。CaKey 的任务是先把“给定关键帧/路径点之后，中间动作怎么自然补出来”这件事稳定下来。只有 Stage 1 的 inbetweening 稳定，Stage 2 的 SceneCo 才更容易专心学习场景。


0. 先给你一个最短结论
问题	结论
CaKey 是什么？	一个插在 Transformer 每层 Self-Attention 和 FFN 之间的小模块。它根据关键帧信号、扩散时间 t、当前 latent a，生成缩放 γ 和平移 β，然后只调制关键帧 token。
它解决什么问题？	解决关键帧/路径点之间的 inbetweening 不稳：比如路径不顺、速度忽快忽慢、关键帧附近抖动。
它和 SceneCo 的关系？	CaKey 先学会“补中间动作”；SceneCo 后学会“根据场景改变动作”。SceneAdapt 原方法就是 Stage 1 训练 CaKey，Stage 2 冻结 CaKey 再训练 SceneCo。
Kimodo 中先加哪里？	先加 root_model。因为你现在主要问题是路径不规则，root 控制全局位置和朝向。root 稳定后，再考虑 body_model。
训练哪些参数？	Stage 1 冻结 Kimodo 原参数，只训练 CaKey；Stage 2 冻结 Kimodo + CaKey，只训练 Voxel ViT + SceneCo。

1. 为什么你现在需要 CaKey
你观察到 Stage 1 生成的路径不规则。这个现象通常说明：模型虽然看到了约束 mask 或路径点，但是没有形成稳定的“从关键帧到关键帧”的补帧能力。
Kimodo 原本支持约束输入：文本、关键帧、末端约束、2D path、2D waypoint 等。Kimodo 官方文档说明，它在去噪时会接收 text prompt、kinematic constraints 和 noisy motion，约束会覆盖 noisy motion 中对应位置，并且 mask 会拼接到输入中；随后 two-stage transformer denoiser 预测 clean motion。
但是“能输入约束”不等于“在你的新数据、新场景、新路径格式下总能稳定补出自然轨迹”。如果你引入了 LINGO、体素场景、TSTMotion 风格路径规划，约束分布可能和 Kimodo 原训练分布不同。这时只靠 mask concat/overwrite 可能不够，需要一个专门适配关键帧/路径补全的 adapter，也就是 CaKey。
2. SceneAdapt 中 CaKey 的原始设计
SceneAdapt 的核心是两阶段适配：Stage 1 加 CaKey 学 motion inbetweening；Stage 2 冻结 CaKey，再加 SceneCo 学 scene-aware inbetweening。论文中说 CaKey 是 Context-aware Keyframing layer，用于只调制 keyframe latents，让模型能做准确的 inbetweening，同时尽量不破坏原来的 text-to-motion latent manifold。
CaKey 的位置：插在每个 Transformer block 的 Self-Attention 之后、FFN 之前。论文附录也说明 CaKey 每层都在 self-attention block 之后调制 latents。
CaKey 的输入：
a：当前 self-attention 输出后的 motion latent。可以理解为 Transformer 当前层已经处理过一次上下文后的动作特征。
x_key：关键帧动作信号。训练时来自 ground-truth motion 中被 mask 选中的关键帧；推理时来自用户给定的关键帧、路径点、目标点等。非关键帧位置填 0 或保留占位。
m：关键帧 mask。m=1 表示这一帧是关键帧，需要被 CaKey 调制；m=0 表示不是关键帧，尽量保持原 latent 不变。
t：扩散 timestep。不同去噪阶段噪声强度不同，所以调制方式也应该跟 t 有关。
CaKey 的输出：
γ：scale，表示每个 latent 通道要放大或缩小多少。
β：shift，表示每个 latent 通道要平移多少。
out：调制后的 latent，只有关键帧 token 被改变，非关键帧 token 保持原样。

图 1：CaKey 的计算流程。关键点不是“全局改变整个动作”，而是“只在 keyframe token 处做稀疏调制”。
3. CaKey 的数学公式：逐项解释
SceneAdapt 中 CaKey 使用两个可学习的 MLP：fθ 和 hφ。它们根据关键帧动作、扩散时间、当前 latent 输出 γ 和 β。论文给出的形式是：
gamma = f_theta(x, t, a)
beta  = h_phi(x, t, a)

a_hat = gamma * a + beta
CaKey(a, m, x, t) = (1 - m) * a + m * a_hat


这里每个符号的直接含义如下：
符号	shape 示例	含义	在 Kimodo 中怎么拿
a	[B, N, D]	当前层 self-attention 后的 motion latent	Transformer layer 中 self-attention 输出后，取 motion tokens，不取 text/timestep/heading prefix
x_key	[B, N, F_key] 或 [B, N, D]	关键帧信号。只有 m=1 的位置是真值/约束，其它位置填 0	由 observed_motion 和 motion_mask 构造；路径点可以转成 root position keyframes
t	[B, D] 或 [B, 1, D]	扩散时间嵌入	Kimodo 原本已有 timestep embedding，扩展到每一帧
m	[B, N, 1]	帧级关键帧 mask	从 motion_mask 压成 frame-level mask。只要该帧 root/path/pose 有约束，就记为 1
γ	[B, N, D]	scale 调制参数	MLP 输出，建议初始化为 1 附近
β	[B, N, D]	shift 调制参数	MLP 输出，建议初始化为 0

最重要的一行是最后一行：CaKey(a, m, x, t) = (1-m)*a + m*a_hat。它保证非关键帧位置不被 CaKey 直接修改。这就是 sparse modulation。论文消融表明，去掉 sparse modulation 会让 inbetweening 质量严重下降。
4. 为什么 CaKey 不是普通 FiLM
普通 FiLM 通常是从一个条件向量生成 γ 和 β，然后调制整个 latent。CaKey 有两个关键差异：
差异	普通 FiLM	CaKey
是否看当前 latent a	通常不看，或只看外部条件	看当前 self-attention activation a。这样调制器知道自己正在修改什么。
是否看扩散 timestep t	不一定	必须看。因为去噪早期和后期需要不同强度的调制。
是否只改关键帧	一般会改所有 token	只改 m=1 的 token，非关键帧保持 identity。
适合什么任务	通用条件控制	关键帧 / 路径点 / inbetweening 适配。

因此 CaKey 的名字里有 Context-aware：它不是盲目地把关键帧条件加进去，而是结合当前 latent、时间步和关键帧信号来决定如何调制。
5. CaKey 在 Kimodo 中应该接在哪里
Kimodo 的 two-stage denoiser 先用 root denoiser 预测 global root motion，再把 root 转成 local representation 作为 body denoiser 的输入，最后拼接 root 和 body 输出。Kimodo 文档明确说明输入是 text prompt + optional constraints，输出是 full-body motion sequence，核心是在每个 diffusion step 用 two-stage transformer denoiser 去噪。
你的现象是路径不规则，路径主要由 root motion 决定。所以第一版不要一开始就把 CaKey 加满 root 和 body，而是按下面顺序做：
阶段	加在哪里	原因
Stage 1A	只在 root_model 每层加 CaKey	优先解决全局路径、速度、朝向过渡不稳。参数少，最容易定位问题。
Stage 1B	root_model + body_model 都加 CaKey	如果 root 路径变顺，但身体姿态在关键帧附近抖动，再加 body CaKey。
Stage 2	冻结 CaKey 后加 SceneCo	CaKey 负责补帧稳定，SceneCo 负责根据体素场景调整动作。


图 2：在 Kimodo 中推荐先给 root_model 加 CaKey。root 稳定后，再决定 body_model 是否也需要 CaKey。
6. Kimodo 中 CaKey 的张量设计
下面给出一个可直接按代码实现的张量约定。假设 batch size 是 B，帧数是 N，Kimodo Transformer latent 维度是 D_model。
变量	shape	来源	说明
motion_tokens	[B, N, D_model]	input_linear(noisy_motion)	进入 Transformer 的动作 token。
prefix_tokens	[B, P, D_model]	text / timestep / first heading 等	不要对 prefix 做 CaKey。CaKey 只处理 motion tokens。
a_motion	[B, N, D_model]	self-attention 后取 motion 部分	CaKey 的主输入 a。
observed_motion	[B, N, F_motion]	Kimodo 原约束管线	只有被约束的位置是真值，其他位置可以是 0。
motion_mask	[B, N, F_motion]	Kimodo 原约束 mask	元素级 mask。需要转换成 frame-level keyframe_mask。
keyframe_mask	[B, N, 1]	motion_mask 聚合	只要一帧有 root/path/pose 约束，该帧就是关键帧。
key_embed	[B, N, D_model]	keyframe_encoder(observed_motion, motion_mask)	把关键帧值和 mask 编成与 latent 同维度的向量。
t_embed_frame	[B, N, D_model]	timestep embedding expand	把 timestep embedding 复制到每一帧。

7. keyframe_mask 怎么从 Kimodo 约束得到
Kimodo 原本的 motion_mask 可能是元素级别的。例如一帧里只约束 root x/y，或者只约束某个手的位置。CaKey 需要的是帧级 mask，因为它按 token 调制。最简单做法如下：
# motion_mask: [B, N, F_motion]
# 只要某一帧有任何维度被约束，就认为这帧是 keyframe
keyframe_mask = (motion_mask.abs().sum(dim=-1, keepdim=True) > 0).float()
# keyframe_mask: [B, N, 1]


如果你只想让 root 路径影响 root_model，可以只聚合 root 相关维度：
# root_indices 是 root position / root heading / root velocity 等维度下标
root_mask = motion_mask[..., root_indices]
keyframe_mask_root = (root_mask.abs().sum(dim=-1, keepdim=True) > 0).float()


建议第一版用 root_mask，而不是 full motion_mask。原因是你要先解决路径不规则。如果一开始把手、脚、身体约束全部变成 keyframe mask，CaKey 学到的信号会混乱。
8. x_key 应该是什么：不要把整段 GT 都喂进去
SceneAdapt 论文公式写的是 x，但在 inbetweening 场景里真正可用的是“被 mask 选中的关键帧值”。推理时你不可能知道整段 ground-truth motion，所以实现时不能把完整 GT motion 当条件喂给 CaKey。
正确实现：
# x0 是训练数据中的 clean motion: [B, N, F]
# keyframe_mask 或 motion_mask 由随机采样生成
observed_motion = x0 * motion_mask
# 非约束位置必须为 0 或明确的 null 值，不能泄漏 GT
x_key = observed_motion


错误实现：
# 错误：这会把完整答案告诉 CaKey，训练 loss 会很好看，推理会崩
x_key = x0


这一点非常重要。你要做一个检查：把 motion_mask=0 的地方全部确认是 0，不能有 ground-truth 信息。
9. CaKey 模块的推荐代码结构
下面是推荐的 PyTorch 伪代码。它不是完整可运行代码，但每一行都对应你要实现的模块。
class CaKeyLayer(nn.Module):
    def __init__(self, motion_feat_dim, d_model=1024, hidden_dim=2048):
        super().__init__()

        # 1. 把关键帧动作值 + mask 编成 D_model 维
        self.key_encoder = nn.Sequential(
            nn.Linear(motion_feat_dim * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # 2. 用 a + key_embed + t_embed 预测 delta_gamma 和 beta
        #    不直接预测 gamma，而是预测 delta_gamma，最终 gamma = 1 + delta_gamma
        self.modulator = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model * 2),
        )

        # 3. 最后一层零初始化，保证刚插进去时 CaKey 不改变原模型
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

    def forward(self, a, observed_motion, motion_mask, keyframe_mask, t_embed):
        # a: [B, N, D_model]
        # observed_motion: [B, N, F]
        # motion_mask: [B, N, F]
        # keyframe_mask: [B, N, 1]
        # t_embed: [B, D_model] 或 [B, 1, D_model]

        B, N, D = a.shape

        if t_embed.dim() == 2:
            t_embed = t_embed[:, None, :].expand(B, N, D)
        elif t_embed.shape[1] == 1:
            t_embed = t_embed.expand(B, N, D)

        key_input = torch.cat([observed_motion, motion_mask], dim=-1)
        key_embed = self.key_encoder(key_input)

        cond = torch.cat([a, key_embed, t_embed], dim=-1)
        delta_gamma, beta = self.modulator(cond).chunk(2, dim=-1)

        gamma = 1.0 + delta_gamma
        a_hat = gamma * a + beta

        # 稀疏调制：只改 keyframe token
        out = (1.0 - keyframe_mask) * a + keyframe_mask * a_hat
        return out


10. 为什么要零初始化
CaKey 是插入到已经训练好的 Kimodo 里。如果刚初始化就随机改变 latent，模型输出会立刻变差。正确做法是让 CaKey 一开始等价于 identity。
具体来说：
modulator 最后一层 weight=0，bias=0。
delta_gamma 初始为 0，所以 gamma=1。
beta 初始为 0。
a_hat = 1*a + 0 = a。
out = (1-m)*a + m*a = a。
这样你插入 CaKey 后，在还没训练之前，模型输出应该和原 Kimodo 几乎一致。这是第一个必须通过的单元测试。
11. Transformer 层里怎么插 CaKey
因为 PyTorch 默认 TransformerEncoderLayer 不方便在 Self-Attention 和 FFN 之间插模块，所以建议把 Kimodo 的 TransformerEncoderBlock 改成自定义层。每层结构如下：
class KimodoEncoderLayerWithCaKey(nn.Module):
    def forward(self, xseq, src_key_padding_mask, pose_start_ind,
                observed_motion, motion_mask, keyframe_mask, t_embed):
        # xseq: [B, P+N, D]
        # P 是 prefix token 数量，比如 text/timestep/heading

        # 1. 原来的 Self-Attention
        xseq = xseq + self.self_attn_block(self.norm1(xseq), src_key_padding_mask)

        # 2. 只取 motion tokens 做 CaKey
        prefix = xseq[:, :pose_start_ind]
        motion = xseq[:, pose_start_ind:]

        motion = self.cakey(
            a=motion,
            observed_motion=observed_motion,
            motion_mask=motion_mask,
            keyframe_mask=keyframe_mask,
            t_embed=t_embed,
        )

        xseq = torch.cat([prefix, motion], dim=1)

        # 3. 原来的 FFN
        xseq = xseq + self.ffn_block(self.norm2(xseq))
        return xseq


关键原则：CaKey 只处理 motion tokens，不处理 text token、timestep token、heading token。否则它可能破坏文本条件和时间条件。
12. 在 root_model 和 body_model 中的不同设计
位置	CaKey 输入应该包含什么	训练目标	适合解决的问题
root_model CaKey	root observed_motion、root mask、path/waypoint 转出的 root keyframes、timestep、root latent	root keyframe error、root path smoothness、root velocity smoothness	路径不规则、轨迹绕行不顺、速度忽快忽慢、朝向抖动
body_model CaKey	body pose keyframes、joint position/rotation mask、local root condition、timestep、body latent	joint keyframe MPJPE、foot skating、姿态平滑	身体姿态关键帧不准、手脚抖动、坐下/接触动作不自然

推荐第一轮：只加 root_model CaKey。等 root 路径指标变好后，再决定是否加 body_model CaKey。
13. Stage 1 训练 CaKey 的完整步骤
Stage 1 的目的不是场景适配，而是让 Kimodo 稳定完成关键帧/路径点之间的动作补全。
步骤	你要做什么	检查点
1	加载原 Kimodo checkpoint。	确认原模型能正常生成。
2	把 CaKey 插入 root_model 每层 SA 和 FFN 之间。	gate/zero-init 后，输出应接近原模型。
3	冻结 Kimodo 原参数。	optimizer 里只能看到 cakey 参数。
4	从训练 motion 中随机采样关键帧 mask。	first 和 last frame 必须为 1，中间按 stride 采样。
5	构造 observed_motion = x0 * motion_mask。	非 mask 位置必须为 0，不能泄漏 GT。
6	训练 CaKey。	root keyframe error 和 path smoothness 应逐步下降。
7	保存 Stage 1 checkpoint。	只保存 CaKey 参数或保存完整模型但标清楚 Kimodo frozen。

SceneAdapt 原文 Stage 1 中会用固定 keyframe stride，并且首帧和末帧总是作为关键帧。论文实验中 sk=20 对应每秒一个关键帧；你的数据如果不是 20 FPS，要按 FPS 换算。
数据 FPS	一秒约几帧	建议 stride
20 FPS	20 帧	sk=20
30 FPS	30 帧	sk=30 或从 {20,30,40} 随机采样
60 FPS	60 帧	sk=60 或先降采样到 30 FPS

14. Stage 1 的 loss 设计
如果你已经有 Kimodo 的训练 loss，就先复用原 loss。额外建议加几项专门检查 root path 是否稳定。
Loss 名称	计算对象	作用	建议权重
L_x0	预测 clean motion 与 GT motion	保持整体 motion reconstruction	1.0
L_root_key	keyframe 处 root position/heading	保证关键帧位置准确	5.0
L_root_all	全部帧 root position	让中间路径接近真实轨迹	1.0
L_vel	root velocity 或 joint velocity	减少速度忽快忽慢	0.5
L_acc	root acceleration	减少轨迹突然折线	0.2
L_jerk	root jerk（三阶差分）	减少路径抖动	0.05
L_foot	foot skating / foot penetration	避免根路径变好但脚滑严重	0.1

不要一开始把 smoothness loss 权重设得太大，否则模型可能生成过度平滑、没有动作细节的轨迹。先保证 keyframe error 下降，再慢慢调 smoothness。
15. Stage 1 配置建议
stage1_cakey:
  freeze_kimodo_backbone: true
  train_modules:
    - root_model.layers.*.cakey
  optional_train_modules:
    - body_model.layers.*.cakey   # 第一轮先不要开

  optimizer: AdamW
  lr_cakey: 1.0e-4
  weight_decay: 0.01
  precision: bf16
  grad_clip_norm: 1.0

  keyframe_sampling:
    always_first_frame: true
    always_last_frame: true
    stride_choices: [20, 30, 40]
    random_drop_middle_keyframes: 0.1

  text_condition:
    first_version: null_text_only
    later_version: mix_text_and_null_text

  training_steps:
    smoke_test: 1000
    first_eval: 10000
    useful_checkpoint: 50000
    full_train: 200000


16. 什么时候说明 CaKey 成功了
检查项	通过标准
零初始化等价性	刚插入未训练时，同一个 seed、同一个输入，输出和原 Kimodo 差异接近 0。
非关键帧 identity 检查	m=0 的 token 进入/离开 CaKey 前后差异应为 0 或极小。
关键帧误差	root keyframe L2 error 明显低于未加 CaKey。
路径平滑度	root acceleration / jerk 明显下降，轨迹折线减少。
速度稳定性	root speed 标准差下降，不再忽快忽慢。
动作质量	foot skating 不应显著变差。
原 Kimodo 能力	text-only benchmark 不应明显下降，因为 Kimodo 主干被冻结。

17. CaKey 与 SceneCo 的先后关系
不要同时从零开始训练 CaKey 和 SceneCo。这样问题会混在一起：路径不规则时，你不知道是 CaKey 没学好、SceneCo 干扰了、还是体素坐标错了。
推荐严格顺序：
Stage 0: 原 Kimodo
  只做基线生成，保存 text/path/keyframe benchmark 结果。

Stage 1: Kimodo + CaKey
  冻结 Kimodo，只训练 CaKey。
  目标：路径和关键帧补全稳定。

Stage 2: Kimodo + frozen CaKey + Voxel ViT + SceneCo
  冻结 Kimodo 和 CaKey，只训练 Voxel ViT + SceneCo。
  目标：在路径稳定基础上学习避障、接触、场景一致性。

Stage 3: 消融和验收
  Original Kimodo / CaKey only / SceneCo only / CaKey+SceneCo / random scene / empty scene。


18. 常见错误与排查
现象	最可能原因	怎么检查
训练 loss 很低，推理很差	训练时把完整 GT motion 喂给 CaKey 了	检查 x_key 是否只在 mask=1 处有值。
原 Kimodo 输出立刻变差	CaKey 没有 zero-init 或改了非关键帧 token	做 gate=0 / zero-init 等价性测试。
路径仍然折线	keyframe stride 太稀或 smoothness loss 不够	画 root x-y 曲线，计算 jerk。
关键帧很准但中间动作僵硬	L_root_key 权重太大，smoothness 或 motion prior 不够	降低 L_root_key，增加 velocity/joint loss。
脚滑变严重	root 被强行拉顺，但 body 没适配	加入 foot skating loss，或第二轮给 body_model 加 CaKey。
SceneCo 训练后路径又乱了	Stage 2 没冻结 CaKey，或 SceneCo CFG 太强	检查 requires_grad；做 w_scene sweep。

19. 最小可行实现文件清单
建议不要直接大改原文件。先新增文件，能跑通后再合并。
kimodo/model/cakey.py
  - CaKeyLayer
  - KeyframeEncoder
  - build_keyframe_mask()

kimodo/model/cakey_backbone.py
  - KimodoEncoderLayerWithCaKey
  - TransformerEncoderBlockWithCaKey

kimodo/model/twostage_denoiser_cakey.py
  - TwostageDenoiserWithCaKey
  - 加载原 Kimodo 参数
  - missing keys 只允许 cakey.*

configs/stage1_cakey_root.yaml
  - root-only CaKey 训练配置

scripts/check_cakey_identity.py
  - 检查插入 CaKey 后原模型输出是否保持一致

scripts/eval_path_smoothness.py
  - 计算 root path length、speed std、acceleration、jerk

scripts/plot_root_path.py
  - 画 x-y 根轨迹图，人工检查是否折线/抖动


20. 给你的执行顺序
顺序	任务	完成标志
1	把原 Kimodo 跑通，保存 20 个固定 seed 的输出。	有原始 baseline 视频/npz。
2	实现 CaKeyLayer，但不训练。	代码能 forward。
3	插入 root_model，每层 SA 后 FFN 前。	模型能加载原 checkpoint。
4	运行 identity check。	输出差异接近 0。
5	构造 observed_motion 和 keyframe_mask。	非约束位置无 GT 泄漏。
6	跑 1000 step smoke test。	loss 正常下降，无 NaN。
7	跑 10k step，画 root path。	路径比 baseline 更顺。
8	跑 50k 或 200k step。	指标稳定改善。
9	冻结 CaKey，再接 SceneCo。	SceneCo 训练时 CaKey 参数不变。

21. 评价指标：专门判断 CaKey 有没有解决路径不规则
指标	公式/计算方式	看什么
Root Keyframe Error	在 m=1 帧计算 root position L2 距离	关键帧是否跟得准。越低越好。
All-frame Root Error	所有帧 root position L2 距离	中间轨迹是否接近真实数据。越低越好。
Path Length Ratio	生成路径长度 / GT 路径长度	是否绕路或抽搐。接近 1 最好。
Speed Std	root speed 的标准差	速度是否忽快忽慢。越低越稳，但不能低到没有动作。
Acceleration Mean	root velocity 的一阶差分均值	是否突然加速/减速。越低越平滑。
Jerk Mean	root position 三阶差分均值	是否抖动或折线。越低越好。
Turn Smoothness	heading yaw 差分的变化量	转身是否突然跳变。越低越好。
Foot Skate	脚接触地面时脚的水平速度	CaKey 不能为了路径顺而让脚滑。越低越好。
Text Regression	原 text-only benchmark 的 R-Precision/FID	确认 CaKey 没破坏原 Kimodo 语义能力。

22. 参考资料
1. SceneAdapt: Scene-aware Adaptation of Human Motion Diffusion, arXiv:2510.13044. 其中第 3.2 节解释 CaKey 的结构：使用 fθ 和 hφ 两个 MLP 从 x、t、a 生成 γ 和 β，并通过 sparse modulation 只调制 keyframe latents。
2. SceneAdapt Appendix Implementation Details. 其中说明 CaKey 使用 single-layer MLP with SiLU activations，初始化为不影响 latent，并且每层都在 self-attention block 之后调制。
3. Kimodo Model Overview. NVIDIA 文档说明 Kimodo 输入 text prompt + optional constraints，输出 full-body motion sequence，并使用 two-stage transformer denoiser：root denoiser 先预测 global root motion，body denoiser 再预测身体。
4. Kimodo Project Structure. NVIDIA 文档列出了 kimodo/model/twostage_denoiser.py、backbone.py、diffusion.py、cfg.py、motion_rep/conditioning.py 等与你修改相关的文件。