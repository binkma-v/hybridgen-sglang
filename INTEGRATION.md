# HybridGen × SGLang 集成全解

> 把 HybridGen 这个研究原型接入 SGLang 生产推理引擎的完整记录。
> 面向**完全不了解 HybridGen 的读者**——从 KV cache 是什么开始讲。

---

## 1. 这是什么文档，怎么读

这份文档讲清楚三件事：

1. **HybridGen 在解决什么问题、它的算法长什么样**
2. **SGLang 是什么、它的内部结构里哪些点和 HybridGen 重叠**
3. **这次集成具体改了哪些文件、复用了 SGLang 的哪些组件、怎么启动**

阅读地图：

| 你想知道 | 跳到 |
|---|---|
| 为什么 KV cache 装不下、要 offload | §2 |
| FlexGen 的故事 | §3 |
| HybridGen 的算法、4 个新组件 | §4 |
| SGLang 引擎结构、各种 KV cache 抽象 | §5 |
| 集成做了什么、动了哪些文件 | §6 |
| 怎么装环境、启动命令 | §7 |
| 测试 / 验证证据 | §8 |
| 没做的、followup | §9 |

---

## 2. 背景：KV cache 是怎么变成瓶颈的

### 2.1 为什么需要 KV cache

Transformer 自回归生成时，每生成一个新 token，自注意力都要把这个 token 与前面**所有**已生成 token 做交互。如果每步都重算前面所有 token 的 K/V，复杂度是 O(N²)，N 是序列长度。

工程上的标准做法：把每层每个 token 的 K（key）、V（value）算出来存起来，叫 **KV cache**。下次只算新 token 的 K/V，旧 K/V 直接从 cache 读出来 concat。这样每步推理是 O(N) 而不是 O(N²)。

### 2.2 KV cache 的容量公式

每个 token 在每一层占的 KV 字节数：

```
bytes_per_token_per_layer = 2 (K+V) × num_kv_heads × head_dim × dtype_bytes
total = bytes_per_token_per_layer × num_layers × seq_len × batch_size
```

举个例子，Llama-2-7B，FP16，单 batch，序列长度 32K：

```
2 × 32 × 128 × 2 × 32 × 32768 × 1 ≈ 17 GB
```

光是 KV cache 就吃掉 17 GB——而 7B 模型权重本身才 14 GB。当你想跑 100K context 或者并发多个请求时，**KV cache 比模型权重还大**。

### 2.3 三种应对策略

| 策略 | 思路 | 代价 |
|---|---|---|
| **Sparse attention** | 跳过不重要的 token（NSA、Quest、BigBird） | 算法层面有近似误差 |
| **Quantization** | KV 用 int8/int4 存 | 精度损失，量化反量化开销 |
| **Offloading** | 把旧 KV 摊到 CPU/disk（FlexGen、HybridGen） | PCIe 带宽是瓶颈 |

HybridGen 走的是**第三条 offloading 路**，但加了一个把"PCIe 瓶颈"变可接受的关键创新（见 §4）。

---

## 3. FlexGen：offloading 这条路是怎么开始的

### 3.1 FlexGen 是什么

FlexGen 是 2023 年 Stanford / UC Berkeley 等人发的论文（"High-Throughput Generative Inference of Large Language Models with a Single GPU"），提出在**单卡 GPU + 大量 CPU 内存 + SSD** 的硬件上跑 LLM 推理。

### 3.2 核心思路

把 GPU 当成**计算核心**，把 CPU 内存和 SSD 当成**存储扩展**，根据访问模式调度：

- 模型权重：按层分块，按需 stream 到 GPU
- KV cache：按层、按 token 分布在 GPU/CPU/disk
- Attention 计算：可以部分搬到 CPU 上做（CPU 算 attention 比从 disk 流回 GPU 快）

它做的是**throughput-oriented**：批处理大量请求时整体每秒输出 token 数高，但**单请求延迟很差**——一个 token 要等很多次 PCIe 来回。

### 3.3 FlexGen 的局限

| 问题 | 后果 |
|---|---|
| 单请求延迟差 | 不能做 chat / interactive 应用 |
| 整套调度 offline-friendly | 不适合 server 场景 |
| KV 分布粗糙 | 只是按 layer / token 区间切，没考虑 token 重要性 |

HybridGen 接力的就是**第三个**：FlexGen 没有"哪些 token 在 attention 里重要"的概念，全部塞过来；HybridGen 引入这个概念，把 PCIe 流量从"全 KV"砍到"top-k KV"。

---

## 4. HybridGen：在 offloading 之上加 top-k

### 4.1 关键洞察

观察：attention 的 softmax 输出中，**绝大部分概率质量集中在少数几个 token**（heavy-hitter 现象，H₂O 等也观察到了）。所以——

> 如果我能用便宜的方法**提前知道哪些 CPU 段 token 的 attention score 大**，我就**只把这几个 token 的 V 拉回 GPU**，其它的丢掉，软件依然能算出近似正确的输出。

PCIe 流量从"O(CPU段长度)"变成"O(top-k)"。这是 HybridGen 的卖点。

### 4.2 算法全流程（一次 decode step）

```
                ┌─────────────────┐
                │  GPU recent KV  │  最近 token，全 dense
                │   (size = R)    │
                └────────┬────────┘
                         │
                         ↓
                ┌────────────────────────────┐
                │   GPU 端 attention scores  │
                │   shape: (heads, R)        │
                └────────────┬───────────────┘
                             │
                             │  concat
                             │
   ┌──────────────────┐      │      ┌──────────────────────────────┐
   │  CPU older KV    │      │      │ CPU top-k scores + idx       │
   │   (size = C)     │──────┴──────│ shape: (heads, top_k)        │
   │                  │             │   computed in CPU memory      │
   └──────────────────┘             └──────────────────────────────┘
            │                                       │
            │  gather V                             ↓
            ↓                            ┌──────────────────────┐
   ┌──────────────────┐                  │   single softmax     │
   │   top-k V slice  │  ──────→ GPU     │  over (top_k + R)    │
   └──────────────────┘                  │   scores             │
                                         └──────────┬───────────┘
                                                    │
                                                    ↓
                                         ┌──────────────────────┐
                                         │  matmul with combined│
                                         │   V = [V_topk, V_gpu]│
                                         └──────────────────────┘
                                                    │
                                                    ↓
                                              attention output
```

### 4.3 四个组件

#### 4.3.1 分层 KV cache（GPU 段 + CPU 段）

- **GPU 段**：保留**最近** `R = gpu_cache_factor × prompt_len` 个 token 的 KV
- **CPU 段**：所有更早的 token KV
- **驱逐策略**：LRG（Least Recently Generated）—— 每生成一个新 token，最旧的 GPU token 迁移到 CPU
- **生命周期**：prefill 阶段两边都在累积；decode 阶段 GPU 段大小不变，CPU 段持续增长

#### 4.3.2 CPU 端 top-k 选择

> **术语澄清**："scores" 在本文档里特指 attention 内部的 `Q @ K^T × (1/√d_k)`，shape 为 `(heads, seq_len)`，softmax 之前的未归一化打分。**不是** LM head 的 logits（那是 vocab 维的，最后一层之后）。HybridGen 原代码里函数名 `compute_score` 也是这个 score。

每次 decode 开始时，在 CPU 上做：

```
q_cpu = Q   (post-projection, post-RoPE) → CPU
k_host = CPU 段 K
scores_cpu = q_cpu @ k_host.T          # 形状: (heads, C)
top_k_scores, top_k_idx = scores_cpu.topk(K)
```

**关键点**：这一步**整段 K 还在 CPU**，没动 PCIe。`top_k_idx` 后续用来 gather V 才触发 PCIe 传输——只传 top-k 个 token 的 V。

可选优化：`cpu_k_cap` 限制 top-k 时考虑的 CPU K 长度（取最近的 cap 个，不考虑更老的），用来在长上下文里压 CPU 计算量。

#### 4.3.3 Merged softmax（不引入近似误差的合并）

这是 HybridGen 算法上**最巧的一步**：

朴素做法是分两次 softmax（CPU top-k 一次、GPU 段一次），然后加权合并——这会**引入精度误差**因为 softmax 是非线性的。

HybridGen 做法：把 CPU top-k scores（**未做 softmax**）和 GPU 段 scores 拼在一起，**一次 softmax 覆盖所有**，然后用合并后的 V 做 matmul：

```
all_scores = concat([top_k_scores_cpu, scores_gpu], dim=seq)   # (heads, K + R)
weights = softmax(all_scores, dim=seq)
all_V = concat([V_topk, V_gpu], dim=seq)                       # (K + R, heads, head_dim)
out = weights @ all_V
```

**数学性质**：如果 top_k = C（即 CPU 段全选），这个公式与 dense full attention **数值完全等价**（误差只来自浮点精度）。

> 我们在 sglang 集成里用 PyTorch 单元测试验证了这一点：top_k = full history 时，merged softmax vs full SDPA，max diff = 1.79e-07（数值精度极限）。

所有"近似"完全来自"top-k 没选中的 token 被丢弃"，**不来自合并 softmax 算法本身**。

#### 4.3.4 反馈延迟调度器（FeedbackLatencyEstimator）

这部分是 HybridGen 的工程亮点。算法的两个关键参数（`topk_ratio`、`cpu_k_cap`）静态选不好——硬件不同最优值不同。所以做了一个**自适应控制器**：

每 N 步测一次：
- `host_latency` = CPU 端 QK^T 计算 + PCIe 传 top-k V 的总耗时
- `device_latency` = GPU 端合并 softmax + V matmul 的耗时

调整规则：

```
if host_latency <= device_latency:
    # CPU 工作被 GPU 工作"隐藏"了 — 还有余地，提精度
    topk_ratio *= 1.2  (上限 0.5)
    放宽 cpu_k_cap
else:
    # CPU 是瓶颈 — 砍工作
    topk_ratio *= 0.8  (下限 0.01)
    if 已经到下限:
        收紧 cpu_k_cap (减半)
```

两种 latency 模式：

| 模式 | 数据来源 |
|---|---|
| `estimated` | 用解析公式（FLOPs / 带宽 = latency） |
| `measured` | 实际测量的 wall-clock 时间 |

### 4.4 原 HybridGen 仓库结构

`/home/binkma/bm_ds/hybridgen` 共 **10,870 行 Python**（0 个 Triton kernel，0 个 CUDA/C++ 文件），分三个模型目录：

```
hybridgen/
├── llama/hybridgen/
│   ├── pytorch_backend.py     ~700 行  纯算术操作（compute_score, mha_gen_hybridgen）
│   ├── hybrid_llama.py        ~900 行  生成主循环、cache 管理、prefetch buffer
│   ├── llama_config.py        ~300 行  模型权重加载
│   ├── estimator.py           ~230 行  FeedbackLatencyEstimator
│   ├── timer.py
│   └── utils.py
├── opt/hybridgen/             同上 (OPT 版)
└── qwen/hybridgen/            同上 (Qwen 版)
```

三个模型的 hybrid 算法**完全相同**，差别只在权重加载和 RoPE/位置编码细节。

**没有的东西**（这点对集成方案很重要）：
- ❌ 没有 batching / 调度器
- ❌ 没有 CUDA graph
- ❌ 没有 Triton / 自定义 CUDA kernel
- ❌ 没有 server / API 层
- ❌ 没有测试 / benchmark

定位：**research-grade 单请求 reference implementation**。

---

## 5. SGLang：生产级推理引擎结构

### 5.1 两层身份

SGLang 既是引擎也是 server：

| 层 | 入口 | 用途 |
|---|---|---|
| **引擎** | `from sglang import Engine` | 嵌入到应用、batch 处理 |
| **HTTP server** | `python -m sglang.launch_server` | 生产部署（FastAPI 包装引擎） |

我们改的是**引擎内的 attention backend**，HTTP 层零改动。

### 5.2 引擎核心组件（与本次集成相关的）

```
┌─────────────────────────────────────────────────────────┐
│                  Scheduler                               │
│  (managers/scheduler.py)                                │
│   维护请求队列 + 决定每步 forward 哪些请求               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│              ModelRunner                                 │
│  (model_executor/model_runner.py)                        │
│   持有：model 权重 / KV pool / req_to_token_pool        │
│         / attention backend                              │
└────────────────────────┬────────────────────────────────┘
                         │ 每个 forward 构造 ForwardBatch
                         ↓
┌─────────────────────────────────────────────────────────┐
│            ForwardBatch                                  │
│  (model_executor/forward_batch_info.py)                  │
│   - forward_mode (extend / decode / mixed)              │
│   - seq_lens / req_pool_indices / out_cache_loc          │
│   - token_to_kv_pool / req_to_token_pool                 │
│   - attn_backend                                         │
└────────────────────────┬────────────────────────────────┘
                         │ 模型 forward
                         ↓
┌─────────────────────────────────────────────────────────┐
│           Model (e.g., LlamaForCausalLM)                 │
│   每个 LlamaAttention 调用：                             │
│     RadixAttention(q, k, v, forward_batch)               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│           RadixAttention (layers/radix_attention.py)     │
│   多态派发到：                                           │
│     forward_batch.attn_backend.forward(...)              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│       AttentionBackend (subclass)                        │
│   - flashinfer / triton / torch_native / hybrid_kvcache  │
│     ← 我们的就插在这一层                                 │
└──────────────────────────────────────────────────────────┘
```

### 5.3 KV pool 抽象

SGLang 把 KV 存储分了好几层抽象，本次集成相关的是：

| 抽象 | 文件 | 用途 |
|---|---|---|
| `MHATokenToKVPool` | `mem_cache/memory_pool.py` | **GPU 上**的 MHA KV 池（按 token 索引，paged） |
| `MHATokenToKVPoolHost` | `mem_cache/memory_pool_host.py` | **CPU pinned 内存上**的 MHA KV 池，可零拷贝传 GPU |
| `req_to_token_pool` | `mem_cache/memory_pool.py` | 全局映射 (request 池索引, token 位置) → KV 池 token id |

### 5.4 上层 KV 系统：HiCache 与 HiSparse

SGLang 在底层 KV pool 之上有两套独立的高层系统，**容易混淆，搞清楚很重要**：

#### HiCache（Hierarchical Cache）

- **作者**：Zhiqiang Xie @ Stanford，PR #2693（2025-02）
- **目的**：**跨请求复用 prefix KV**（多轮对话共享 system prompt、共享文档前缀等）
- **结构**：L1 GPU + L2 host RAM + L3 分布式存储（Mooncake / 3FS / NIXL）
- **核心抽象**：`HiRadixCache`（在 RadixCache 基础上扩展）
- **不适合 hybridgen**：HiCache 的语义是"按 prefix 共享"，不是 per-request 的"自己的旧 KV offload"

#### HiSparse（Hierarchical Sparse Attention）

- **作者**：同一作者，PR #20343（2026-03）
- **目的**：给 sparse attention 算法（NSA、Quest）提供分层 KV 基建
- **结构**：HiSparseCoordinator + HiSparseNSATokenToKVPool（仅 NSA pool）
- **不适合 hybridgen**：HiSparse 假定**全 sparse**；hybridgen 的 GPU 段是 dense 的；且只接 NSA pool（DeepSeek-DSA 专用）

#### 真正适合 hybridgen 的：底层 `MHATokenToKVPoolHost`

- 它就是个 **CPU KV 内存分配器**，没绑定 prefix / sparse 任何上层语义
- HiCache 用它作为 L2，HiSparse 用它作为 staging buffer，我们也可以**直接用它做 per-request offload**

---

## 6. 集成实现：怎么把 HybridGen 装进 SGLang

### 6.1 总体策略

**核心原则**：SGLang 已经做的（模型加载、调度、forward 流水、ForwardBatch、KV pool 分配、HTTP 服务）**全部复用**。我们只新写 attention backend 这一个文件，加上配套的 top-k 算法模块和 feedback 控制器。

**跳过的事**（明确不做）：
- 不改模型代码（`LlamaForCausalLM` 等）—— `RadixAttention` 多态派发自动生效
- 不改 Scheduler / ModelRunner / ForwardBatch（除了 backend 注册）
- 不写 Triton/CUDA kernel（先用 PyTorch 跑通，与 hybridgen 原版对齐）
- 不集成进 HiCache 或 HiSparse 的 RadixTree——直接用底层 host pool

### 6.2 关键架构决策：不"插入 decode loop"，而是"挂在 sglang 的 hook 上"

集成的第一个直觉常常是："找到 sglang 的 decode 主循环，把 hybridgen 的逻辑塞进去"。这条路**不走得通**，原因是 sglang **不存在单一可塞入的 decode loop**——它的循环逻辑分散在多个组件里：

| 阶段 | 在哪里 |
|---|---|
| 请求批组成、决定本步 forward 哪些 request | `Scheduler.event_loop` (`managers/scheduler.py`) |
| 构造 ForwardBatch、调模型 | `ModelRunner.forward_batch_generation` |
| 模型逐层 forward | `LlamaModel.forward` |
| 每层 attention 算 | `LlamaAttention.forward` → `RadixAttention.forward` |
| Attention kernel 实际计算 | `attn_backend.forward_decode/extend` |

要"塞入" loop 必须修改其中某一处，而这正是我们要避免的——动 sglang 核心代码就难维护、难升级。

**SGLang 已经为这种"在 loop 关键时刻插入逻辑"的需求预留了扩展点**，就是 `AttentionBackend` 接口。它对外暴露几个 hook，每个对应 hybridgen 原 loop 中的一个时刻：

| HybridGen 原 loop 中的位置 | SGLang 对外暴露的 hook | 我们的实现 |
|---|---|---|
| 每步 decode 开始 — 决定要不要驱逐旧 token 到 host | `attn_backend.init_forward_metadata(forward_batch)`<br>每个 forward batch 调 1 次（layer 之前） | `_maybe_evict_per_request()` |
| 每层 — 算 attention（dense GPU + top-k host） | `attn_backend.forward_decode(q, k, v, layer, forward_batch)`<br>每层调 1 次 | `_run_hybrid_decode()` |
| 每步 decode 结束 — 测延迟、调 topk_ratio | 在 `forward_decode` 里检查 `layer.layer_id == last_layer_id` 自己判断 | `_maybe_step_feedback()` |
| KV 写入 cache | `forward_batch.token_to_kv_pool.set_kv_buffer()`<br>**SGLang 自动处理** | 复用，不写 |
| 请求结束、req_pool_idx 复用 | 在 `init_forward_metadata` 里检测 `extend_seq_lens[i] == seq_lens[i]` | `_reset_finished_requests()` |

**结果**：

```
hybridgen 原版 (research-grade prototype):
   ┌────────────────────────────────────────┐
   │  自己写的 generation_loop_hybridgen     │
   │   ├── load_cache       (load V)         │
   │   ├── compute_score    (CPU top-k)      │
   │   ├── mha_gen_hybridgen(merged softmax) │
   │   ├── store_cache      (evict)          │
   │   └── feedback_update  (调参数)         │
   └────────────────────────────────────────┘

我们的集成 (sglang-native plugin):
   sglang 自有 generation loop
   │  for each forward step:
   │    ├── attn_backend.init_forward_metadata()   ← 我们在这里 evict + reset
   │    │
   │    ├── for each layer:
   │    │     └── attn_backend.forward_decode()    ← 我们在这里 hybrid attention
   │    │           │
   │    │           └── (last layer) feedback step ← 我们在这里调参数
   │    │
   │    └── set_kv_buffer (sglang 自动)
   │
   └── 我们没写一行 loop 代码
```

逻辑结构 1:1 对应，但物理上：**hybridgen 原版自己开餐厅（从买菜到上菜全做）；我们的集成是去 sglang 餐厅当厨师——服务员、收银、清洁都是 sglang 的人，但主菜是我们做的**。

代码量数据印证这个差别：

| 指标 | hybridgen 原版 | 我们的集成 |
|---|---|---|
| 总行数 | 10,870 行 | 1,087 行 |
| 模型代码 | 自己重新实现 3 份 (Llama/OPT/Qwen) | 0 行（复用 sglang 全部模型） |
| Loop 代码 | 自己写 generation_loop | 0 行（复用 sglang scheduler） |
| KV pool 代码 | 自己写 cache 管理 | 0 行（复用 `MHATokenToKVPool`、`MHATokenToKVPoolHost`） |
| HTTP 服务 | 无 | 0 行（复用 sglang FastAPI 层） |

**这就是"hook 注入"比"loop 注入"省 10× 代码的原因**——我们没有重新发明 sglang 已经做好的所有"周边设施"。

### 6.3 集成点（精确文件 + 行号）

基于 **sglang v0.5.10.post1** (cu12 时代最后一个 tag, commit `7c35342c1`)。

#### A. 抽象基类

| 文件 | 行号 | 内容 |
|---|---|---|
| `python/sglang/srt/layers/attention/base_attn_backend.py` | 17 | `class AttentionBackend(ABC)` 基类 |
| 同上 | 21 | `init_forward_metadata(forward_batch)` |
| 同上 | 79+ | `forward()` 派发到 `_decode/_extend/_mixed` |
| 同上 | 123 | `forward_decode(q, k, v, layer, forward_batch, save_kv_cache)` |
| 同上 | 135 | `forward_extend(...)` |

#### B. Backend 注册

| 文件 | 行号 | 用途 |
|---|---|---|
| `python/sglang/srt/layers/attention/attention_registry.py` | 12 | `ATTENTION_BACKENDS = {}` |
| 同上 | 15 | `register_attention_backend(name)` 装饰器 |
| 同上 | 105 | `create_torch_native_backend` —— 我们仿这个写工厂 |

#### C. CLI / ServerArgs

| 文件 | 行号 | 用途 |
|---|---|---|
| `python/sglang/srt/server_args.py` | 127 | `ATTENTION_BACKEND_CHOICES` 列表 |
| 同上 | 588 | `enable_double_sparsity` —— 我们仿其样板加 hybridgen_* 字段 |
| 同上 | 4685 | `--decode-attention-backend` argparse —— 我们插在它附近加新选项 |

#### D. 模型层调用（不改）

| 文件 | 行号 | 关键 |
|---|---|---|
| `python/sglang/srt/layers/radix_attention.py` | 47 | `class RadixAttention` |
| 同上 | 99–137 | `forward()` 派发到 `forward_batch.attn_backend.forward()` |
| `python/sglang/srt/models/llama.py` | 240 | `attn_output = self.attn(q, k, v, forward_batch)` —— 唯一 attention 调用点，**零改动** |

#### E. KV pool（直接复用）

| 文件 | 行号 | 用途 |
|---|---|---|
| `python/sglang/srt/mem_cache/memory_pool.py` | 697 | `class MHATokenToKVPool` (GPU) |
| 同上 | 663 | `get_key_buffer(layer_id)` |
| 同上 | 667 | `get_value_buffer(layer_id)` |
| 同上 | 951 | `set_kv_buffer(layer, loc, k, v)` |
| `python/sglang/srt/mem_cache/memory_pool_host.py` | 274 | `class MHATokenToKVPoolHost` (CPU) |
| 同上 | 256 | `alloc(need_size)` |
| 同上 | 286 | `free(indices)` |
| 同上 | 512 | `backup_from_device_all_layer` —— **核心**: GPU→host 全层拷贝 |
| 同上 | 397 | `load_to_device_per_layer` —— host→GPU 单层拉回 |

#### F. ForwardBatch（用其字段）

| 文件 | 行号 | 字段 |
|---|---|---|
| `python/sglang/srt/model_executor/forward_batch_info.py` | 322 | `token_to_kv_pool: KVCache` |
| 同上 | 323 | `attn_backend: AttentionBackend` |
| 同上 | (运行时) | `req_pool_indices`, `seq_lens`, `out_cache_loc`, `extend_seq_lens` |

#### G. 模板参考：DoubleSparseAttnBackend

| 文件 | 内容 |
|---|---|
| `python/sglang/srt/layers/attention/double_sparsity_backend.py` | 仅 257 行，是"top-k 重要 token attention"的最直接前例 |

DoubleSparseAttnBackend 是 enable 通过 feature flag (`enable_double_sparsity`)，绑定 triton backend；我们做成顶级 backend `--attention-backend hybrid_kvcache`，更直观。

### 6.4 我们新增 / 修改的文件

#### 新增文件（共 1010 行）

##### G.1 `python/sglang/srt/layers/attention/hybrid_kvcache_backend.py` (593 行)

核心 backend 类。与原 hybridgen 的对应关系：

| 原 hybridgen 函数 | 本文件中的对应物 |
|---|---|
| `pytorch_backend.compute_score_from_q` | 通过 `hybridgen_topk.compute_topk_on_cpu` 调用 |
| `pytorch_backend.mha_gen_hybridgen` 的合并 softmax 主体 | 通过 `hybridgen_topk.merged_softmax_attention_per_head_v` 调用 |
| `hybrid_llama.store_cache_hybridgen` (KV 驱逐) | `_maybe_evict_per_request()` |
| `hybrid_llama.load_cache_hybridgen` (top-k V 拉回) | `_run_hybrid_decode()` 内联实现 |
| `hybrid_llama.feedback_update` | `_maybe_step_feedback()` |
| `init_cache_one_gpu_batch_hybridgen` (host pool 预分配) | `_ensure_host_pool()` |

主要方法：

```
HybridKVCacheAttnBackend(AttentionBackend)
├── __init__                          读 9 个 hybridgen_* 配置；构造 fallback (TorchNativeAttnBackend)
├── _build_feedback_controller        从 model_config 读 D / N_HEADS / N_LAYERS 构造 estimator
├── _ensure_host_pool                 从 forward_batch.token_to_kv_pool 懒构造 MHATokenToKVPoolHost
├── _alloc_host / _free_host          page-aligned 分配
├── _backup_layer_to_host             封装 backup_from_device_all_layer
├── _load_layer_from_host             封装 load_to_device_per_layer
├── _reset_finished_requests          检测 fresh prefill (req_pool_idx 复用)，清 stale state
├── _maybe_evict_per_request          **核心**: GPU 段 > gpu_cap 时驱逐到 host
├── _maybe_step_feedback              **核心**: 调用 controller，更新 topk_ratio / cpu_k_cap
├── _run_hybrid_decode                **核心**: 每请求 hybrid 路径 — top-k + merged softmax
├── init_forward_metadata             先 ensure_host_pool，再 reset/evict，再委托 fallback
├── forward_extend                    委托 fallback (prefill 走 dense)
├── forward_decode                    无 host 段→fallback；有 host 段→_run_hybrid_decode；末层后 _maybe_step_feedback
└── support_triton                    return False
```

##### G.2 `python/sglang/srt/layers/attention/hybridgen_topk.py` (449 行)

**纯算术**模块。函数：

```python
def compute_topk_on_cpu(
    q: Tensor,          # (num_q_heads, head_dim) on CPU
    k_host: Tensor,     # (S_host, num_kv_heads, head_dim) on CPU
    top_k: int,
    num_q_heads: int, num_kv_heads: int, scaling: float,
    workspace: Optional[CPUTopKWorkspace] = None,
) -> tuple[Tensor, Tensor]:  # (top_k_scores, top_k_indices)

def merged_softmax_attention_per_head_v(
    q: Tensor,                      # (num_q_heads, head_dim) on GPU
    k_gpu, v_gpu: Tensor,           # (S_gpu, num_kv_heads, head_dim) on GPU
    cpu_topk_scores: Optional[Tensor],  # (num_q_heads, k_eff) on GPU
    v_topk_per_head: Optional[Tensor],  # (num_q_heads, k_eff, head_dim) on GPU
    num_q_heads: int, num_kv_heads: int, scaling: float,
) -> Tensor:  # (num_q_heads, head_dim)

def merged_softmax_attention_per_head_v_triton(...)
    # Triton fused: QK^T(GPU) + merged softmax + weighted V sum

class CPUTopKWorkspace:
    # 复用 q_grouped / scores_grouped / topk_scores / topk_indices

def merged_softmax_attention(...)
    # 保留旧 shared-unique-column 版本，供单测对照

def _maybe_repeat_kv(...)  # GQA 助手：扩展 K/V 头维度
```

数学等价性：

> `merged_softmax_attention_per_head_v` 与旧的 `unique + scatter + merged_softmax_attention` shared-column 表示数值一致；新路径避免了 decode 热路径里的 `torch.unique`、inverse index、`scatter_` 和 `-inf` score remap。`merged_softmax_attention_per_head_v_triton` 进一步把 GPU 段 QK、merged softmax、weighted V sum 融合进单个 Triton kernel；单段 microbench 约 `0.15ms -> 0.05ms`。

##### G.3 `python/sglang/srt/layers/attention/hybridgen_feedback.py` (267 行)

**纯 Python**（不依赖 torch）。组件：

```python
@dataclass class ModelSpecs    # B / D / N_HEADS / N_LAYERS / D_FFN / DTYPE_BYTES
@dataclass class SystemSpecs   # cpu_power / gpu_power / pcie_speed (FLOPs/s, B/s)

# 解析延迟模型（FLOP 累加 + 带宽除法）
def host_side_latency(seq_len, topk, model, sys) -> float
def device_side_latency(gpu_seq_len, cpu_seq_len, topk, model, sys) -> float

@dataclass class FeedbackLatencyEstimator   # estimated 模式 vs measured 模式
@dataclass class FeedbackPolicy             # min/max ratio, grow_factor, shrink_factor

def adapt(topk_ratio, cpu_k_cap, cpu_cache_len, host_lat, device_lat, policy):
    """完全移植自 hybrid_llama.py:540 的 feedback_update 决策逻辑"""

@dataclass class FeedbackController:
    estimator: FeedbackLatencyEstimator
    policy: FeedbackPolicy
    interval: int
    def maybe_step(...) -> (new_topk_ratio, new_cpu_k_cap)
```

#### 修改文件（共 +77 行）

##### G.4 `python/sglang/srt/layers/attention/attention_registry.py` (+9 行)

在 `create_torch_native_backend` 后插入：

```python
@register_attention_backend("hybrid_kvcache")
def create_hybrid_kvcache_backend(runner):
    from sglang.srt.layers.attention.hybrid_kvcache_backend import (
        HybridKVCacheAttnBackend,
    )
    return HybridKVCacheAttnBackend(runner)
```

##### G.5 `python/sglang/srt/server_args.py` (+68 行)

三处修改：

1. `ATTENTION_BACKEND_CHOICES` 列表加 `"hybrid_kvcache"`
2. 数据类字段段加 9 个 `hybridgen_*` 配置（仿 `enable_double_sparsity` 模式）：
   ```
   hybridgen_topk_ratio: float = 0.05
   hybridgen_cpu_k_cap: int = 2048
   hybridgen_gpu_cache_factor: float = 1.0
   hybridgen_feedback_interval: int = 0
   hybridgen_gpu_q_proj: bool = True
   hybridgen_host_size: int = 0
   hybridgen_host_ratio: float = 2.0
   hybridgen_host_layout: str = "layer_first"
   hybridgen_io_backend: str = "kernel"
   ```
3. argparse 段加对应 9 个 `--hybridgen-*` flag

### 6.5 关键工程问题与解法

集成过程中踩到的几个非平凡问题：

#### 问题 1: 谁触发驱逐 / 何时触发

**问题**：HybridGen 原版在自己的生成循环里手控驱逐时机；SGLang 的循环不归我们管。

**解法**：在 `init_forward_metadata` 里做（每个 forward step 开头都会调）。此时：
- `seq_lens[i]` 已反映本 step 后的总长度
- 上一 step 的 KV 已写入 GPU pool
- 还没开始这一 step 的 layer 计算

非常理想的"两 step 之间"的窗口。

#### 问题 2: GPU 槽位生命周期

**问题**：sglang 的 KV 槽位由 scheduler / radix cache 全局管理，我们 backend 不能随便 free。

**第一版折中**：先做 shadow-copy，不 free GPU 槽——把要"驱逐"的 token KV 拷贝到 host，但 GPU 槽留着等 sglang 自己回收。这样不省 GPU 内存，但算法正确性、host 通路、top-k 路径能先跑通。

**当前修复**：decode 阶段已经做保守释放。约束是：
- `cache_protected_len == 0`，即这个请求没有被 RadixCache 保护 / 共享的 prefix；
- 已经备份到 host 的 GPU slot 释放回 `token_to_kv_pool_allocator`；
- `req_to_token` 对应位置写成 `0`，后续 cache cleanup 只 free 非 0 slot，避免 double-free；
- prefill/extend 阶段仍保留 GPU shadow，因为 `forward_extend` 目前走 `torch_native`，需要完整 GPU prefix。

这不是完整的 prefix-cache 集成，但已经解决了非共享请求 decode 阶段的 shadow-copy 常驻问题。

#### 问题 3: req_to_token 索引切片

**问题**：sglang 的 `req_to_token[req_pool_idx, :seq_len]` 包含已驱逐位置的旧 GPU 槽位 id。如果在 `_run_hybrid_decode` 里把这个全段 K/V 都拿来做 dense，会和 top-k 路径**重复计算同一 token**，破坏数学正确性。

**解法**：切片为 `[n_host:seq_len]`——n_host 是已驱逐的位置数（即 `_host_segments[req_pool_idx]` 的长度）。这样：
- GPU 段：`positions [n_host, seq_len)` → dense
- Host 段：`positions [0, n_host)` → top-k 选

互不重叠。

#### 问题 4: 请求池索引复用

**问题**：sglang 请求结束后会复用 `req_pool_idx`。如果不清理 `_host_segments[old_idx]` 和 `_prompt_lens[old_idx]`，新请求会"继承"已经被释放的 host 槽位 id——指向脏数据。

**解法**：`_reset_finished_requests` 在 extend 模式下检测 `extend_seq_lens[i] == seq_lens[i]`（无 cached prefix → 全新 prompt）作为复用信号，清状态 + free host slots。

#### 问题 5: HiCache JIT kernel 张量校验

**问题**：第一次实测时 `backup_from_device_all_layer` 报错：
```
Tensor match failed for Tensor<1>[strides=<1>, dtype=int64, device=cpu] at hicache.cuh:329
```

JIT kernel 对 tensor descriptor 检查严格。

**解法**：照抄 `srt/managers/cache_controller.py:717 move_indices`：
- `io_backend == "kernel"` 时，host_indices 必须**搬到 GPU**（虽然语义是"host 索引"，但 kernel 里需要 GPU tensor 做地址计算）
- dtype **不要强转 int64**，跟 `req_to_token` 的 native dtype 走（int32）

修后实测通过——服务器 log 里出现 feedback 调整的 `topk_ratio 0.0500 → 0.0600 → 0.0720 → 0.0864`，证明驱逐 + 反馈链路全部走通。

#### 问题 6: CUDA graph

**问题**：sglang 启动会调 `attn_backend.init_cuda_graph_state()`。我们没实现，base class 抛 `NotImplementedError`。

**解法**：启动加 `--disable-cuda-graph`。CUDA graph 支持留作 follow-up——hybridgen 原版没有，本来也不是 hybridgen 的能力。

#### 问题 7: `gpu_cache_factor` 语义错位

**问题**：sweep benchmark（4k–28k context, factor=2.0 默认）里 hybrid 和 torch_native 完全重合，0 次 feedback 事件——证明 eviction 全程没触发。看代码：

```python
gpu_cap = max(int(self.gpu_cache_factor * prompt_len), prompt_len)  # ← 多了 prompt_len floor
```

`max(int(2.0 * 8000), 8000) = 16000`，但实际 seq_len 最多 8032 → `excess = -7968` → 永远不 evict。`prompt_len` 这个 floor 让 factor < 1 都失效，等于「装得下整个 prompt 才考虑 evict」。

**对照原作** `hybridgen/llama/hybridgen/pytorch_backend.py:280`:
```python
gpu_cache_max = max(1, min(int(prompt_len * policy.gpu_cache_factor), max_len))
```

只有 `max(1)` 不让它降为 0，没有 prompt_len floor。default 也是 1.0 不是 2.0。

**解法（与原作 1:1）**：
```python
gpu_cap = max(int(self.gpu_cache_factor * prompt_len), 1)
```
+ default 2.0 → 1.0 + help 文本对齐原作语义。

**额外**：原版 `_prompt_lens` 只在 first sighting 时 set，chunked prefill 第一片可能只有 4 token → `prompt_len=4` 卡住，让所有后续 step 都触发 eviction。改成 `is_extend()` 期间持续取 `max(existing, seq_len)`。

修后 factor=0.1 实测：prompt 4k → cpu_len=3601, gpu_len=399（精确 10% on GPU）；feedback 按设计触发 topk_ratio 0.05 → 0.072 → 0.0461 → ... → 0.01（floor），cpu_k_cap 1803 → 901 → ... → 32。

#### 问题 8: Feedback 跨请求状态污染

**问题**：sweep 后续观察到 8k/16k/24k 三个长 prompt 只 decode 出 12 token 就早停。原因：feedback 控制器的 `topk_ratio`/`cpu_k_cap` 是 **backend 全局状态**，不会跨请求复位。第一个 4k bench 把 feedback 砍到 `topk=0.01, cap=32`，后续请求继承这个退化状态——attention 几乎只看 32 个 host token，temperature=0 下决定性挑到 EOS / 重复 token，触发 sglang 内置 stop。

原 hybridgen 是单请求 batch，每次 `gen_loop` 都重新初始化，不存在这个问题；移到 sglang 的多请求 serving 必须显式复位。

**解法（1 行）**：`__init__` 里存 `_initial_topk_ratio` / `_initial_cpu_k_cap`，`_reset_finished_requests` 检测到任意一个 fresh prefill 就复位回这两个初值。

```python
if any_fresh:
    self.topk_ratio = self._initial_topk_ratio
    self.cpu_k_cap = self._initial_cpu_k_cap
```

#### 问题 9: Per-decode-step Python overhead

**问题**：factor=0.1 强制 evict 后，prompt=4k decode 单 token 要 1.5 s，36 层 × 32 token output = 49 s（torch_native 同 prompt 是 0.78 s）。瓶颈不是 FLOP——profile 出来全在 Python / 同步 / 内存分配：

| 每层每 token 干的事 | 成本 |
|---|---|
| `host_indices = list(...)` + `torch.as_tensor(list, dtype=long)` | 3601 项 list→tensor 复制 |
| `q_cpu = per_req_q.to("cpu", torch.float32)` | GPU→CPU 强制同步 + cast |
| `k_host.to(torch.float32)` + `repeat_interleave(8, dim=1)` | 30 MB 临时 buffer 分配（GQA 展开） |
| `for h in range(nq): scores_remapped[h, cols] = ...` | nq=16 次 Python 循环、CPU index 写 |
| 36 层 × 上面 = 36 × ~40 ms = 1.5 s | per token |

**解法**（第一轮 4 处改动）：

1. **`_host_segments` 改 tensor 存储**——append 走 `torch.cat`，read `tensor[-cap:]`，省掉每层 `list→tensor` 的 O(n) 复制。

2. **跳过 `repeat_interleave`，改 grouped bmm**（hybridgen_topk.py）：
   ```python
   # 原: k.repeat_interleave(8, dim=1)  → (S, 16, d)  // 30 MB 分配
   # 新: q.view(nkv=2, g=8, d) × k.permute(1,2,0) → (nkv, g, S) → reshape (16, S)
   ```
   省掉 K 段 expansion 的 30 MB／层／token。

3. **删掉 `.float()` 上转**——CPU bmm 直接走 model dtype（bf16），bf16 在 oneDNN 下也能跑 BLAS。下游 softmax/cat 也是 bf16，链路一致。

4. **删 `for h in range(nq)` Python 循环**，换成单次 `scatter_`：
   ```python
   unique_local, inverse_flat = torch.unique(idx.flatten(), return_inverse=True)
   inverse_per_head = inverse_flat.view(nq, -1)
   scores_remapped.scatter_(1, inverse_per_head, topk_scores)
   ```

修后 sanity：`merged_softmax_attention` 与 dense SDPA 数值一致 max diff **1.71e-07**（与重构前同一量级）。

**第二轮优化（2026-05-08）**：

5. **去掉 decode 热路径里的 `torch.unique + scatter_`**。旧实现把所有 head 的 top-k token 合并成 shared unique 列，再给每个 head scatter 出 `-inf` mask；新实现直接按 Q head gather 自己的 top-k V，形成 `(num_q_heads, k_eff, head_dim)`，用 `merged_softmax_attention_per_head_v` 合并 softmax。这样少了 `torch.unique`、inverse index、`torch.full(-inf)`、`scatter_` 和 shared-column score remap。

6. **复用 CPU top-k workspace**。`CPUTopKWorkspace` 复用 `q_grouped`、`scores_grouped`、`topk_scores`、`topk_indices`；backend 还缓存 `q_head -> kv_head` 和 top-k 展开后的 kv-head index，避免每层重复分配固定形状 scratch。

> 这里只优化了 *Python overhead*；真正想跑 hybrid path 与 GPU triton attention 同档次，还是需要 §9.3 的 fused Triton kernel。

---

## 7. 怎么用

### 7.1 环境

依赖：
- 一个有 GPU 的节点（我们用的 A100-40GB，driver 12.4+，配 cu12 系 torch 2.9.1）
- Python 3.10+
- Rust toolchain（sglang 自己有 grpc Rust ext）
- protoc 编译器（Rust grpc ext 需要）

#### 一次性环境搭建

```bash
# 1. 装 Rust（user 级，无 sudo）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile minimal
. "$HOME/.cargo/env"

# 2. 装 protoc 27.x prebuilt 到 ~/.local/bin
mkdir -p ~/.local/bin
curl -LsSf https://github.com/protocolbuffers/protobuf/releases/download/v27.3/protoc-27.3-linux-x86_64.zip \
    -o /tmp/protoc.zip
unzip -o /tmp/protoc.zip -d /tmp/protoc_extracted
cp /tmp/protoc_extracted/bin/protoc ~/.local/bin/protoc
chmod +x ~/.local/bin/protoc

# 3. 建 venv 并装 sglang（必须用 v0.5.10.post1，cu12 时代最后一个 tag）
cd /home/binkma/bm_ds/hybridgen-sglang/sglang
git checkout v0.5.10.post1 -B hybridgen-integration  # 如果不在这个分支
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e "python[all]"
```

(如果驱动是 12.4 但 install 拉到了 cu13 包，回去检查 `git log -1` 看 commit；用错的 sglang tag 会拖来 cu13 nvidia 包链。)

### 7.2 启动 baseline（验证基线）

```bash
cd /home/binkma/bm_ds/hybridgen-sglang/sglang
. "$HOME/.cargo/env"
source .venv/bin/activate
export NO_PROXY="127.0.0.1,$no_proxy"  # 集群上有 HTTP_PROXY 时必须，否则 warmup 会报 503

python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B \
  --host 127.0.0.1 --port 30000 \
  --attention-backend triton \
  --mem-fraction-static 0.6 \
  --max-running-requests 4 \
  --max-total-tokens 8192
```

### 7.3 启动 hybrid_kvcache backend

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B \
  --host 127.0.0.1 --port 30000 \
  --attention-backend hybrid_kvcache \
  --disable-cuda-graph \
  --hybridgen-topk-ratio 0.05 \
  --hybridgen-gpu-cache-factor 1.5 \
  --hybridgen-feedback-interval 4 \
  --hybridgen-host-ratio 2.0 \
  --hybridgen-host-layout layer_first \
  --mem-fraction-static 0.6 \
  --max-running-requests 4 \
  --max-total-tokens 8192
```

注意：必须加 `--disable-cuda-graph`（CUDA graph 支持是 follow-up）。

启动期间应该看到日志：

```
HybridKVCacheAttnBackend: host pool ready (size=16385 slots, layout=layer_first, ratio=2.00).
```

—— 这是我们 backend 的 `_ensure_host_pool` 输出。

### 7.4 发请求验证

```bash
# 短 prompt（不会触发驱逐，走 fallback 路径）
curl --max-time 60 -s http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"The capital of France is",
       "sampling_params":{"max_new_tokens":16,"temperature":0}}'

# 长 prompt（触发驱逐 + top-k）
LONG_PROMPT=$(python3 -c "print('In recent decades, advances in machine learning have ' * 80)")
curl --max-time 120 -s http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$LONG_PROMPT The conclusion is that\",
       \"sampling_params\":{\"max_new_tokens\":48,\"temperature\":0}}"
```

### 7.5 Hybrid 路径正在工作的证据

server log 里会出现：

```
HybridKVCacheAttnBackend feedback: topk_ratio 0.0500 -> 0.0600, cpu_k_cap 0 -> 0 (cpu_len=4, gpu_len=9).
HybridKVCacheAttnBackend feedback: topk_ratio 0.0600 -> 0.0720, cpu_k_cap 0 -> 0 (cpu_len=8, gpu_len=7).
HybridKVCacheAttnBackend feedback: topk_ratio 0.0720 -> 0.0864, cpu_k_cap 0 -> 0 (cpu_len=12, gpu_len=7).
```

逐条解读：
- `cpu_len=4 → 8 → 12`：host 段在累积（驱逐生效）
- `gpu_len=9 → 7 → 7`：GPU 段被 cap 在 ~7（`gpu_cache_factor × prompt_len` 起作用）
- `topk_ratio 0.0500 → 0.0600 → ...`：feedback 控制器认为 host 工作可以被 GPU 工作隐藏，每次 ×1.2 提精度

---

## 8. 验证证据

### 8.1 单元测试（CPU 即可跑）

#### 测试 1: `hybridgen_topk.py` (2 个测试)

| 测试 | 验证目标 | 结果 |
|---|---|---|
| `test_workspace_matches_plain_topk_and_reuses_buffers` | workspace top-k 与普通 top-k 一致，并复用输出 buffer | ✅ |
| `test_per_head_v_matches_unique_scatter_representation` | per-head V merged attention 与旧 `unique + scatter` 表示一致 | ✅ |

#### 测试 2: `hybridgen_feedback.py` (4 个测试)

| 测试 | 验证 | 结果 |
|---|---|---|
| `test_bottleneck_shrinks_ratio_and_cap_together` | host bottleneck 时 ratio 与 cap 一起收紧 | ✅ |
| `test_bottleneck_initializes_cap_when_unlimited` | `cpu_k_cap=0` 时 bottleneck 会初始化 cap | ✅ |
| `test_hidden_host_relaxes_cap_gradually` | host hidden 时 cap 逐步放宽，避免大幅反弹 | ✅ |
| `test_controller_uses_effective_cpu_len_for_latency` | latency estimator 使用 cap 后的 effective CPU len | ✅ |

### 8.2 端到端 server 实测

环境：A100-40GB / sglang v0.5.10.post1 / Qwen2.5-1.5B / cu12 torch 2.9.1

| 测试项 | 结果 |
|---|---|
| `--attention-backend hybrid_kvcache` 启动成功 | ✅ |
| 短 prompt 输出与 baseline 一致 | ✅ "Paris..." |
| 725-token 长 prompt 出 47 个 token | ✅ |
| `host pool ready (size=16385)` 日志 | ✅ |
| `_maybe_evict_per_request` 实际触发 | ✅ cpu_len 增长可见 |
| `backup_from_device_all_layer` 不报错 | ✅ tensor format 修补后 |
| `_maybe_step_feedback` 实际触发 | ✅ topk_ratio 实时上调 |
| 9 个 `--hybridgen-*` CLI 选项可见 | ✅ `--help` 输出 |

### 8.3 修复后 2k / 4k / 8k benchmark

环境：
- 日期：2026-05-08
- GPU：A100-SXM4-40GB
- 模型：`Qwen2.5-Coder-3B` 本地 snapshot
  `/grand/hp-ptycho/binkma/HFmodel/hub/models--Qwen--Qwen2.5-Coder-3B/snapshots/09d9bc5d376b0cfa0100a0694ea7de7232525803`
- 请求：`input_ids` 直传，`max_new_tokens=32`，`temperature=0`，`ignore_eos=true`
- 公共启动参数：`--disable-radix-cache --disable-cuda-graph --disable-piecewise-cuda-graph --max-running-requests 1 --max-total-tokens 12288 --mem-fraction-static 0.70`
- Hybrid 参数：`--hybridgen-gpu-cache-factor 0.1 --hybridgen-topk-ratio 0.05 --hybridgen-cpu-k-cap 2048 --hybridgen-feedback-interval 4 --hybridgen-host-ratio 2.0 --hybridgen-host-layout layer_first`

| Prompt tokens | torch_native | hybrid_kvcache<br>(cap 修复后) | hybrid_kvcache<br>(unique/workspace 后) | hybrid_kvcache<br>(Triton fused 后) | hybrid_kvcache<br>(guarded overlap 后) | hybrid:native | 输出 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2,048 | 814 ms | 2,670 ms | 1,865 ms | 1,723 ms | 1,739 ms | 2.1× | 32 / 32 token ✅ |
| 4,096 | 848 ms | 3,199 ms | 1,952 ms | 1,809 ms | 1,843 ms | 2.2× | 32 / 32 token ✅ |
| 8,192 | 1,027 ms | 3,423 ms | 2,128 ms | 1,992 ms | 2,016 ms | 2.0× | 32 / 32 token ✅ |

Hybrid 日志证据：
- 2k：`cpu_len=1847, gpu_len=204`，符合 `factor=0.1` 后约 90% host / 10% GPU。
- 4k：`cpu_len=3690, gpu_len=409`，同样符合 90% host / 10% GPU。
- 8k：`cpu_len=7376, gpu_len=819`，同样符合 90% host / 10% GPU。
- `cpu_k_cap` 有效限制 CPU 扫描窗口：8k 场景从 `effective_cpu_len=2048` 收紧到 `1024 -> 512 -> 256 -> 128`，之后按反馈逐步放宽到 `153 -> 183 -> 219 -> 262`。
- cleanup 路径正常：`completion_tokens=32`，没有早停、double-free 或 cache cleanup 崩溃。
- guarded overlap 路径正常：top-k scores / V 的 host→device copy 现在可在独立 CUDA stream 上执行；GPU dense partial attention 也有两阶段 partial+merge Triton path，可与 CPU top-k 并行。默认参数下 2k/4k/8k 的 top-k 传输和 CPU cap 都偏小，guard 会保留原 fused path，避免小 case 因多 kernel launch / event / partial 写回而变慢。

结论：
- shadow-copy 释放修复后的功能正确性通过：强制 eviction 下 2k/4k/8k 都能完整 decode 32 token。
- `cpu_k_cap` 默认 2048 + feedback 收紧后，8k hybrid 从上一轮 `20.9s` 降到 `3.42s`。
- 去掉 `torch.unique + scatter_` 并复用 top-k workspace 后，8k 进一步从 `3.42s` 降到 `2.13s`；2k/4k 也分别降到 `1.87s` / `1.95s`。
- Triton fused merged-softmax 后，8k 进一步降到 `1.99s`；2k/4k 分别为 `1.72s` / `1.81s`。剩余 gap 主要来自 CPU top-k eager 路径、每层 CPU/GPU 同步和 host→device top-k V 传输。
- guarded overlap 后，latest 2k/4k/8k 为 `1.74s` / `1.84s` / `2.02s`。实测未加 guard 的 CPU/GPU partial overlap 为 `1.91s` / `2.02s` / `2.22s`，说明默认 `cpu_k_cap=2048` 下拆 kernel 不划算；最终实现只在 `effective_cpu_len >= 4096` 且 GPU dense segment `>= 1024` 时启用两阶段 CPU/GPU overlap。

---

## 9. 没做的事 / Followup

按重要性排序：

### 9.0 Decode profiling

为长生成场景加入了默认关闭的 profiling 开关：

```bash
SGLANG_HYBRIDGEN_PROFILE=1 \
SGLANG_HYBRIDGEN_PROFILE_INTERVAL=64 \
python -m sglang.launch_server ...
```

日志每 N 个 decode step 汇总一次 per-layer 平均时间：
- `decode_total/layer`
- `q_to_cpu/layer`
- `cpu_topk/layer`
- `host_v_gather/layer`
- `h2d/layer`
- `gpu_attn_or_merge/layer`
- `gpu_partial/layer`
- `evict_total/batch`, `evict_backup/batch`, `evict_release/batch`
- 平均 `cpu_len` / `effective_cpu_len` / `gpu_len` / `topk`

注意：profile 模式会在 GPU copy/kernel 周围插入同步，只用于时间归因，不用于最终吞吐 benchmark。对于真实 2k generated tokens，应该把 interval 设大一些（例如 64 或 128），观察 steady-state decode 均值；prefill/offload 是一次性成本，会被长 decode 摊薄。

### 9.1 实际释放 GPU 内存（论文卖点之一）

当前已做第一版保守释放：decode 阶段、且请求没有 RadixCache-protected 共享前缀时，被驱逐 token 的 KV 在备份到 host 后会立刻归还 GPU KV allocator，并在 `req_to_token` 中标记为已释放，结束清理时跳过这些 slot，避免 double-free。

仍未覆盖的场景：
- prefill/extend 阶段仍是 shadow-copy，因为 `forward_extend` 目前走 `torch_native`，需要完整 GPU prefix；非共享请求进入 decode 后，host-backed prompt shadow 会被释放。
- RadixCache-protected prefix 仍是 shadow-copy，因为这些 slot 由缓存树持有，可能被其他请求共享；直接释放会破坏 prefix cache。
- 要完整覆盖共享前缀，需要把 host-resident KV 作为 prefix cache 的一等状态，或让 RadixCache/HiCache 接管 HybridGen 的 offload 元数据。

### 9.2 CUDA graph 支持

当前必须 `--disable-cuda-graph`。要支持需要实现：
- `init_cuda_graph_state(max_bs, max_num_tokens)`
- `init_forward_metadata_capture_cuda_graph(...)`
- `init_forward_metadata_replay_cuda_graph(...)`
- `get_cuda_graph_seq_len_fill_value()`

挑战：hybrid 路径里有 host↔device 同步操作（top-k 在 CPU 算然后传 GPU），可能与 CUDA graph 的"无 stream sync"假设冲突。需要仔细设计——可能要把 host 工作放到独立 stream，让 graph capture 只覆盖 device 路径。

### 9.3 Triton 化 merged_softmax_attention

当前是 PyTorch eager 实现。理论上可融合成单个 Triton kernel：

```
fused_merged_attention_kernel(
    q, k_gpu, v_gpu, cpu_topk_scores, v_topk,
    out
):
    # 单 kernel 完成 QK^T(GPU) + concat + softmax + matmul
```

预期收益：减少 GPU↔HBM 流量，对长 context decode 性能有显著提升。

### 9.4 Long-context 精度回归

当前只验证了"能正常生成 token"，没做：
- 与 baseline 的 logit/perplexity 对比
- lm-eval-harness 下游任务（hellaswag / arc-easy / longbench）
- 不同 `topk_ratio` 下的精度退化曲线

这些是论文实验级验证。集成层不阻塞——但 production 部署前应该跑。

### 9.5 prefill 路径优化

当前 `forward_extend` 走 fallback（dense torch_native）。理论上 prefill 也可以借鉴 hybridgen 思路把超长 prompt 部分摊到 host。但原 hybridgen 也没做，先不做。

### 9.6 多请求共享 host pool

当前每个请求独立维护 `_host_segments`。批处理多请求时如果它们共享 prompt prefix，host pool 会重复存 KV。可与 HiCache 的 RadixTree 集成做共享——但这就回到了"为什么 HiCache 不直接合身"的问题，需要重新设计。

---

## 10. 文件清单

### 10.1 我们新建 / 修改的文件（基于 sglang clone 根 `/home/binkma/bm_ds/hybridgen-sglang/sglang/`）

| 文件 | 类型 | 行数 | 描述 |
|---|---|---|---|
| `python/sglang/srt/layers/attention/hybrid_kvcache_backend.py` | 新建 | 747 | HybridKVCacheAttnBackend 类 |
| `python/sglang/srt/layers/attention/hybridgen_topk.py` | 新建 | 268 | CPU top-k workspace + merged softmax 函数 |
| `python/sglang/srt/layers/attention/hybridgen_feedback.py` | 新建 | 267 | Feedback latency estimator + 调度策略 |
| `python/sglang/srt/mem_cache/hybridgen_release.py` | 新建 | 52 | HybridGen released-slot sentinel / filter / mark helper |
| `test/registered/unit/layers/test_hybridgen_topk.py` | 新建 | 102 | workspace 复用与 per-head V merged attention 单测 |
| `test/registered/unit/mem_cache/test_hybridgen_release.py` | 新建 | 128 | shadow release helper 与幂等释放单测 |
| `python/sglang/srt/layers/attention/attention_registry.py` | 修改 | +9 | 注册 `hybrid_kvcache` 工厂 |
| `python/sglang/srt/server_args.py` | 修改 | +68 | 加 9 个 hybridgen_* CLI 字段 + argparse |
| `python/sglang/srt/managers/schedule_batch.py` | 修改 | +2 | 向 worker batch 传递 `cache_protected_len` |
| `python/sglang/srt/model_executor/forward_batch_info.py` | 修改 | +2 | `ForwardBatch` 携带 `cache_protected_lens` |
| `python/sglang/srt/mem_cache/chunk_cache.py` | 修改 | +11 | cleanup 过滤 released sentinel，避免 double-free |
| `python/sglang/srt/mem_cache/radix_cache.py` | 修改 | +22 | released sentinel 下跳过 device radix 插入，只 free live slots |
| `python/sglang/srt/mem_cache/radix_cache_cpp.py` | 修改 | +20 | C++ radix cache 同步 released sentinel 语义 |
| **共计** | | **约 +1320 行** | |

### 10.2 完全没动的 sglang 文件

| 类别 | 文件 |
|---|---|
| 模型代码 | `python/sglang/srt/models/llama.py`, `qwen.py`, `opt.py` 等 |
| Attention 抽象 | `python/sglang/srt/layers/radix_attention.py`, `base_attn_backend.py` |
| KV pool | `python/sglang/srt/mem_cache/memory_pool.py`, `memory_pool_host.py` |
| HiCache / HiSparse | HiCache / HiSparse 上层逻辑未接入 HybridGen |
| Scheduler | `python/sglang/srt/managers/scheduler.py` |
| Model runner | `python/sglang/srt/model_executor/model_runner.py` |
| HTTP 层 | `python/sglang/srt/entrypoints/`, `launch_server.py` 等 |

### 10.3 集成的复用关系图

```
┌─────────────────────────────────────────────────────────────┐
│                我们写的（1087 行）                            │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │   hybrid_kvcache_backend.py          │                  │
│  │   HybridKVCacheAttnBackend           │                  │
│  └────────────┬─────────────────────────┘                  │
│               │  uses                                       │
│               ↓                                             │
│  ┌──────────────────┐  ┌──────────────────────┐           │
│  │ hybridgen_topk.py│  │ hybridgen_feedback.py │           │
│  └──────────────────┘  └──────────────────────┘           │
│                                                             │
│  + attention_registry.py  +9 行                            │
│  + server_args.py        +68 行                            │
└─────────────────────────────────────────────────────────────┘
                       │
                       │ 调用 / 继承
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                复用 sglang 已有                              │
│                                                             │
│  AttentionBackend (base class)                             │
│  TorchNativeAttnBackend (作为 fallback)                    │
│  MHATokenToKVPool (GPU pool)                               │
│  MHATokenToKVPoolHost (host pool, 我们直接实例化)          │
│   ├── alloc / free                                          │
│   ├── backup_from_device_all_layer  ← JIT CUDA kernel      │
│   └── load_to_device_per_layer                              │
│  ForwardBatch (字段 token_to_kv_pool / req_pool_indices /  │
│                seq_lens / extend_seq_lens / out_cache_loc) │
│  RadixAttention (派发到我们的 backend，零改动)             │
│  Scheduler / ModelRunner / HTTP 层 (全部不动)              │
└─────────────────────────────────────────────────────────────┘
                       │
                       │ 灵感参考（不直接调）
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                参考但不直接使用                              │
│                                                             │
│  DoubleSparseAttnBackend (257 行)                          │
│   ← 形态参考: top-k attention backend 怎么写              │
│  HiCacheController.move_indices                            │
│   ← 张量准备参考: io_backend="kernel" 时索引如何 prep     │
│  HiSparseCoordinator                                        │
│   ← 不用，但理解了"hybridgen ≠ sparse attention"          │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. 致谢与参考

### 论文 / 项目

- **FlexGen**: Sheng et al., "High-Throughput Generative Inference of Large Language Models with a Single GPU", ICML 2023
- **H₂O**: Zhang et al., "H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", NeurIPS 2023
- **HybridGen**: 原仓库 `/home/binkma/bm_ds/hybridgen`
- **SGLang**: https://github.com/sgl-project/sglang
  - 本次基于 v0.5.10.post1, commit `7c35342c1` (cu12 时代最后一个 tag)
- **HiCache** (SGLang): Zhiqiang Xie 等, PR #2693 (2025-02)
- **HiSparse** (SGLang): 同作者, PR #20343 (2026-03)
- **DoubleSparsity** (SGLang): top-k attention 的现有 backend 模板

### 关键启发

| 概念 | 来源 |
|---|---|
| Offloading + 三层存储 | FlexGen |
| Heavy-hitter / top-k attention | H₂O |
| Top-k → merged softmax 不引入近似 | HybridGen 原创合并 |
| 反馈延迟调度 | HybridGen 原创 |
| Backend 注册模式 | SGLang AttentionBackend 接口 |
| Host KV 内存池管理 | SGLang MHATokenToKVPoolHost |
| HiCache 张量准备约定 | SGLang `cache_controller.py:move_indices` |

---

## 附录 A: 全部 CLI 选项

```
--attention-backend hybrid_kvcache    选这个 backend
--hybridgen-topk-ratio FLOAT          top-k / cpu_cache_len 比例 (默认 0.05)
--hybridgen-cpu-k-cap INT             CPU K 扫描窗口上限 (默认 2048；0=初始不限，feedback 可收紧)
--hybridgen-gpu-cache-factor FLOAT    GPU 段大小 / prompt_len (默认 1.0)
--hybridgen-feedback-interval INT     反馈步数 (0=关闭) (默认 0)
--hybridgen-gpu-q-proj                Q 在 GPU 投影后再 copy (默认 True)
--hybridgen-host-size INT             host pool 大小 GB (0=用 ratio)
--hybridgen-host-ratio FLOAT          host pool / GPU pool 倍数 (默认 2.0)
--hybridgen-host-layout STR           host 内存布局: layer_first|page_first|...
--hybridgen-io-backend STR            host↔device 传输: kernel|direct (默认 kernel)
--disable-cuda-graph                  必加 (CUDA graph 是 follow-up)
```

---

## 附录 B: 故障排除

| 现象 | 原因 | 修复 |
|---|---|---|
| 启动报 `NotImplementedError` from `init_cuda_graph_state` | hybrid_kvcache 没实现 CUDA graph | 加 `--disable-cuda-graph` |
| warmup 期 503 (Squid) | 集群 HTTP_PROXY 把 127.0.0.1 也走代理 | `export NO_PROXY="127.0.0.1,$no_proxy"` |
| `Tensor match failed` from `hicache.cuh` | host_indices 在 CPU、dtype 强转 int64 | 确认走的是修补后的 `_maybe_evict_per_request`：host_slots 搬 GPU、保持 native dtype |
| `cuptiActivityEnableDriverApi` undefined | torch wheel 比 driver 新（cu130 vs cu12.4） | 必须用 sglang v0.5.10.post1（cu12 tag）；不要用 main / v0.5.11 |
| `can't find Rust compiler` 装包失败 | sglang 的 grpc Rust ext 需要 rustc | 装 rustup |
| `Could not find protoc` | grpc-build 需要 protoc | 装 protobuf 27.x |
| `enable_double_sparsity` 不存在 (老 sglang) / 存在 (新 sglang) | 代码版本对不上 | 锁 v0.5.10.post1 |

---

**文档版本**: 1.0
**生成时间**: 2026-05-06
**对应 SGLang commit**: `7c35342c1`（v0.5.10.post1 tag, cu12 时代）
**作者集成代码**: 1087 新增行 / 5 个文件
