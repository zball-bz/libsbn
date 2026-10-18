# 浮点库开发章程

本章程约束 libsbn 浮点层的重写，范围包括新代码 `src/fp/`、它在 L0 中新增的内核与执行器、生成器、测试、基准和实验。astra 的旧实现在 P0 归档后删除。章程只服务两个目标：

1. **库的性能**：每个函数在每个精度段都达到 [算法架构与性能目标](float-algorithms.md) 中的目标。
2. **开发和调优的效率**：从改一行代码到看到可比较的数字，循环要短；不能直接换来性能或正确性的流程，一律删掉。

其他考虑（接口完备、文档完整、流程留痕）都排在这两个目标之后。

**与 AGENTS.md 的关系**：浮点工作中，本章程覆盖 `libsbn_v3/AGENTS.md` 和 `experiments/AGENTS.md` 的以下规则（需用户确认，见 §15）：

| 被覆盖的规则 | 改为 |
|---|---|
| 保留原始重复样本、标识和全部原始结果 | 只留被决策引用的 CSV，负结果记成 `LOG.md` 的一行（§7） |
| 每次性能测量前做内存查询和准入 | 只在 ≥2^24 bit 或多线程时做（§7） |
| fresh 调用计时，以及 `experiments/docs/performance-acceptance-2026-09-18.md` 的验收协议 | warm 主指标加冷启动锚点（§7） |

基准锁、温度门槛、`src/` 禁止 malloc/new、ISA intrinsics（内建函数，如 `_mm512_madd52lo_epu64`）只出现在 native 组等其他规则继续有效。

---

## 1. 总目标与硬约束

| 项 | 规定 |
|---|---|
| 主指标 | warm、单线程、独立调用（吞吐模式）的单次调用耗时，覆盖 32 bit 到 2^24 bit |
| 次指标 | 批量吞吐（纵向 SIMD 接口）、多线程（只在高精度段）、冷启动 |
| 硬约束 | 球层结果必须包含真值，不得为性能牺牲；交付精度不低于 64n − c_f（c_f = 1 + ⌈log2 err⌉，由 `//@` 规格的误差界导出，且 S 段 c_f ≤3、M–X 段 c_f ≤4；fbench 按这个上限判 acc_bits）；半径紧致度是性能指标的一部分 |
| 非目标 | 正确舍入。它推迟到 L3，L1/L2 不得为它预付任何成本（舍入模式、ternary 值、NaN 细节、守护 limb、紧变体内核等） |
| 比较对象 | 该点最快的 warm 竞品，不区分点与球：MPFR 4.2.2，FLINT 3.7.0-dev（arb/arf/nfloat/mp_real）。例外：基本运算的 L2 行以球竞品为参照，与点竞品的比较由同精度的 L1 行承担。派生函数以我方原语为单位。逐点的 warm 目标（含 X 段、派生函数与 K/E）和冷启动行只来自 `experiments/float/runs/2026-10-08-warm-baseline/targets.py`；批量、多线程、常数构建、谓词、表构建和 P7 特殊函数的目标写在算法文档 §2.2、§2.4、§2.5 |

---

## 2. 分层

| 层 | 名字 | 职责 | 规则 |
|---|---|---|---|
| L0 | 整数层（已有，并扩展） | limb 内核、u52/IFMA、pq16 FFT、p48 NTT、Newton、二分级数、radix | L1 直接调用内核和私有 Frame 接口，不走带查询/规划的公开 C 入口。浮点需要而整数层缺少的能力，一律补在 L0（清单见算法文档 §6.5），配整数测试和 π/e 锚点 A/B，不在浮点层里绕过。生成的定长 limb 内核和 bit-burst 执行器也属于 L0，整数层和常数生产者同样使用 |
| L1 | 原始层 `sbn::fp`（内部） | `fx` 定点内核：生成，误差界是 `constexpr` 常量。`raw` 浮点级函数：返回误差（输出网格上的 ulp 数），需要计划时从计划缓存取 | **没有通用契约**：每个函数用一行 `//@` 规格声明定义域、输出网格、误差、n 的范围。没有报告对象，没有舍入模式，不检查参数（debug 断言可以），不分配内存，调用路径上不规划 |
| L2 | 球层 `sbn::ball`（公开） | 中点 + 半径；特殊值；输入半径传播；按半径降精度；精度段调度 | 唯一对外承诺精度（包含性）的一层。每次调用只在入口做一次快路径判断；每个函数只调用一次 L1，不在内部串联 L2 运算 |
| L3 | 定精度正确舍入层 | 推迟 | 用 L1 的误差做 Ziv 测试；可以有自己的小尺寸内核；不得反过来要求 L1/L2 付出代价 |

---

## 3. 热路径硬规则

违反下列任何一条都视为性能缺陷，必须修正，或在算法文档相应条目里写明理由。

1. **校验只做一次**，在 L2 入口。内核里不做校验、重叠检查、报告构造，也不检查"预算"。
2. **调用路径上不做规划。** 选择算法只能靠对 limb 数的 `switch`，或查生成的小表（维度：limb 数、输入有效高度、|exp| 类别、线程数）。需要计划的方法从进程级计划缓存取。缓存的键是（方法, 尺寸类, 线程数, 约化模式, 输入类），尺寸类按 1/16 倍频程划分；缓存按 LRU 淘汰。计划元数据与计划持有的数据（D 树、频谱、1/ΠD）的构建合计 ≤ 同精度 warm 调用的 10%（p ≥ 2^18 bit）或 25%（2^16 ≤ p < 2^18 bit），元数据另有 ≤50 µs 的绝对上限；单独测量（§7，算法文档 §1.4）。
3. **常数和约化表不按调用重算。**
   - 常数：π、ln2、2/π 静态到 2^17 bit，其余常数静态到 2^16 bit。
   - 按位约化表静态到 4096 bit；更高精度由常数库按租雪橇规则惰性生产，按 ×2 增长。
   - 生产者不得调用通用的 exp/log。
4. **精度按 limb 取整，不自动加守护 limb。** 守护位逐个方法证明到最小值。禁止 `p+128`、`2p+256` 这种通用膨胀；嵌套调用时守护位不重复叠加。
5. **只要高半部分就用短积**（mulhigh）；求残差用中间积。
6. **小尺寸（n ≤ 8 limb）用生成的直线代码**，内联进算子所在的 TU。每个 limb 运算都不得经过函数指针、虚函数或跨 TU 调用。
7. **不分配内存**：小尺寸放栈上；大尺寸用调用者给的 scratch，或线程级只增不减的 scratch。
8. **不许有断崖。** 断崖指相邻网格点的耗时比同时超过 1.25 和"模型运算量比 × 1.15"。≤8 limb 时，n → n+1 的模型比取自生成器 `SUMMARY.txt` 的指令数；n > 8 时模型比取 `targets.py` 中这两个网格点的目标之比（X 段每倍频程 4 点时约 1.25–1.33）；同一 n 之内，相邻点之比 ≤1.1。参数域同理：例如 |x| > 1 时不能掉进慢 10 倍的通用路径。算法的切换点必须落在两条曲线的交点上。
9. **没有死代码。** 实现了的算法，要么在它最优的区间被调度到，要么删掉。每个算法变体都导出内部符号 `sbn3i_fp_<f>_<variant>`，可以直接调用做 A/B。
10. **生成的代码不手改。** 改生成器，提交生成物；`build.py` 用 `--check` 保证两者一致。

---

## 4. SIMD 与 ISA

- 算法代码写成以"内核提供者"为参数的模板（`src/fp/alg/`），不含 ISA intrinsics。现在实现三个提供者：
  - `portable`：`unsigned __int128` C++，用作测试参考，也是移植的起点；
  - `x86adx`：包装 L0 中生成的 MULX/ADX 定长内核；
  - `x86v4ifma`：在 x86adx 之上包装 L0 的 IFMA 内核（u52 链式驻留、线性遍、纵向 8 道），自己不放内核。
- **所有浮点实例化 TU 都放在 native 组**（native 的编译标志是 common 的超集）。一个函数族的 L1 和 L2 放在同一个 TU，不需要 LTO 也能完全内联。
- 调参参数按（提供者, 微架构）分文件：`config/float-tune/<provider>-<uarch>.json`，算法代码通过 `P::tune` 读取。
- 设计每个新内核时，都要写下四种做法的周期估算再选：标量 MULX/ADX、单数内横向 IFMA、跨数纵向 IFMA（批量）、FP64。估算依据是算法文档附录 B 的 Zen 5 指令表，或生成器 `--explain` 给出的 llvm-mca 预测（校准通过之后才用）。
- AVX2、AVX-IFMA、AArch64 NEON/SVE 只需保证架构上能加一个提供者，现在不做。将来如果需要运行时多 ISA 分派，只能在算子粒度做。

---

## 5. 代码组织

```
include/sbn3/ball.hpp                    L2 公开接口（C++；C 接口 ball.h 后加）
src/fp/core/                             类型、xd、调度、scratch
src/fp/provider/                         提供者概念；portable、x86adx、x86v4ifma；gen/（调参头文件）
src/fp/alg/                              ISA 无关的算法模板（arith div root exp log trig atan agm pow series reduce newton；burst 只放胶水）
src/fp/ball/                             半径运算、L2 运算、谓词与区间运算、慢路径（n=0、哨兵、重定中心）
src/fp/const/                            常数库
src/fp/runtime/                          计划缓存、线程级 scratch、limb 池
src/fp/gen/                              生成物（poly_*.inc、tables、consts、M1 整函数内核、SUMMARY.txt）
src/fp/cr/                               L3（暂空）
src/core/x86_64/gen/                     L0：生成的定长 MULX/ADX 内核（头文件内联汇编，不是独立 TU）
src/kernels/x86v4ifma/gen/               L0：生成的 IFMA 定长、u52、u52 MAC2 与纵向 8 道内核
src/backend/pq16/（或 src/product/）      L0：pq16 形状缓存（BurstTree 上层也要用）
src/series/burst*                        L0：BurstTree 执行器（浮点与常数生产者共用）
src/instances/x86v4ifma/fp_*.cpp         实例化 TU（native 组，-O3），每个函数族一个，编译 ≤8 s
tools/generate_float.py、tools/fpgen/    生成器（同时生成调参头文件）；tools/check_fp_codegen.py、check_fp_size.py
config/float-gen.json                    生成清单与代码体积预算
config/float-tune/<provider>-<uarch>.json  阈值与参数
tests/fp/                                库内测试
```

整数层的 TU 不得包含 `src/fp` 的任何头文件：改一次浮点头文件，不能引起整数层几十秒的重编译。L0 的生成内核头文件不依赖 `src/fp`；BurstTree 用到的内核与缓存（纵向 IFMA、u52 MAC2、形状缓存）因此都放在 L0。

---

## 6. 正确性

| 层面 | 内容 | 时长上限 |
|---|---|---|
| 库内测试（无 GMP） | 代数运算用现有整数 oracle 精确核对；每条 `//@` 规格做自洽检查：\|y_n − trunc_n(y_{n+2})\| ≤ (err_n + 1)·ulp_n + err_{n+2}·ulp_{n+2}，即高精度结果必须落在低精度区间内；球恒等式（1 ∈ exp(x)·exp(−x)，1 ∈ sin² + cos²）；portable 与 native 提供者交叉核对；特殊值；limb 边界 p = 64k−1、64k、64k+1 bit（k = 1…16，更高处每倍频程取 1 个 k） | 单个测试 ≤1 s；浮点默认测试合计 ≤30 s，纳入 `make check`；聚焦运行：`python3 tools/build.py check --only-test <name>`（P0 给 `--only-test` 加 fnmatch 通配之后才能写 `fp_*`） |
| 实验侧 `fcheck`（MPFR/Arb 作参考） | 随机输入和对抗输入（抵消、端点、巨大参数、极小参数、宽球）；检查包含性；报告每个内核的"实测误差 / 声明误差"：比值 >1 是硬失败；<1/8 只提示复查误差证明 | 不计时；里程碑时全量运行 |
| 生成物 | 生成器带 `--check`；表和常数用第二种公式在 Python 大整数下交叉验证（`--verify`），里程碑时运行 | — |
| 消毒器 | ASan/TSan | 里程碑时运行 |

守护位只能凭新的证明减少，不得凭实测减少。

---

## 7. 性能测量纪律

**工具**：只用一个工具 `fbench`（位于 `experiments/float/`）。

- 一个进程扫一条曲线，各实现在每个点上轮流测。
- 参考库的适配器各自成 TU，只编译一次；改 libsbn 时只重编一个薄适配器。
- fbench 读取 `targets.py` 的目标和竞品集合（`COMP`），逐点输出 t/target、交付精度和计划缓存未命中次数，§14 的检查因此是自动的。
- 实验仓库里的通用 perf 框架（campaign 预算、resume、attempt 目录、收据、冻结二进制）不用于浮点。

| 项 | 规定 |
|---|---|
| 网格 | ≤4096 bit：每倍频程 ≥32 点，并覆盖每个 limb 数；4096 bit–2^20 bit：每倍频程 ≥16 点；>2^20 bit：每倍频程 4 点，只在里程碑测 |
| 样本 | 每点 1 个样本：≥2 ms 的调用循环，输入至少 8 个轮换；单次调用超过 2 ms 时只测 1 次调用。切换实现时先预热；p > 4096 bit 时先调用 `sbn3i_fp_prewarm(kind, p)`（建 M 段按位表与 H 段原子库），另报一条"表/库未建"曲线。warm 曲线在确定性模式下固定原子库（主曲线：exp/log k_at = 13、三角 32 个高斯原子；冲刺曲线 k_at = 48），计时内不允许原子库扩建。与 MPFR 比较时，每点用 ≥3 个固定输入取中位数（MPFR 有 Ziv 重试尖峰） |
| 模式 | 吞吐模式是验收依据。≤8 limb 另测延迟模式（输出作为下一次输入），只作诊断。初等函数用精确输入；L2 基本运算用 rad ≠ 0 的输入，另测精确输入行（`*_l2e`） |
| 精度 | 每行记录交付精度 acc_bits（球：−log2(rad/\|mid\|)）。L2 验收点要求 acc_bits ≥ 64n − c_f，不满足就判失败 |
| 主指标（warm） | 进程级缓存（常数、约化表、FFT 表与形状、计划、原子库）已热，libsbn 和竞品同等对待。**不得在计时内清竞品缓存**。另在 H/X 锚点用 `sbn3i_fp_plan_clear()` 测"只清计划缓存"：p ≥ 2^18 bit ≤1.1× warm，2^16 ≤ p < 2^18 bit ≤1.25× warm（即 §3 第 2 条的预算） |
| 冷启动 | 每个锚点起一个新进程，计时前预先触碰 .text/.rodata，并做一次同尺寸的整数乘法热起堆和页表，然后测第一次调用（常数库的旧缓冲活到进程结束，进程内清不掉）；只在 p ≥ 4096 bit 的锚点、里程碑时测。目标见算法文档 §2.5 与 `targets.py` 的 `OP.cold` 行（竞品首次调用在 `flint_cleanup()` 之后测，`bench1.c coldfn`）。另测精度重试阶梯（p₀·1.5^k，每级一次调用） |
| 参数域 | 里程碑时，每个函数在 64、4096、2^18 bit 上扫 \|x\| = 2^k（k 每隔 4 取一点），以及零点、极点、π/4 倍数附近，适用同一条断崖规则 |
| 竞品范围 | 竞品集合取自 `targets.py` 的 `COMP`（MPFR 是最佳竞品的格子见算法文档附录 A）。初等函数 exp、log、sin_cos、atan 的 `COMP` 只含 arb 与 mp_real：基线中 MPFR 与 nfloat 在全网格都不比它们快，每月复测时核对 |
| 环境 | 编译和计时都在 `build/benchmark.lock` 下，且互不重叠；开测时 Tctl ≤62 °C，起止温度写进 CSV 头；固定在 CCD0 的同一个空闲核（默认 CPU 2），不用 CPU 0。内存准入只在 ≥2^24 bit 或多线程时做 |
| 平滑性 | 自动标出违反 §3 第 8 条的点，只重测这些点。里程碑时对 n > 8 且 ≤2^20 bit 的部分做 J/R/B 检查：x = log2(bits)，y = ln(warm ns)，Theil–Sen 1 倍频程窗（≥9 点），相邻超额 P95 ≤5%、最大 ≤10%，倍频程波纹 ≤15%。S 段与 2^20 bit 以上（每倍频程只有 4 点，凑不出 9 点窗口）只按 §3 第 8 条判定 |
| 竞品构建 | 固定版本、静态链接，装在 `.work/` 之外的稳定位置（`~/bigint/refs/`），MPFR 源码包放进 `reference/`。每月对 FLINT 上游 HEAD 复测一次锚点，更新 `targets.py` 和算法文档 |

**时间预算**：

- 测量时间不超过工作时间的 20%。先建模（指令数、依赖链、周期估算或 llvm-mca），再测量；测量只用来裁决模型无法裁决的问题。
- 从改代码到出数字 ≤20 s。
- 一个函数一条完整曲线（32 bit–2^20 bit，三个实现）≤1 min；2^20 bit 以上的锚点单独计时，≤3 min。

**CSV 格式**：头部一行注释 `# commit=… tree=<工作树 diff 哈希> fp=<src/fp 树哈希> cc=clang-21.1.8 flags=… cpu=2 refs=mpfr-4.2.2,flint-657a283,gmp-6.3.0-zen3 tctl=起/止 date=…`；列为 `op,impl,bits,ns,calls,acc_bits,mode,target_ns,plan_miss`。2026-10-08 的基线采用旧格式（`impl,op,bits,ns`，吞吐模式），只作参照。

**留存**：

| 留存 | 不留存 |
|---|---|
| 被决策引用的 CSV 及其测量源码 | 原始重复样本、二进制、`.a` 副本、源码快照、构建收据、attempt 目录、stdout/stderr、锁文件、`__pycache__` |
| 每个实验 ≤15 行的 NOTE；`LOG.md` 一行 | 探索性 CSV（写到 `/tmp` 或被忽略的 `float/scratch/`） |
| 调参文件与它引用的 run | 图（随时可由 `plot.py` 重新生成；只有文档嵌入的 SVG 才提交） |

被新结果回答过、且没有被调参文件引用的旧 run 直接删除。`experiments/float/` 总量上限 50 MB、500 个文件；超过时先清理再继续。

---

## 8. 调优流程

1. 每个改动按"假设 → 带数字的预测 → 测量 → 保留或回退 → `LOG.md` 一行"走完，负结果同样记一行。
2. 交叉点：`fbench --crossover A,B --bits lo:hi` 在每个点上交替测两个变体，对 log(tA/tB) 做 5 点中值平滑；平滑后的比值连续 ≥4 点落在同一侧，就判定在那里交叉。差异 <2% 时选代码、内存或计划更小的那个。结果写进 `config/float-tune/<provider>-<uarch>.json`，再生成头文件。变体都能直接调用，所以试阈值不需要重新编译。
3. 内核一旦变化，重扫受影响的阈值。
4. 优先级按"预期收益 × 覆盖的精度段宽度 ÷ 工作量"排序。fbench 输出的 t/target 就是待办清单：离目标最远的格子优先。
5. **改目标**：收紧目标可以直接改 `targets.py` 和算法文档。放宽目标必须附一个下界论证（指令数或依赖链），或者两种以上方案的实测结果，并且单独做一次提交。

`LOG.md` 的行格式：

```
2026-10-09 | exp S 段 n=4 Tang 短表 | 256 bit 41→33 ns，128 bit 无变化 | 保留 | runs/2026-10-09-exp-tang/exp.csv
```

---

## 9. 文档纪律

- `docs/` 下的浮点文档只有三份：本章程、[算法架构与性能目标](float-algorithms.md)，以及以后的 L2 接口文档。都就地更新；不新建带日期的设计文档，不写"审计 / 契约 / 完成度 / 交接"类文档。
- 算法文档和 `targets.py` 是唯一的设计记录与目标来源。
- 实验侧只有 `experiments/float/README.md`（不超过一页）、`LOG.md`，以及决策引用的 NOTE。
- 代码注释只写"为什么"和误差界，不复述代码。每个 L1 函数声明处的 `//@` 规格行是唯一的精度说明，测试和 fcheck 自动读取它，例如：

```cpp
//@ exp_core | dom: t in [0,2^-8) fixed(N) | out: fixed(N+1) | err: exp_core_err<N> | n: 1..8 | gen
```

c_f 不单独写：`specs.py` 从 err 导出 c_f = 1 + ⌈log2 err⌉。输出网格相对于结果量级的内核（log1p、expm1 在 0 附近）在 out 字段写明截断点，例如 `out: rel(N)`。

---

## 10. 版本管理与协作

- **共享工作树的现实**：`libsbn_v3` 是多个代理共用的单一工作树，在 `main` 上有大量未提交的改动，切分支会影响所有人。所以浮点工作用独立的 worktree：
  ```
  git worktree add ../libsbn_v3-float -b float-rewrite
  ```
  `build.py` 和 fbench 统一使用绝对路径 `/home/dev/bigint/libsbn_v3/build/benchmark.lock`（通过环境变量，需要对 `build.py` 做一个小改动）。experiments 通过 `SBN_EXPERIMENTS_ROOT=/home/dev/bigint/libsbn_v3/experiments` 接入（`tools/reference_paths.py` 已支持），不建符号链接：`.gitignore` 的 `/experiments/` 只忽略目录，符号链接会被暂存。
- **worktree 的起点**：HEAD（fb1d517）里没有浮点依赖的 L0（`normalized_newton_*`、pq16/NTT 的 matrix 程序、`finite_frame_*`、u52 middle 的改动等，都只在共享工作树的未提交改动里）。这些改动由其负责人提交之前，不建 float-rewrite。P0 的 astra 删除在共享工作树上执行（§15 第 5 项确认之后）并提交，float-rewrite 从该提交分出。
- **实验身份** = (HEAD, 工作树 diff 哈希, `src/fp` 树哈希)，写进每个 CSV 头。`float-rewrite` 分支上频繁提交，每个可度量的改进一次，提交信息附一行性能变化（算子 / 精度段 / 前后数值）。浮点所依赖的 L0 改动，先由正在编辑这些文件的代理提交，或者与其协调后再动。
- **归档 astra 代码**：astra 的代码全部未被跟踪。用单独的 index 把整个工作树（含未跟踪文件，遵守 `.gitignore`）快照成 tag，不动 HEAD、真实 index 和工作树：
  ```
  IDX=$(git rev-parse --absolute-git-dir)/archive.index      # 放在 .git 内，不会被 add -A 收进去
  GIT_INDEX_FILE=$IDX git read-tree HEAD
  GIT_INDEX_FILE=$IDX git add -A
  tree=$(GIT_INDEX_FILE=$IDX git write-tree)                 # write-tree 也必须用临时 index
  c=$(git commit-tree $tree -p HEAD -m 'archive: astra float before rewrite')
  git tag archive/pre-float-rewrite-2026-10-08 $c
  rm -f $IDX
  ```
  这个流程让 tag 包含全部未跟踪文件和工作树修改，不改真实 index，也不收被忽略的目录。核对（在仓库顶层执行，两条命令都与当前目录有关）：
  ```
  cd "$(git rev-parse --show-toplevel)"
  diff <(git diff --name-only --diff-filter=A HEAD archive/pre-float-rewrite-2026-10-08 | sort) \
       <(git ls-files --others --exclude-standard | sort)
  ```
  输出必须为空（逐个文件比较，覆盖 §12 删除清单中 docs/、tests/、examples/ 下的未跟踪文件），且 tag 中 `src/float` 有 54 个文件。不为空就重新打 tag；为空之前不得执行 P0 删除：这些文件未被跟踪，删了无法从 git 恢复。核对与删除之间不得有人新增文件。
- 不推送、不发布，除非用户明确要求。

---

## 11. 实验目录治理

新建的自包含区域，与旧框架隔离：

```
experiments/float/
  README.md     一页：构建参考库，运行 fbench/fcheck，留存规则
  refs.sh       从 reference/ 中的源码包构建 GMP/MPFR/FLINT 到 ~/bigint/refs/
  fbench/       计时工具（网格、计时、温度、绑核、CSV）；参考库适配器单独成 TU，只编译一次
  fcheck/       不计时的包含性与紧致度检查
  plot.py       CSV → SVG
  runs/YYYY-MM-DD-主题/   只放被决策引用的 CSV、测量源码和 NOTE
  LOG.md        一行一个实验
```

现有内容：

- `runs/2026-10-08-warm-baseline/`：warm 竞品曲线、乘法曲线、`targets.py`（全部目标的唯一来源）。
- `runs/2026-10-08-design-probes/`：设计阶段的探针、级数引擎探针和代价模型。
- `README.md`、`LOG.md`。

astra 的研究文档留在原处作参考。astra 浮点工作产生的结果与冻结二进制如何处理，以及实验仓库的全局清理，都列在 §15，需要用户确认。

---

## 12. 开发阶段与验收

| 阶段 | 交付 | 验收 |
|---|---|---|
| P0 基础设施 | 归档并删除 astra 浮点代码（清单见下）；目录骨架；`fbench`/`fcheck`；生成器框架与 `--check`；表示类型与提供者概念；运行时组件（计划缓存、线程级 scratch、limb 池；L0 的 pq16 形状缓存按算法文档 §6.5 与其负责人协调后加入）；扩展 `audit_build.py`，让 `fp_*.cpp` 也检查无分配器导入、IFMA 提供者 TU 确实含 `vpmadd52`；`build.py` 并行编译、`--wait-lock` 与 `--only-test` 的 fnmatch 通配（需协调）；`generate_float.py` 从 `config/float-gen.json` 生成 `sources.json`/`tests.json` 中的 `fp_*` 条目（带 `--check`），避免每个 TU 和测试手写 8 字段和约 35 行的登记 | 改代码到出数字 ≤20 s；fbench 用本章程协议重测参考库，在基线已有的点上，中位数比值落在 [0.9, 1.1]（MPFR 为 [0.85, 1.15]）；`make lib/check/examples/audit/bench` 全部通过 |
| P1 基本运算 | add/sub/mul/sqr/fma，全精度段，L1 + L2；谓词与区间运算；转换与十进制输出；L0 的定长内核、u52 mulhigh/平方、无 Frame 入口、点积 | `targets.py` 中对应各行在全网格达标；包含性与 acc_bits 全部通过 |
| P2 除法与开方 | div/inv/sqrt/rsqrt/rootn/cbrt；L0 的 `RSQRT_GENERAL`、`MUL_RSQRT_GENERAL`、小尺寸 Newton、预备除数、K 链 `divrem_1` | 同 P1 |
| P3 常数 | 静态表；常数库；L0 的 π/e 近似入口、Frame 形式生产者、闭式停止下标、决策重放、级数小节点快路径（连同其底层的 L0 纵向 IFMA 内核与 u52 MAC2；附 π/e 锚点 A/B） | 读常数 ≤3 ns（≤4096 bit）；1M bit 冷构建 π ≤5.0 ms、e ≤1.8 ms、ln2 ≤9.0 ms |
| P4 exp/log 族（S–M 段） | exp/expm1/exp2/exp10、log/log1p/log2/log10、双曲与反双曲 | 同 P1，另加冷启动锚点 |
| P5 三角族（S–M 段） | sin/cos/tan/sincos 及其 π 倍角版本、atan/asin/acos/atan2、pow、hypot、AGM 与 K/E（S–M 段） | 同 P4，含巨大参数与参数域扫描 |
| P6 H–X 段 | L0 的 BurstTree（复用 P3 的 L0 纵向内核）、多素数原子库、Newton–Taylor log/atan、高斯 sincos、H 段 AGM、Team | H/X 段目标；"原子库未建"、"只清计划"、冷启动三条附加曲线 |
| P7 完整性 | 特殊函数（Γ 族、erf、W）、γ 的矩阵合并配方与 γ 生产者、公开的批量接口、C 接口、S 段 `inline_export`、复球评估 | 批量吞吐目标；全函数全段复核；J/R/B |
| P8 L3 | 正确舍入定精度层 | 推迟，另行立项 |

**P0 删除 astra 浮点代码的完整清单**：

- 库：
  - `src/float/`；`src/cpp/float*.{cpp,hpp}`；`include/sbn3/float*.hpp`；
  - `include/sbn3/sbn.hpp` 与 `context.hpp` 中对它们的引用；
  - `config/sources.json`、`tests.json`、`examples.json` 中的对应条目；
  - `tests/*/float_*`、`tests/integration/cpp_float_test.cpp`、`examples/cpp_float.cpp`；
  - `docs/float.md`、`docs/bigfloat-*.md`、`docs/elementary-implementation-design.md`；删除 `README.md` 的整节「## C++ bigfloats」（现为第 67–79 行）和能力列表中的 bigfloat 一行（现为第 34 行），修好 `docs/architecture.md`、`README.md`、`docs/ooc-architecture-cleanup-2026-10-05.md`（第 6 行）中指向它们的链接。
- 实验仓库：`bench/ops/bigfloat.cpp`；`config/benchmarks.json` 中的 `perf_bigfloat`；`tools/perf.py` 中的 `bigfloat` 适配器与 float-campaign/float-report/float-diagnostics 子命令；`tools/perflib/float_{campaign,report,model,diagnostics}.py`；`tools/perflib/build.py`、`model.py`、`runner.py` 中的 bigfloat 分支；`bench/common/harness.cpp` 的 F1 扩展；`config/experiments/bigfloat-*.json`（8 个）。

每个阶段的测试和基准曲线在它开始的第一天就存在。验收数字只来自 warm 主指标曲线，外加该阶段要求的附加曲线。

---

## 13. 代理协作规则

- 开始任何浮点任务前，先读本章程和算法文档的相关小节。与 AGENTS.md 冲突时：§15 第 1、2 项确认之前按 AGENTS.md 执行（或只做不涉及这些规则的工作）；确认并在两份 AGENTS.md 加上指向本章程的一行之后，按本章程执行。
- 报告不超过一页；不新建文档类型、注册表或框架，除非能证明它节省时间。
- 不为"完成度"写代码：不实现不会被调度到的算法，不写没人读的检查。
- 测试和基准从第一天就存在；不允许先写完整实现再补测量。
- 并行代理共用基准锁；不对别人工作树中的文件做无关修改；修改共享文件（注册表、L0 源码、`build.py`）之前先协调。

---

## 14. 合入前检查清单

1. fbench 的 t/target 都 ≤1 吗？哪个格子还没达标？
2. 曲线平滑吗？有没有违反 §3 第 8 条的新断崖？
3. 包含性、acc_bits、规格自洽和库内测试都通过了吗？
4. 违反 §3 热路径规则了吗？计时循环里有没有计划缓存未命中？
5. `LOG.md` 记了一行吗？
6. 有没有新增文档类型或留存物？

---

## 15. 待用户确认

1. **主指标**：把 2026-09-18 的"fresh 调用计时"规则重新解释为"warm 主指标（含进程级计划缓存）+ 只清计划的检查 + 冷启动锚点"（§7）。理由：浮点的常数和表是静态的；计划是进程级、按尺寸类缓存的；竞品也是在缓存热的状态下使用的；规划和冷启动的成本仍然要测，并且有上限。
2. **AGENTS.md 的另外两项覆盖**：不留存原始重复样本；内存准入只在 ≥2^24 bit 或多线程时做。确认后，在两份 AGENTS.md 末尾各加一行："浮点（src/fp、experiments/float）工作遵循 docs/float-charter.md，冲突时以其为准"。
3. **精度语义**：类 Arb，n = ⌈p/64⌉，不自动加守护 limb，交付精度 ≥ 64n − c_f（算法文档 §1.2）。实验 #15 完成之前，这一条保持开放。
4. **git worktree 与 `float-rewrite` 分支**，以及在该分支上做常规本地提交；修改 `build.py` 以共享绝对路径的锁。前提是浮点依赖的 L0 改动已由其负责人提交（§10）。
5. **打归档 tag（§10 的流程与核对），再按 §12 的清单删除 astra 浮点代码。** `include/sbn3/context.hpp`、`sbn.hpp` 和各注册表都含有其他代理未提交的改动，需要与 C++ 接口的负责人协调。
6. **实验仓库中 astra 的产物**：把 `results/bigfloat_*` 与 `results/perf/bigfloat-*` 打包成一个归档文件；删除 `results/perf/artifacts/*/` 中 `artifact.json` 的 adapter 为 `bigfloat` 的 9 个 `driver` 二进制（共约 1.2 GB），保留它们的 `artifact.json` 和源码快照。
7. **实验仓库的全局清理**（`git gc`、删除可以再生成的未跟踪产物）。
8. **L0 改动的协调**：算法文档 §6.5 列出的改动，有几项落在其他代理正在编辑、且已与 y-cruncher 对标认证过的文件上。
