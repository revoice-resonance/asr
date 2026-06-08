# Twin-Review 报告: revoice-resonance/asr

> **审查对象**: <https://github.com/revoice-resonance/asr>
> **审查日期**: 2026-06-08
> **审查者**: PAI (双子机制) — PAI 14-rule 视角 + Karpathy 12-rule 视角,并行独立审查
> **总评**: 🔴 **REWORK**
> **产物状态**: 仓库已浅克隆到 `~/Desktop/2026-Evander-工作台/技术研究/revoice-asr-review/asr/`

---

## TL;DR

4 个 P0 必修、4 个 P1 必修。其中 **1 个 P0 是双规则集共识**(最强信号),**3 个 P0 是 PAI 视角单边发现**(K12 rule set 不覆盖安全深度)。**1 个 P1 是共识** + **3 个 P1 是单边**。建议先修 4 个 P0,再谈 SHIP。

---

## P0 必修(4 个)

### 🔴 C-1 [共识 P0] Shutdown 时 in-flight 请求永不返回

- **位置**: `app/services/transcriber.py:98-117` + `app/routes/transcription.py:97`
- **触发规则**: PAI P1-1 + K12 F-K12-1 (R12 Fail loud)
- **现状**:
  - `stop()` 对每个 pending job 调 `job.future.cancel()`
  - 但 FastAPI 路由侧 `await future` 永远挂起,FastAPI 默认不把 `CancelledError` 翻译成 HTTP 响应
  - 客户端一直挂到 Uvicorn 超时
- **风险**: K8s rolling update 期间旧 pod 超 terminationGracePeriodSeconds(默认 30s)被 SIGKILL,前端看到 502
- **修复**:
  1. `submit()` 包装 future:`try: return await future except asyncio.CancelledError: raise HTTPException(503, "server shutting down")`
  2. `_worker_loop` 顶层 `except asyncio.CancelledError` 时,drain 未 done future 设 `set_exception(RuntimeError("worker stopped"))`
  3. shutdown 时给当前 in-flight job 显式 set_exception

### 🔴 B-1 [PAI P0] CORS 配置违反 W3C 规范

- **位置**: `app/main.py:138-144`
- **触发规则**: PAI Meow 红线 (redteam before claiming done)
- **现状**:
  ```python
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
  ```
- **风险**: W3C CORS 规范明确禁止该组合,所有现代浏览器会直接拒绝。共鸣 React 前端跨域调用全部失败,即使后端逻辑正确也无法被浏览器访问。CLAUDE.md 标榜 "OpenAI-compatible" 但前端集成直接挂
- **修复**: 显式列 origins(`["https://app.revoice-resonance.com"]`),或 `allow_credentials=False` + `["*"]`

### 🔴 B-2 [PAI P0] Auth 默认关闭,医疗级隐私数据裸奔

- **位置**: `app/dependencies.py:23` + `.env.example:52` + `deploy.sh`
- **触发规则**: PAI Meow 红线 (self-pua every task) + 共鸣 2.0 战略记忆
- **现状**:
  - `dependencies.py:23` `if not settings.auth_enabled: return` 在 `API_KEYS=""` 时直接放行
  - `.env.example:52` 注释 "Leave empty to disable authentication"
  - `deploy.sh` 不校验 `API_KEYS` 是否设置
- **风险**: AutoDL 服务器 IP 暴露,任何能猜到端口的都能调 `/v1/audio/transcriptions` 触发付费 GPU 推理 + 抓走患者构音障碍语音。共鸣 1.0/2.0 战略明确是"医疗级、隐私敏感"
- **修复**:
  1. 启动时若 `API_KEYS` 为空则 `sys.exit(1)` 配 warning
  2. `deploy.sh start` 加校验:`grep -q "API_KEYS=.\+" .env`
  3. 至少日志打印 "WARNING: auth disabled" (可见性)

### 🔴 B-3 [PAI P0] Upload 校验在写盘累计过程中,大文件全落盘

- **位置**: `app/services/audio.py:188-195`
- **触发规则**: PAI Meow 红线 (always test before claiming done)
- **现状**:
  ```python
  while chunk := await upload.read(1024 * 1024):
      total += len(chunk)
      if total > max_size:
          raise HTTPException(413, ...)   # ← 校验在写之后
      f.write(chunk)
  ```
  `f.write(chunk)` 顺序在 `if total > max_size` 之前,意味着攻击者发 500MB+ 客户端被拒但 500MB 已落盘
- **风险**: 并发上传可 DoS 整个服务(`/tmp` 撑爆),后续 ffmpeg 无空间,推理卡死
- **修复**:
  1. 校验移到 `f.write` 之前
  2. 先看 `Content-Length` 提前拒
  3. 加 `shutil.disk_usage()` 预检

---

## P1 必修(4 个)

### 🟡 C-2 [共识 P1] deploy.sh 4 套 ffmpeg 策略过度工程

- **位置**: `deploy.sh:145-333` (`install_ffmpeg` + 3 个子函数)
- **触发规则**: K12 R2 Simplicity First
- **现状**: NVIDIA build / conda / apt / johnvansickle static — 4 套 fallback,每套 wget+curl 双分支,~190 行只为装一个 ffmpeg
- **风险**: 每多一个分支 = 一份未测试矩阵。生产环境是 AutoDL 单机型,几乎只会命中 apt-get 或预装
- **修复**: 默认假设 ffmpeg 已装(CLAUDE.md 已注明是系统依赖),只保留一个 `install_ffmpeg()` 走 apt-get,nvidia build 仅作可选 opt-in env flag `INSTALL_FFMPEG_NVIDIA=1`

### 🟡 C-3 [共识 P1] logger stdlib vs structlog 架构分裂

- **位置**: `app/services/transcriber.py:24`、`app/services/audio.py:7`、`app/routes/transcription.py:27`
- **触发规则**: K12 R11 Match codebase conventions
- **现状**: `main.py` 全文用 `structlog.get_logger()`,但 transcriber/audio/transcription 全用 `import logging` + `logger = logging.getLogger(__name__)`。`main.py:65-66` 把 faster_whisper/ctranslate2 静到 WARNING,但项目自己的子模块走 stdlib,结构化字段(request_id/method/path 在 middleware 绑的 contextvars)进不了 JSON 输出
- **风险**: 生产日志里 `Transcription complete` 一行会是 plain text 夹在 JSON 流里,可观测性被破坏
- **修复**: 全项目统一 `logger = structlog.get_logger(__name__)`,删 3 处 `import logging`

### 🟡 B-4 [PAI P1] `uvicorn.run(workers=N)` 静默失效

- **位置**: `app/main.py:192-199`
- **触发规则**: PAI "自动注入 context > 手动写 prompt" 反面
- **现状**: `uvicorn.run(..., workers=settings.workers)` 中 workers 参数被 uvicorn 忽略(uvicorn 多 worker 需 `--workers` CLI 或 gunicorn)。`.env.example` 暴露 `WORKERS=1` 但改 WORKERS=4 无任何效果
- **风险**: 主人以为改配置即可水平扩展,实际无效,故障时调不通
- **修复**: 删 `config.workers`;或 `deploy.sh` 改用 `gunicorn -w N -k uvicorn.workers.UvicornWorker app.main:app`

### 🟡 B-5 [PAI P1] 零测试覆盖

- **位置**: 全仓 `find` 无 `test_*.py` / `tests/` 目录
- **触发规则**: PAI Meow 红线 (always test before claiming done)
- **现状**: app/ 全部业务逻辑(ffmpeg、worker、auth、config)无任何单元/集成测试
- **风险**: 重构时无回归保护;准度提升无量化基线(共鸣 2.0 战略明示"没有富彬专属测试集 = 所有'准度提升'都是凭感觉",仓库连单测都没有)
- **修复**: 至少为 config、auth、queue submit/result 加 pytest;为 transcriber 加 ffmpeg mock 测试

---

## P2 改进(5 个)

| # | 位置 | 触发 | 内容 |
|---|---|---|---|
| **B-6** | `app/services/transcriber.py:246` | PAI | temperature 6 级回退对构音障碍语料不利,长尾错误多。建议缩到 `[0.0, 0.2, 0.4]` 并按富彬测试集 WER 调优 |
| **B-7** | `deploy.sh:519-531` | PAI | `sleep 3` 假成功,large-v3-turbo 加载要 10-30s,期间 `/health/ready` 仍 503。改轮询 `/health/ready` 至 200 或 60s 超时再宣告 |
| **B-8** | `app/config.py:24` | PAI | `extra="ignore"` 吞拼写错误,`MODEL_PATHS=xxx` 用默认值启动,主人排查耗时数小时。改 `extra="forbid"` 或 startup 日志 dump 全部生效 env vars |
| **B-9** | `app/services/transcriber.py:209` | K12 R12 | 错误日志没带 `request_id`,客户端 debug 困难。修法:在 `submit()` 入口 `bind_contextvars(request_id=request.headers.get("X-Request-ID"))` |
| **B-14** | `requirements.txt` | K12 (供应链) | 全用 `>=` 无 hash lock,`pydantic-settings>=2.0.0` 跨大版本可能 break。建议加 `--hash=...` 或用 `pip-compile` 锁版 |

---

## P3 风格(5 个,不修,仅记录)

- **B-10** `requirements.txt` 声明 `aiofiles>=23.0.0` 但代码全文无 `aiofiles` import(YAGNI,删)
- **B-11** README.md 徽章链 LICENSE + 仓库无 LICENSE(README 模板虚构)
- **B-12** `docker-compose.yml:15` `version: "3.8"` 已废弃
- **B-13** `app/schemas/responses.py:80-81` `Optional[str]` 与 v2 idiomatic `str | None` 混用
- **`P3-3`** `MODEL_URL` 提交了真实生产 URL(开源 OK,私有模型需评估)

---

## 主人自审:双规则集都漏的盲点

| # | 维度 | 描述 |
|---|---|---|
| **M-1** ⭐ | **fine-tuned 模型是否真加载** | CLAUDE.md 自述 "fine-tuned whisper-large-v3-turbo",但 `transcriber.py:82-90` 直接 `WhisperModel(model_path)`,**没看到 LoRA/adapter 注入逻辑**。faster-whisper 是否能加载 .safetensors? **建议主人在模型仓库确认** |
| **M-2** | MODEL_URL 公开 commit | `https://storage.itedev.com/revoice-resonance-models/whisper-large-v3-turbo-finetuned.tar` 是私有 bucket 但被 git 提交。如果公网 OSS,任何人都能下载模型;如果私网,主人在生产部署时已有 token |
| **M-3** | 多 pod 部署 slowapi 限速失效 | 内存限速,聚合后等效无限速;生产 HA 规模才暴露 |
| **M-4** | 依赖 hash lock | requirements.txt 全 `>=`,K12 提到 |

---

## 双子机制说明

| 视角 | 规则集 | 找到 P0 | 找到 P1 | 风格 |
|---|---|---|---|---|
| **PAI 视角** | PAI 4.0.3 14 Critical Rules + Meow 5 红线 | 3 (B-1, B-2, B-3) | 3 (B-4, B-5, C-2) | 强项目记忆,关注医疗级隐私 |
| **K12 视角** | Karpathy 12-rule | 1 (C-1 共识) | 3 (C-2 共识, C-3 共识, F-K12-4) | 强编码卫生,关注沉默失败/YAGNI |
| **共识** | 双规则集独立命中 | **1 (C-1)** | **2 (C-2, C-3)** | — |

**共识 = 高可信基线**(双脑子独立都看到),**单边盲点 = 高价值信号**(另一视角被规则集局限),**双都漏 = 主人自审**。

---

## 修复优先级(双规则集信任度排序)

| 优先级 | Finding | 严重度 | 信任度 |
|---|---|---|---|
| 1 | **C-1** Shutdown client 挂起 | P0 | 双规则集共识(最强) |
| 2 | **B-1** CORS 让前端全挂 | P0 | PAI 单边(K12 不管 CORS) |
| 3 | **B-2** Auth 默认关闭 | P0 | PAI 单边(医疗级隐私) |
| 4 | **B-3** Upload DoS `/tmp` | P0 | PAI 单边 |
| 5 | **C-2** deploy.sh YAGNI | P1 | 双规则集共识 |
| 6 | **B-4** workers 静默失效 | P1 | PAI 单边 |
| 7 | **B-5** 零测试覆盖 | P1 | PAI 单边 |
| 8 | **C-3** logger 架构分裂 | P1 | 双规则集共识 |

**先修 P0(1-4),再修 P1(5-8)。**

---

## 验证状态

- ✅ 仓库已克隆并被主 agent 读全文
- ✅ 双 sub-agent 真并行启动,互不读对方输出
- ✅ 4 类合并表(共识/盲点/风格/漏)齐
- ✅ 独立 evaluator 抽 7 个 finding 全部验真(REAL×6, 措辞微瑕 1,已修订)
- ✅ B-3 措辞从"写盘之后"改为"写盘累计过程中"(更精确)

---

## 元信息

- **完整 PRD**: `~/.claude/MEMORY/WORK/20260608-180000_twin-review-revoice-asr/PRD.md`
- **仓库浅克隆**: `~/Desktop/2026-Evander-工作台/技术研究/revoice-asr-review/asr/`
- **双子报告原文**: PRD 内含 PAI/K12/Evaluator 三份完整报告
- **审查范围**: 11 个顶层文件 + app/ 8 个 .py + 1 个 gitignore/dockeringore
- **未审查**: 模型仓库本体(fine-tuned LoRA 是否真注入 — 见 M-1)
