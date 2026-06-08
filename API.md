## API 参考

| Method | Path | Auth | Rate Limit | Description |
|--------|------|------|------------|-------------|
| `GET` | `/health/live` | — | — | 存活探针，进程存活即返回 200 |
| `GET` | `/health/ready` | — | — | 就绪探针，检查模型已加载 + GPU 可用 + 队列未满，否则 503 |
| `GET` | `/health/gpu` | — | — | GPU 显存指标（设备名、总/已用/空闲 MB、利用率%） |
| `POST` | `/v1/audio/transcriptions` | Bearer token（可选） | 60 rpm/IP（可配） | OpenAI 兼容语音转文本 |

### POST /v1/audio/transcriptions

**请求：** `multipart/form-data`

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | file | ✅ | — | 音频文件，支持 ffmpeg 可解码的所有格式 |
| `language` | string | — | `zh` | 语言代码，空字符串 = 自动检测 |
| `response_format` | string | — | `json` | `json` 仅返回 text；`verbose_json` 返回 segments + 元数据 |

**响应 `json`：**
```json
{ "text": "转录文本" }
```

**响应 `verbose_json`：**
```json
{
  "text": "完整转录文本",
  "language": "zh",
  "duration": 12.34,
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "片段文本",
      "tokens": [50364, 104, "..."],
      "temperature": 0.0,
      "avg_logprob": -0.23,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.01
    }
  ]
}
```

**错误码：**

| 状态码 | 类型 | 说明 |
|--------|------|------|
| `400` | `invalid_request` | 文件为空、格式不支持、音频过长/过短 |
| `401` | `unauthorized` | API key 缺失或无效 |
| `413` | `file_too_large` | 超过 `MAX_UPLOAD_BYTES`（默认 500MB） |
| `429` | `rate_limited` | 超过 `RATE_LIMIT_RPM`（默认 60次/分钟） |
| `500` | `internal_error` | ffmpeg 超时、模型推理失败等 |
| `503` | `service_unavailable` | 模型未加载或 GPU 不可用 |

### GET /health/ready

```json
{
  "status": "ready",
  "model_loaded": true,
  "gpu_available": true,
  "queue_depth": 0
}
```

### GET /health/gpu

```json
{
  "device_name": "NVIDIA GeForce RTX 4090",
  "device_index": 0,
  "total_memory_mb": 24564,
  "used_memory_mb": 3120,
  "free_memory_mb": 21444,
  "utilization_pct": 12.7
}
```
