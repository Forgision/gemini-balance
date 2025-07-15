# 多人对话TTS功能

这个模块为Gemini Balance项目添加了多人语音TTS（Text-to-Speech）功能，采用智能检测和继承模式设计，保持与原始代码的完全兼容性。

## 🎯 设计原则

- **智能检测**：只有包含 `multiSpeakerVoiceConfig` 的请求才启用多人TTS
- **继承而非修改**：所有扩展都继承自原始类，不修改源码
- **完全兼容**：原有TTS功能（单人TTS、OpenAI兼容TTS）完全不受影响
- **动态模型选择**：支持用户在请求URL中指定不同的TTS模型
- **自动回退**：多人TTS处理失败时自动回退到标准服务
- **完整日志记录**：包含请求日志、错误日志和性能监控
- **易于维护**：更新原始代码时不会产生冲突

## 📁 文件结构

```
app/service/tts/
├── tts_service.py           # 原有的OpenAI兼容TTS服务
└── multi_speaker/           # 多人对话TTS扩展
    ├── __init__.py          # 模块初始化
    ├── README.md            # 使用说明（本文件）
    ├── tts_models.py        # TTS数据模型（继承自原始模型）
    ├── tts_response_handler.py  # TTS响应处理器（继承自原始处理器）
    ├── tts_chat_service.py  # TTS聊天服务（继承自原始服务）
    └── tts_routes.py        # TTS路由扩展和依赖注入
```

## 🚀 多人TTS功能

### 智能检测机制（当前实现）

多人TTS功能通过智能检测自动启用，无需任何配置：

1. **自动启用**：
```bash
# 直接启动服务，多人TTS功能自动可用
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

2. **无需配置**：
- 不需要环境变量
- 不需要修改配置文件
- 完全基于请求内容智能判断

### 工作原理

系统会智能检测请求内容：
- **多人TTS请求**：包含 `multiSpeakerVoiceConfig` → 使用多人TTS增强服务
- **单人TTS请求**：不包含 `multiSpeakerVoiceConfig` → 使用原有Gemini TTS服务
- **普通请求**：非TTS模型 → 使用原有Gemini聊天服务

```python
# app/router/gemini_routes.py 中的智能检测逻辑
if "tts" in model_name.lower():
    # 检查是否包含多人语音配置
    speech_config = raw_data.get("generationConfig", {}).get("speechConfig", {})
    if "multiSpeakerVoiceConfig" in speech_config:
        # 使用多人TTS增强服务
        tts_service = await get_tts_chat_service(key_manager)
        return await tts_service.generate_content(...)
    # 否则使用原有服务
```

## 📝 使用示例

### 多人语音TTS请求（自动启用增强服务）

包含 `multiSpeakerVoiceConfig` 的请求会自动使用多人TTS增强服务：

```bash
curl -X POST "https://your-domain.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent" \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: your-token" \
  -d '{
    "contents": [{
      "parts": [{
        "text": "小雅： 听众朋友们大家好！欢迎收听今天的节目。\n李想： 小雅好，听众朋友们好！今天我们来聊聊人工智能的发展。"
      }]
    }],
    "generationConfig": {
      "responseModalities": ["AUDIO"],
      "speechConfig": {
        "multiSpeakerVoiceConfig": {
          "speakerVoiceConfigs": [
            {
              "speaker": "李想",
              "voiceConfig": {
                "prebuiltVoiceConfig": {
                  "voiceName": "Kore"
                }
              }
            },
            {
              "speaker": "小雅",
              "voiceConfig": {
                "prebuiltVoiceConfig": {
                  "voiceName": "Puck"
                }
              }
            }
          ]
        }
      }
    }
  }'
```

### 单人TTS请求（使用原有服务）

不包含 `multiSpeakerVoiceConfig` 的TTS请求会使用原有的Gemini TTS服务：

```bash
curl -X POST "https://your-domain.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent" \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: your-token" \
  -d '{
    "contents": [{
      "parts": [{
        "text": "Hello, this is a single speaker test."
      }]
    }],
    "generationConfig": {
      "responseModalities": ["AUDIO"],
      "speechConfig": {
        "voiceConfig": {
          "prebuiltVoiceConfig": {
            "voiceName": "Kore"
          }
        }
      }
    }
  }'
```

### 普通文本生成（使用原有服务）

非TTS模型的请求会使用原有的Gemini聊天服务，完全不受影响：

```bash
curl -X POST "https://your-domain.com/v1beta/models/gemini-1.5-flash:generateContent" \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: your-token" \
  -d '{
    "contents": [{
      "parts": [{
        "text": "请简单介绍一下人工智能的发展历程。"
      }]
    }],
    "generationConfig": {
      "maxOutputTokens": 200,
      "temperature": 0.7
    }
  }'
```

## 🔧 技术实现

### 继承关系

```
GeminiChatService
    ↓ (继承)
TTSGeminiChatService
    ├── 重写 generate_content() 方法
    ├── 添加 _handle_tts_request() 方法
    └── 集成完整的日志记录功能

GeminiResponseHandler
    ↓ (继承)
TTSResponseHandler
    └── 重写 handle_response() 方法

GenerationConfig (Pydantic模型)
    ↓ (扩展)
TTSGenerationConfig
    ├── responseModalities: List[str]
    └── speechConfig: Dict[str, Any]
```

### 工作流程

1. **请求接收**：系统接收到API请求
2. **智能检测**：
   - 检查模型名称是否包含 "tts"
   - 如果是TTS模型，解析请求体检查是否包含 `multiSpeakerVoiceConfig`
3. **服务选择**：
   - **多人TTS请求**：使用 `TTSGeminiChatService` 增强服务
   - **单人TTS请求**：使用原有 `GeminiChatService`
   - **普通请求**：使用原有 `GeminiChatService`
4. **请求处理**：
   - **多人TTS**：使用 `_handle_tts_request()` 特殊处理
   - **其他请求**：使用标准 `generate_content()` 方法
5. **字段处理**：从原始HTTP请求体提取TTS字段（`responseModalities`, `speechConfig`）
6. **API调用**：构建优化的payload并调用Gemini API
7. **自动回退**：如果多人TTS处理失败，自动回退到标准服务
8. **响应处理**：
   - **TTS响应**：检测音频数据，直接返回原始响应
   - **普通响应**：使用标准处理方法
9. **日志记录**：记录请求时间、成功状态、错误信息到数据库

## 📊 功能特性

### ✅ 已实现功能

- **智能多人语音合成**：支持 `multiSpeakerVoiceConfig` 配置
- **智能检测机制**：只有多人TTS请求才启用增强服务
- **动态模型选择**：支持用户在URL中指定不同TTS模型
- **完全向后兼容**：原有TTS功能（单人TTS、OpenAI兼容TTS）完全不受影响
- **自动回退机制**：多人TTS处理失败时自动使用标准服务
- **完整日志记录**：请求日志、错误日志、性能监控
- **API配额管理**：自动重试和密钥轮换
- **零配置启用**：无需环境变量或配置文件修改
- **错误处理**：完整的异常捕获和错误记录

### 🎵 支持的语音配置

```json
{
  "responseModalities": ["AUDIO"],
  "speechConfig": {
    "multiSpeakerVoiceConfig": {
      "speakerVoiceConfigs": [
        {
          "speaker": "角色名称",
          "voiceConfig": {
            "prebuiltVoiceConfig": {
              "voiceName": "Kore|Puck|其他预设语音"
            }
          }
        }
      ]
    }
  }
}
```

## ⚠️ 注意事项

### API要求
- 确保API密钥有TTS权限
- TTS功能需要 `gemini-2.5-flash-preview-tts` 模型
- 注意API配额限制（免费版每天15次）

### 性能考虑
- TTS响应通常比文本响应更大（音频数据）
- 建议监控API调用频率和成功率
- 扩展功能不影响原始功能的性能和稳定性

### 部署建议
- 生产环境建议先测试普通功能
- 逐步启用TTS功能并监控日志
- 定期检查API配额使用情况

## 📈 监控和调试

### 日志查看
- **服务器日志**：查看TTS请求处理过程
- **管理界面**：在"API 调用详情"中查看请求记录
- **错误日志**：查看失败请求的详细信息

### 调试技巧
```bash
# 启用详细日志
export LOG_LEVEL=DEBUG

# 查看实时日志
tail -f logs/app.log

# 多人TTS功能无需配置，自动启用
# 可通过请求内容智能检测
```

## 🔄 TTS系统对比

项目中现在有三套TTS系统，各自服务不同的用途：

| TTS类型 | 路径 | 模型选择 | 语音配置 | 使用场景 | 我们的影响 |
|---------|------|----------|----------|----------|------------|
| **OpenAI兼容TTS** | `/v1/audio/speech` | 固定配置文件 | 单人语音 | OpenAI API兼容 | ✅ 无影响 |
| **Gemini单人TTS** | `/v1beta/models/{model}:generateContent` | 用户指定 | 单人语音 | 原生Gemini TTS | ✅ 无影响 |
| **Gemini多人TTS** | `/v1beta/models/{model}:generateContent` | 用户指定 | 多人语音 | 对话场景 | ✅ 我们的增强 |

### 智能路由机制

```mermaid
flowchart TD
    A[API请求] --> B{路径检查}
    B -->|/v1/audio/speech| C[OpenAI兼容TTS服务]
    B -->|/v1beta/models/{model}:generateContent| D{模型名包含'tts'?}
    D -->|否| E[标准Gemini聊天服务]
    D -->|是| F{包含multiSpeakerVoiceConfig?}
    F -->|否| G[原有Gemini TTS服务]
    F -->|是| H[多人TTS增强服务]
    H --> I{处理成功?}
    I -->|是| J[返回多人TTS响应]
    I -->|否| K[自动回退到标准服务]
    C --> L[完成]
    E --> L
    G --> L
    J --> L
    K --> L
```

## 🎉 成功案例

基于智能检测的多人TTS解决方案已经成功实现：

- ✅ **零配置启用**：无需任何环境变量或配置修改
- ✅ **智能检测**：只有多人TTS请求才使用增强服务
- ✅ **完全向后兼容**：所有原有TTS功能零影响
- ✅ **动态模型选择**：支持用户指定不同TTS模型
- ✅ **自动回退机制**：处理失败时自动使用标准服务
- ✅ **多人语音合成**：支持复杂的对话场景
- ✅ **完整日志记录**：可在管理界面查看所有请求
- ✅ **错误处理完善**：API配额和重试机制
- ✅ **易于维护**：更新原始代码无冲突

这个实现展示了如何在不修改原始代码的情况下，优雅地扩展复杂系统的功能，同时保持完美的向后兼容性。
