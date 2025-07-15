# 多人对话TTS功能

这个模块为Gemini Balance项目添加了多人语音TTS（Text-to-Speech）功能，采用继承模式设计，保持与原始代码的完全兼容性。

## 🎯 设计原则

- **继承而非修改**：所有扩展都继承自原始类，不修改源码
- **向后兼容**：原始功能完全不受影响
- **环境变量控制**：通过 `ENABLE_TTS` 环境变量动态启用
- **完整日志记录**：包含请求日志、错误日志和性能监控
- **易于维护**：更新原始代码时不会产生冲突

## 📁 文件结构

```
app/service/tts/
├── tts_service.py           # 原有的基础TTS服务
└── multi_speaker/           # 多人对话TTS扩展
    ├── __init__.py          # 模块初始化
    ├── README.md            # 使用说明（本文件）
    ├── tts_models.py        # TTS数据模型（继承自原始模型）
    ├── tts_response_handler.py  # TTS响应处理器（继承自原始处理器）
    ├── tts_chat_service.py  # TTS聊天服务（继承自原始服务）
    ├── tts_config.py        # TTS配置管理和工厂方法
    └── tts_routes.py        # TTS路由扩展和依赖注入
```

## 🚀 启用TTS功能

### 自动集成（当前实现）

TTS功能已经完全集成到主路由中，通过环境变量自动控制：

1. **TTS功能默认启用**：
```bash
# 直接启动服务，TTS功能已默认启用
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

2. **禁用TTS功能**（如需要）：
```bash
# Windows PowerShell
$env:ENABLE_TTS="false"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Linux/macOS
export ENABLE_TTS=false
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 工作原理

系统会自动检测 `ENABLE_TTS` 环境变量：
- `true`, `1`, `yes`, `on`（默认值）：启用TTS功能
- `false`, `0`, `no`, `off`：使用原始服务

```python
# app/router/gemini_routes.py 中的自动切换逻辑
async def get_chat_service(key_manager: KeyManager = Depends(get_key_manager)):
    import os
    if os.getenv("ENABLE_TTS", "false").lower() in ("true", "1", "yes", "on"):
        return await get_tts_chat_service(key_manager)
    else:
        return GeminiChatService(settings.BASE_URL, key_manager)
```

## 📝 使用示例

### 多人语音TTS请求

启用TTS功能后，可以发送多人语音请求：

```bash
curl -X POST "http://localhost:8000/v1beta/models/gemini-2.5-flash-preview-tts:generateContent" \
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

### 普通文本生成（兼容性测试）

TTS功能启用后，普通文本生成仍然正常工作：

```bash
curl -X POST "http://localhost:8000/v1beta/models/gemini-1.5-flash:generateContent" \
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

1. **环境检测**：系统启动时检查 `ENABLE_TTS` 环境变量（默认为true）
2. **服务选择**：根据环境变量选择 `GeminiChatService` 或 `TTSGeminiChatService`
3. **请求处理**：
   - **TTS模型**：使用 `_handle_tts_request()` 处理
   - **普通模型**：调用父类 `generate_content()` 方法
4. **字段处理**：从原始HTTP请求体提取TTS字段（`responseModalities`, `speechConfig`）
5. **API调用**：构建完整payload并调用Gemini API
6. **响应处理**：
   - **TTS响应**：检测音频数据，直接返回原始响应
   - **普通响应**：使用父类处理方法
7. **日志记录**：记录请求时间、成功状态、错误信息到数据库

## 📊 功能特性

### ✅ 已实现功能

- **多人语音合成**：支持 `multiSpeakerVoiceConfig` 配置
- **自动模型检测**：根据模型名称自动启用TTS处理
- **完整日志记录**：请求日志、错误日志、性能监控
- **API配额管理**：自动重试和密钥轮换
- **向后兼容性**：原始功能完全不受影响
- **环境变量控制**：TTS功能默认启用，可通过环境变量禁用
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

# TTS功能已默认启用，如需禁用可设置：
# export ENABLE_TTS=false
```

## 🎉 成功案例

基于继承的TTS解决方案已经成功实现：

- ✅ **完全向后兼容**：原始功能零影响
- ✅ **多人语音合成**：支持复杂的对话场景
- ✅ **完整日志记录**：可在管理界面查看所有请求
- ✅ **环境变量控制**：默认启用，可灵活控制
- ✅ **错误处理完善**：API配额和重试机制
- ✅ **易于维护**：更新原始代码无冲突

这个实现展示了如何在不修改原始代码的情况下，优雅地扩展复杂系统的功能。
