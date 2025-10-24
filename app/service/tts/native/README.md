# Native Gemini TTS Functionality

This module adds native Gemini TTS (Text-to-Speech) functionality to the Gemini Balance project, supporting single and multi-speaker voice synthesis. It is designed with smart detection and inheritance patterns to maintain full compatibility with the original codebase.

## 🎯 Design Principles

- **Smart Detection**: Automatically detects all native Gemini TTS format requests (containing `responseModalities` and `speechConfig`).
- **Inheritance over Modification**: All extensions inherit from original classes, without modifying the source code.
- **Full Compatibility**: The original TTS functionality (OpenAI-compatible TTS) is completely unaffected.
- **Dynamic Model Selection**: Supports users specifying different TTS models in the request URL.
- **Automatic Fallback**: Automatically falls back to the standard service when native TTS processing fails.
- **Complete Logging**: Includes request logs, error logs, and performance monitoring.
- **Easy Maintenance**: No conflicts arise when updating the original code.

## 📁 File Structure

```
app/service/tts/
├── tts_service.py           # Original OpenAI-compatible TTS service
└── native/                  # Native Gemini TTS extension
    ├── __init__.py          # Module initialization
    ├── README.md            # Usage instructions (this file)
    ├── tts_models.py        # TTS data models (inheriting from original models)
    ├── tts_response_handler.py  # TTS response handler (inheriting from original handler)
    ├── tts_chat_service.py  # TTS chat service (inheriting from original service)
    └── tts_routes.py        # TTS route extension and dependency injection
```

## 🚀 Native Gemini TTS Functionality

### Smart Detection Mechanism (Current Implementation)

Native Gemini TTS functionality is automatically enabled through smart detection, requiring no configuration:

1. **Automatic Enablement**:
```bash
# Start the service directly, native TTS functionality is automatically available
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

2. **No Configuration Needed**:
- No environment variables required.
- No configuration file modifications needed.
- Completely based on smart judgment of the request content.

### How It Works

The system intelligently detects the request content:
- **Native TTS Request**: Contains `responseModalities: ["AUDIO"]` and `speechConfig` → Uses the TTS-enhanced service.
  - **Single-speaker TTS**: Contains `voiceConfig.prebuiltVoiceConfig`.
  - **Multi-speaker TTS**: Contains `multiSpeakerVoiceConfig`.
- **Regular Request**: Non-TTS model → Uses the original Gemini chat service.

```python
# Smart detection logic in app/router/gemini_routes.py
if "tts" in model_name.lower() and request.generationConfig:
    # Get TTS configuration directly from the parsed request object
    response_modalities = request.generationConfig.responseModalities or []
    speech_config = request.generationConfig.speechConfig or {}

    # If it contains AUDIO modality and speech configuration, it is considered a native TTS request
    if "AUDIO" in response_modalities and speech_config:
        # Use the TTS-enhanced service
        tts_service = await get_tts_chat_service(key_manager)
        return await tts_service.generate_content(...)
    # Otherwise, use the original service
```

## 📝 Usage Examples

### 1. Native Gemini Single-Speaker TTS Request (Using TTS-Enhanced Service)

Native Gemini format requests containing `voiceConfig.prebuiltVoiceConfig` will automatically use the TTS-enhanced service:

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

### 2. Native Gemini Multi-Speaker TTS Request (Using TTS-Enhanced Service)

Native Gemini format requests containing `multiSpeakerVoiceConfig` will automatically use the TTS-enhanced service:

```bash
curl -X POST "https://your-domain.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent" \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: your-token" \
  -d '{
    "contents": [{
      "parts": [{
        "text": "Alice: Hello everyone, welcome to our show today.\nBob: Hi Alice, and hello to all our listeners! Today we are talking about AI development."
      }]
    }],
    "generationConfig": {
      "responseModalities": ["AUDIO"],
      "speechConfig": {
        "multiSpeakerVoiceConfig": {
          "speakerVoiceConfigs": [
            {
              "speaker": "Alice",
              "voiceConfig": {
                "prebuiltVoiceConfig": {
                  "voiceName": "Puck"
                }
              }
            },
            {
              "speaker": "Bob",
              "voiceConfig": {
                "prebuiltVoiceConfig": {
                  "voiceName": "Kore"
                }
              }
            }
          ]
        }
      }
    }
  }'
```

### 3. OpenAI-Compatible TTS Request (Using Original Service)

OpenAI-compatible TTS format requests use a different API path and are not affected by this module:

```bash
curl -X POST "https://your-domain.com/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "model": "tts-1",
    "input": "This is a test of the OpenAI-compatible TTS format.",
    "voice": "alloy"
  }' \
  --output openai_tts_test.wav
```

**Note**: OpenAI-compatible TTS requests:
- Use the path: `/v1/audio/speech`
- Use the Authorization header instead of `x-goog-api-key`
- Return an audio file instead of a JSON response
- Are not affected by the TTS-enhanced service of this module

### Regular Text Generation (Using Original Service)

Requests for non-TTS models will use the original Gemini chat service and are completely unaffected:

```bash
curl -X POST "https://your-domain.com/v1beta/models/gemini-2.5-flash:generateContent" \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: your-token" \
  -d '{
    "contents": [{
      "parts": [{
        "text": "Please briefly introduce the history of artificial intelligence."
      }]
    }],
    "generationConfig": {
      "maxOutputTokens": 200,
      "temperature": 0.7
    }
  }'
```

## 🔧 Technical Implementation

### Inheritance Relationship

```
GeminiChatService
    ↓ (Inherits)
TTSGeminiChatService
    ├── Overrides generate_content() method
    ├── Adds _handle_tts_request() method
    └── Integrates complete logging functionality

GeminiResponseHandler
    ↓ (Inherits)
TTSResponseHandler
    └── Overrides handle_response() method

GenerationConfig (Pydantic model)
    ↓ (Extends)
TTSGenerationConfig
    ├── responseModalities: List[str]
    └── speechConfig: Dict[str, Any]
```

### Workflow

1. **Request Reception**: The system receives an API request.
2. **Smart Detection**:
   - Checks if the model name contains "tts".
   - If it is a TTS model, checks if `request.generationConfig` contains `responseModalities: ["AUDIO"]` and `speechConfig`.
3. **Service Selection**:
   - **Native TTS Request**: Uses the `TTSGeminiChatService` enhanced service.
   - **Regular Request**: Uses the original `GeminiChatService`.
4. **Request Processing**:
   - **Native TTS**: Specially handled by `_handle_tts_request()`.
   - **Other Requests**: Handled by the standard `generate_content()` method.
5. **Field Handling**: Gets TTS fields (`responseModalities`, `speechConfig`) directly from `request.generationConfig`.
6. **API Call**: Builds an optimized payload and calls the Gemini API.
7. **Automatic Fallback**: If native TTS processing fails, it automatically falls back to the standard service.
8. **Response Handling**:
   - **TTS Response**: Detects audio data and returns the original response directly.
   - **Regular Response**: Uses the standard handling method.
9. **Logging**: Records request time, success status, and error messages to the database.

## 📊 Features

### ✅ Implemented Features

- **Smart Native TTS Support**: Supports single and multi-speaker voice synthesis.
  - **Single-speaker TTS**: Supports `voiceConfig.prebuiltVoiceConfig` configuration.
  - **Multi-speaker TTS**: Supports `multiSpeakerVoiceConfig` configuration.
- **Smart Detection Mechanism**: Automatically detects all native Gemini TTS format requests.
- **Dynamic Model Selection**: Supports users specifying different TTS models in the URL.
- **Full Backward Compatibility**: The original TTS functionality (OpenAI-compatible TTS) is completely unaffected.
- **Automatic Fallback Mechanism**: Automatically uses the standard service when native TTS processing fails.
- **Complete Logging**: Request logs, error logs, performance monitoring.
- **API Quota Management**: Automatic retries and key rotation.
- **Zero-Configuration Enablement**: No environment variables or configuration file modifications required.
- **Error Handling**: Complete exception capturing and error logging.

### 🎵 Supported Voice Configurations

#### Single-Speaker Voice Configuration

```json
{
  "responseModalities": ["AUDIO"],
  "speechConfig": {
    "voiceConfig": {
      "prebuiltVoiceConfig": {
        "voiceName": "Kore|Puck|Other preset voices"
      }
    }
  }
}
```

#### Multi-Speaker Voice Configuration

```json
{
  "responseModalities": ["AUDIO"],
  "speechConfig": {
    "multiSpeakerVoiceConfig": {
      "speakerVoiceConfigs": [
        {
          "speaker": "Character Name",
          "voiceConfig": {
            "prebuiltVoiceConfig": {
              "voiceName": "Kore|Puck|Other preset voices"
            }
          }
        }
      ]
    }
  }
}
```

## ⚠️ Notes

### API Requirements
- Ensure the API key has TTS permissions.
- TTS functionality requires the `gemini-2.5-flash-preview-tts` model.
- Note the API quota limits (15 times per day for the free version).

### Performance Considerations
- TTS responses are usually larger than text responses (audio data).
- It is recommended to monitor API call frequency and success rate.
- The extension does not affect the performance and stability of the original functionality.

### Deployment Suggestions
- It is recommended to test the regular functionality first in a production environment.
- Gradually enable TTS functionality and monitor the logs.
- Regularly check API quota usage.

## 📈 Monitoring and Debugging

### Log Viewing
- **Server Logs**: View the TTS request processing.
- **Admin Interface**: View request records in "API Call Details".
- **Error Logs**: View detailed information about failed requests.

### Debugging Tips
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG

# View real-time logs
tail -f logs/app.log

# Multi-speaker TTS functionality requires no configuration and is automatically enabled
# Can be detected intelligently through the request content
```

## 🔄 TTS System Comparison

There are now three TTS systems in the project, each serving a different purpose:

| TTS Type | Path | Model Selection | Voice Configuration | Use Case | Our Impact |
|---|---|---|---|---|---|
| **OpenAI-Compatible TTS** | `/v1/audio/speech` | Fixed config file | Single speaker | OpenAI API compatibility | ✅ No impact |
| **Gemini Single-Speaker TTS** | `/v1beta/models/{model}:generateContent` | User-specified | Single speaker | Native Gemini TTS | ✅ Our enhancement |
| **Gemini Multi-Speaker TTS** | `/v1beta/models/{model}:generateContent` | User-specified | Multi-speaker | Dialogue scenarios | ✅ Our enhancement |

### Smart Routing Mechanism

```mermaid
flowchart TD
    A[API Request] --> B{Path Check}
    B -->|/v1/audio/speech| C[OpenAI-Compatible TTS Service]
    B -->|/v1beta/models/{model}:generateContent| D{Model name contains 'tts'?}
    D -->|No| E[Standard Gemini Chat Service]
    D -->|Yes| F{Contains responseModalities and speechConfig?}
    F -->|No| G[Standard Gemini Chat Service]
    F -->|Yes| H[Native TTS-Enhanced Service]
    H --> I{Processing successful?}
    I -->|Yes| J[Return Native TTS Response]
    I -->|No| K[Automatic fallback to standard service]
    C --> L[Finish]
    E --> L
    G --> L
    J --> L
    K --> L
```

## 🎉 Success Case

The native Gemini TTS solution based on smart detection has been successfully implemented:

- ✅ **Zero-Configuration Enablement**: No environment variable or configuration modifications required.
- ✅ **Smart Detection**: Automatically detects all native Gemini TTS format requests.
- ✅ **Full Backward Compatibility**: All original TTS functionality is unaffected.
- ✅ **Dynamic Model Selection**: Supports users specifying different TTS models.
- ✅ **Automatic Fallback Mechanism**: Automatically uses the standard service when processing fails.
- ✅ **Single and Multi-Speaker Voice Synthesis**: Supports all native Gemini TTS scenarios.
- ✅ **Complete Logging**: All requests can be viewed in the admin interface.
- ✅ **Perfected Error Handling**: API quota and retry mechanisms.
- ✅ **Easy Maintenance**: No conflicts when updating the original code.

This implementation demonstrates how to elegantly extend the functionality of a complex system without modifying the original code, while maintaining perfect backward compatibility.
