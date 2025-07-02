from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.config.config import settings
from app.log.logger import get_main_logger
import re
import json

logger = get_main_logger()

class SmartRoutingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        # 简化的路由规则 - 直接根据检测结果路由
        pass

    async def dispatch(self, request: Request, call_next):
        if not settings.URL_NORMALIZATION_ENABLED:
            return await call_next(request)
        
        original_path = str(request.url.path)
        method = request.method
        
        # 尝试修复URL
        fixed_path, fix_info = self.fix_request_url(original_path, method, request)
        
        if fixed_path != original_path:
            logger.info(f"URL fixed: {method} {original_path} → {fixed_path}")
            if fix_info:
                logger.debug(f"Fix details: {fix_info}")
            
            # 重写请求路径
            request.scope["path"] = fixed_path
            request.scope["raw_path"] = fixed_path.encode()
        
        return await call_next(request)

    def fix_request_url(self, path: str, method: str, request: Request) -> tuple:
        """修复错误的请求URL - 简化版本"""

        # 首先检查是否已经是正确的格式，如果是则不处理
        if self.is_already_correct_format(path):
            return path, None

        # 检测是否为流式请求
        is_stream_request = self.detect_stream_request(path, request)

        # 1. 优先检测OpenAI格式请求（避免被v1beta误判）
        if self.is_openai_request(path, request):
            return self.fix_openai_request(path, method, request)

        # 2. 检测HF格式请求
        if self.is_hf_request(path, request):
            return self.fix_hf_request(path, method, request)

        # 3. 检测Vertex Express格式请求（优先级高于Gemini）
        if self.is_vertex_express_request(path, request):
            return self.fix_vertex_express_request(path, method, request, is_stream_request)

        # 4. 检测Gemini请求
        if self.is_gemini_request(path):
            return self.fix_gemini_request(path, method, request, is_stream_request)

        # 5. 默认处理其他请求（转为最快的v1端点）
        return self.fix_default_request(path, method, request)

    def is_already_correct_format(self, path: str) -> bool:
        """检查是否已经是正确的API格式"""
        # 检查是否已经是正确的端点格式
        correct_patterns = [
            r'^/v1beta/models/[^/:]+:(generate|streamGenerate)Content$',  # Gemini原生
            r'^/gemini/v1beta/models/[^/:]+:(generate|streamGenerate)Content$',  # Gemini带前缀
            r'^/v1beta/models$',  # Gemini模型列表
            r'^/gemini/v1beta/models$',  # Gemini带前缀的模型列表
            r'^/v1/(chat/completions|models|embeddings|images/generations)$',  # v1格式
            r'^/openai/v1/(chat/completions|models|embeddings|images/generations)$',  # OpenAI格式
            r'^/hf/v1/(chat/completions|models|embeddings|images/generations)$',  # HF格式
            r'^/vertex-express/v1beta/models/[^/:]+:(generate|streamGenerate)Content$',  # Vertex Express
            r'^/vertex-express/v1beta/models$',  # Vertex Express模型列表
        ]

        for pattern in correct_patterns:
            if re.match(pattern, path):
                return True

        return False

    def is_openai_request(self, path: str, request: Request) -> bool:
        """检测OpenAI格式请求"""
        return '/openai/' in path.lower()

    def is_hf_request(self, path: str, request: Request) -> bool:
        """检测HF格式请求"""
        return '/hf/' in path.lower()

    def is_vertex_express_request(self, path: str, request: Request) -> bool:
        """检测Vertex Express格式请求"""
        return '/vertex-express/' in path.lower()

    def fix_openai_request(self, path: str, method: str, request: Request) -> tuple:
        """修复OpenAI格式请求"""
        if method == 'POST':
            if 'chat' in path.lower() or 'completion' in path.lower():
                return '/openai/v1/chat/completions', {'type': 'openai_chat'}
            elif 'embedding' in path.lower():
                return '/openai/v1/embeddings', {'type': 'openai_embeddings'}
            elif 'image' in path.lower():
                return '/openai/v1/images/generations', {'type': 'openai_images'}
        elif method == 'GET':
            if 'model' in path.lower():
                return '/openai/v1/models', {'type': 'openai_models'}

        return path, None

    def fix_hf_request(self, path: str, method: str, request: Request) -> tuple:
        """修复HF格式请求"""
        if method == 'POST':
            if 'chat' in path.lower() or 'completion' in path.lower():
                return '/hf/v1/chat/completions', {'type': 'hf_chat'}
            elif 'embedding' in path.lower():
                return '/hf/v1/embeddings', {'type': 'hf_embeddings'}
            elif 'image' in path.lower():
                return '/hf/v1/images/generations', {'type': 'hf_images'}
        elif method == 'GET':
            if 'model' in path.lower():
                return '/hf/v1/models', {'type': 'hf_models'}

        return path, None

    def fix_vertex_express_request(self, path: str, method: str, request: Request, is_stream: bool) -> tuple:
        """修复Vertex Express请求"""
        if method != 'POST':
            if method == 'GET' and 'models' in path.lower():
                return '/vertex-express/v1beta/models', {'rule': 'vertex_express_models', 'preference': 'vertex_express_format'}
            return path, None

        # 提取模型名称
        try:
            model_name = self.extract_model_name(path, request)
        except ValueError:
            # 无法提取模型名称，返回原路径不做处理
            return path, None

        # 构建目标URL
        if is_stream:
            target_url = f'/vertex-express/v1beta/models/{model_name}:streamGenerateContent'
        else:
            target_url = f'/vertex-express/v1beta/models/{model_name}:generateContent'

        fix_info = {
            'rule': 'vertex_express_generate' if not is_stream else 'vertex_express_stream',
            'preference': 'vertex_express_format',
            'is_stream': is_stream,
            'model': model_name
        }

        return target_url, fix_info

    def fix_default_request(self, path: str, method: str, request: Request) -> tuple:
        """修复默认请求（转为最快的v1端点）"""
        if method == 'POST':
            if 'chat' in path.lower() or 'completion' in path.lower():
                return '/v1/chat/completions', {'type': 'default_chat'}
            elif 'embedding' in path.lower():
                return '/v1/embeddings', {'type': 'default_embeddings'}
            elif 'image' in path.lower():
                return '/v1/images/generations', {'type': 'default_images'}
        elif method == 'GET':
            if 'model' in path.lower():
                return '/v1/models', {'type': 'default_models'}

        return path, None

    def fix_gemini_request(self, path: str, method: str, request: Request, is_stream: bool) -> tuple:
        """修复Gemini请求"""
        if method != 'POST':
            if method == 'GET' and 'models' in path.lower():
                return '/v1beta/models', {'rule': 'gemini_models', 'preference': 'gemini_format'}
            return path, None

        # 提取模型名称
        try:
            model_name = self.extract_model_name(path, request)
        except ValueError:
            # 无法提取模型名称，返回原路径不做处理
            return path, None

        # 构建目标URL
        if is_stream:
            target_url = f'/v1beta/models/{model_name}:streamGenerateContent'
        else:
            target_url = f'/v1beta/models/{model_name}:generateContent'

        fix_info = {
            'rule': 'gemini_generate' if not is_stream else 'gemini_stream',
            'preference': 'gemini_format',
            'is_stream': is_stream,
            'model': model_name
        }

        return target_url, fix_info

    def detect_stream_request(self, path: str, request: Request) -> bool:
        """检测是否为流式请求"""
        # 1. 路径中包含stream关键词
        if 'stream' in path.lower():
            return True

        # 2. 查询参数
        if request.query_params.get('stream') == 'true':
            return True

        return False


    def is_gemini_request(self, path: str) -> bool:
        """判断是否为Gemini API请求"""
        path_lower = path.lower()

        # 如果已经是OpenAI、HF或Vertex Express格式，不应该被识别为Gemini
        if '/openai/' in path_lower or '/hf/' in path_lower or '/vertex-express/' in path_lower:
            return False

        # 1. 检查是否是明确的Gemini路径模式
        gemini_path_patterns = [
            r'/v1beta/models',  # Gemini原生API路径
            r'/gemini/v1beta',  # 带gemini前缀的路径
        ]

        for pattern in gemini_path_patterns:
            if re.search(pattern, path_lower):
                return True

        # 2. 检查是否包含Gemini模型名称
        if 'gemini' in path_lower and ('models' in path_lower or 'generatecontent' in path_lower):
            return True

        return False

    def extract_model_name(self, path: str, request: Request) -> str:
        """从请求中提取模型名称，用于构建Gemini API URL"""
        # 1. 从请求体中提取
        try:
            if hasattr(request, '_body') and request._body:
                body = json.loads(request._body.decode())
                if 'model' in body and body['model']:
                    return body['model']
        except Exception:
            pass

        # 2. 从查询参数中提取
        model_param = request.query_params.get('model')
        if model_param:
            return model_param

        # 3. 从路径中提取（用于已包含模型名称的路径）
        match = re.search(r'/models/([^/:]+)', path, re.IGNORECASE)
        if match:
            return match.group(1)

        # 4. 如果无法提取模型名称，抛出异常
        raise ValueError("Unable to extract model name from request")


