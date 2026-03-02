# embedding_client.py —— 精简版（仅 sentence-transformers）
import os
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

# ✅ 强制只用 sentence-transformers（兼容 bge-small-zh-v1.5）
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    raise ImportError(
        "请安装 sentence-transformers:\n"
        "pip install sentence-transformers"
    )

class LocalEmbeddingClient:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = None):
        self.model_name = model_name
        
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        logger.info(f"🌍 使用设备: {device}")
        
        # ✅ 只加载 SentenceTransformer
        logger.info(f"🚀 加载 SentenceTransformer 模型: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.backend = "sentence_transformers"
        
        logger.info("✅ 本地 Embedding 模型加载完成")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 25) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"❌ Embedding 生成失败: {e}")
            raise

    @property
    def dim(self) -> int:
        sample = self.encode(["测试"], batch_size=1)
        return len(sample[0]) if sample else 0


_EMBEDDING_CLIENT = None

def get_embedding_client(model_name: str = "BAAI/bge-small-zh-v1.5") -> LocalEmbeddingClient:
    global _EMBEDDING_CLIENT
    if _EMBEDDING_CLIENT is None:
        _EMBEDDING_CLIENT = LocalEmbeddingClient(model_name)
    return _EMBEDDING_CLIENT

def compute_embeddings(texts: List[str], batch_size: int = 25) -> List[List[float]]:
    client = get_embedding_client()
    return client.encode(texts, batch_size=batch_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = LocalEmbeddingClient("BAAI/bge-small-zh-v1.5")
    test_texts = ["你好", "采购流程是什么？"]
    embeddings = client.encode(test_texts)
    print(f"✅ 向量维度: {len(embeddings[0])}")