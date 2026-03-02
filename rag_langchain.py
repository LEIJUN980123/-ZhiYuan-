# rag_langchain.py —— 本地免费 RAG（基于 BGE + 完全离线，兼容 Python 3.12）
import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 自定义模块
from embedding_client import compute_embeddings  # 👈 使用标准兼容接口
from qwen_client import call_qwen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# 全局会话历史（用于多轮对话）
# ==============================
SESSION_HISTORY: Dict[str, List[Dict]] = defaultdict(list)

# ==============================
# Re-Ranker 初始化（可选）—— 暂时禁用以兼容 Python 3.12
# ==============================
HAS_RERANKER = False
RERANKER = None
logger.info("ℹ️ Re-Ranker 已禁用（如需启用，请使用 Python ≤3.11 并取消上方注释）")

# ==============================
# 知识性问题模板（用于语义判断）
# ==============================
_KNOWLEDGE_TEMPLATES = [
    "这个怎么操作？",
    "流程是什么？",
    "如何配置？",
    "步骤有哪些？",
    "有什么要求？",
    "包含哪些内容？",
    "具体说明一下",
    "详细解释",
    "根据文档回答",
    "请说明...",
    "怎么做？",
    "是什么意思？",
    "如何安装？",
    "怎么使用？",
    "规则是什么？",
    "标准是什么？",
    "需要什么条件？"
]

_SIMILARITY_THRESHOLD = 0.45     # 语义判断阈值
_DISTANCE_THRESHOLD = 1.2        # Chroma 距离过滤阈值


# ==============================
# 本地 Embedding 封装（兼容 LangChain）
# ==============================
class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        valid_texts = [str(t) for t in texts if t is not None and str(t).strip()]
        if not valid_texts:
            return [[]] * len(texts)
        embeddings = compute_embeddings(valid_texts)
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class LangChainRAGConfig:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        enable_hyde: bool = True,
        top_k: int = 4,
        max_context_chars: int = 2500,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_hyde = enable_hyde
        self.top_k = top_k
        self.max_context_chars = max_context_chars


class LangChainRAGWithMemory:
    def __init__(self, document_path: str, config: Optional[LangChainRAGConfig] = None):
        self.document_paths = [Path(p.strip()) for p in document_path.split(",") if p.strip()]
        self.config = config or LangChainRAGConfig()
        
        doc_hash = "_".join([p.stem for p in sorted(self.document_paths)])
        persist_dir = f"./chroma_db_lc/multi_{doc_hash}/bge_small_zh_v1_5"
        
        logger.info(f"📁 持久化目录: {persist_dir}")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        embedding_model = LocalEmbeddings()

        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )

        if self.vectorstore._collection.count() == 0:
            logger.info("🔄 向量库为空，开始构建...")
            docs = self._load_and_split_documents()
            if docs:
                batch_size = 100
                for i in range(0, len(docs), batch_size):
                    batch = docs[i:i + batch_size]
                    self.vectorstore.add_documents(batch)
                logger.info(f"✅ 向量库构建完成，共 {len(docs)} 个文本块")
            else:
                logger.warning("⚠️ 无有效文档可加载")

    def _load_and_split_documents(self) -> List[Document]:
        all_docs = []
        logger.info(f"📂 开始加载 {len(self.document_paths)} 个文档文件")

        for doc_path in self.document_paths:
            logger.info(f"📄 处理文件: {doc_path}")
            if not doc_path.exists():
                logger.warning(f"⚠️ 文件不存在，跳过: {doc_path}")
                continue

            with open(doc_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else [data]

            for item in items:
                filename = item.get("filename", "unknown")
                file_type = item.get("file_type", "unknown")
                source_path = item.get("source_path", str(doc_path))
                sections = item.get("sections", [])

                if not sections:
                    continue

                paragraphs = []
                for sec in sections:
                    sec_type = sec.get("type")
                    if sec_type == "paragraph":
                        text = sec.get("text", "").strip()
                        if text:
                            paragraphs.append(text)
                    elif sec_type == "heading":
                        text = sec.get("text", "").strip()
                        if text:
                            level = sec.get("level", 2)
                            paragraphs.append(f"{'#' * level} {text}")
                    elif sec_type == "table":
                        rows = sec.get("rows", [])
                        if rows:
                            caption = sec.get("caption", "表格")
                            header = rows[0]
                            table_md = f"【{caption}】\n"
                            table_md += "| " + " | ".join(str(cell) for cell in header) + " |\n"
                            table_md += "|-" + "-|-".join([""] * len(header)) + "-|\n"
                            for row in rows[1:]:
                                table_md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
                            paragraphs.append(table_md.strip())
                    elif sec_type == "image_ocr":
                        ocr_text = sec.get("text", "").strip()
                        caption = sec.get("caption", "图片")
                        if ocr_text:
                            paragraphs.append(f"【{caption} OCR结果】\n{ocr_text}")
                    elif sec_type == "slide":
                        slide_content = sec.get("content", [])
                        slide_texts = []
                        for sub_sec in slide_content:
                            sub_type = sub_sec.get("type")
                            if sub_type == "paragraph":
                                t = sub_sec.get("text", "").strip()
                                if t:
                                    slide_texts.append(t)
                            elif sub_type == "image_ocr":
                                ocr_t = sub_sec.get("text", "").strip()
                                cap = sub_sec.get("caption", "图片")
                                if ocr_t:
                                    slide_texts.append(f"【{cap} OCR结果】\n{ocr_t}")
                        if slide_texts:
                            paragraphs.append("\n".join(slide_texts))
                    else:
                        text = sec.get("text", "").strip()
                        if text:
                            paragraphs.append(text)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
                    length_function=len
                )

                for idx, para in enumerate(paragraphs):
                    metadata = {
                        "source_file": filename,
                        "file_type": file_type,
                        "original_index": idx,
                        "source_path": source_path,
                        "section_type": "mixed"
                    }
                    chunks = splitter.create_documents([para], metadatas=[metadata])
                    all_docs.extend(chunks)

        return all_docs

    def _rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        if not HAS_RERANKER or len(docs) <= 1:
            return docs
        try:
            pairs = [[query, d.page_content] for d in docs]
            scores = RERANKER.compute_score(pairs)
            if not isinstance(scores, list):
                scores = [scores]
            scored = list(zip(docs, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [d for d, s in scored]
        except Exception as e:
            logger.error(f"Re-Ranker 执行失败: {e}")
            return docs

    def _hyde_generate(self, question: str) -> str:
        hyde_prompt = f"""你是一个领域专家，请基于你的知识生成一段详细、具体、包含关键实体的假设答案。
即使不确定，也要生成合理内容用于检索。

问题：{question}

假设答案：
"""
        return call_qwen(hyde_prompt, model="qwen-turbo")

    def _is_knowledge_question(self, query: str) -> bool:
        """智能判断是否为知识性问题（需检索文档）"""
        query = query.strip()
        if len(query) < 3:
            return False

        try:
            query_emb = compute_embeddings([query])[0]
            template_embs = compute_embeddings(_KNOWLEDGE_TEMPLATES)
            
            query_norm = query_emb / np.linalg.norm(query_emb)
            template_norms = template_embs / np.linalg.norm(template_embs, axis=1, keepdims=True)
            similarities = np.dot(template_norms, query_norm)
            
            max_sim = float(np.max(similarities))
            logger.debug(f"🔍 问题 '{query}' 最大相似度: {max_sim:.3f}")
            
            return max_sim >= _SIMILARITY_THRESHOLD
        except Exception as e:
            logger.warning(f"⚠️ 相似度判断失败，回退到简单规则: {e}")
            trivial_patterns = ["你好", "谢谢", "聪明吗", "你是谁", "在吗", "ok", "嗯", "哈", "厉害吗", "叫什么"]
            return not any(p in query for p in trivial_patterns)

    def _retrieve_with_hyde(self, question: str) -> List[Document]:
        start_time = time.time()
        query = self._hyde_generate(question) if self.config.enable_hyde else question
        
        results = self.vectorstore.similarity_search_with_score(query, k=self.config.top_k)
        docs = []
        for doc, dist in results:
            if dist <= _DISTANCE_THRESHOLD:
                docs.append(doc)
            else:
                logger.debug(f"🔍 跳过低相关片段 (dist={dist:.3f})")

        docs = self._rerank_docs(question, docs)
        logger.debug(f"🔍 检索+重排耗时: {time.time() - start_time:.2f}s, 得到 {len(docs)} 个片段")
        return docs

    def qa_chain(self, question: str, session_id: str = "default") -> Tuple[str, List[Dict]]:
        """仅用于知识性问题"""
        docs = self._retrieve_with_hyde(question)
        
        if not docs:
            fallback = "根据现有资料无法确定"
            logger.info("ℹ️ 无检索结果，使用兜底回答")
            return fallback, []

        context, citations = self.format_docs_with_citation(docs)
        
        if not context.strip():
            return "根据现有资料无法确定", []

        prompt_template = """你是一个严谨的 AI 助手，请严格根据以下【上下文】回答问题。
上下文可能包含：
- 普通段落
- Markdown 表格（用 | 分隔，第一行为列名）
- 标题（以 # 开头）
- 图片 OCR 结果（标记为【...OCR结果】）

请遵守：
1. 如果问题涉及表格，请从表格中提取**精确数据**
2. 如果上下文无相关信息，回答：“根据现有资料无法确定”
3. 回答应简洁，不要解释推理过程

上下文：
{context}

问题：
{question}

【回答】
"""
        prompt = prompt_template.format(context=context, question=question)

        answer = call_qwen(prompt, model="qwen-turbo")
        if not answer or "未知错误" in answer:
            answer = "生成答案时出错，请稍后重试。"

        return answer, citations

    @staticmethod
    def format_docs_with_citation(docs: List[Document]) -> Tuple[str, List[Dict]]:
        context_parts = []
        citations = []
        total_len = 0
        max_chars = 2500

        for i, doc in enumerate(docs):
            text = doc.page_content.strip()
            if not text:
                continue
            if text.startswith("#") and len(text.split()) < 3:
                continue
            if total_len + len(text) > max_chars:
                break
            context_parts.append(text)
            meta = doc.metadata
            citations.append({
                "source": meta.get("source_file", "unknown"),
                "type": meta.get("file_type", "text"),
                "index": i,
                "path": meta.get("source_path", "")
            })
            total_len += len(text)

        context = "\n\n".join(context_parts)
        return context, citations

    def ask(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """
        主入口：先判断问题类型
        """
        if not self._is_knowledge_question(question):
            # === 非知识性问题：直接回答，不检索，不显示来源 ===
            answer = "是的，我是一个聪明的 AI 助手。"
            return {
                "question": question,
                "answer": answer,
                "retrieved_chunks": []
            }

        # === 知识性问题：正常检索 + 来源 ===
        answer, citations = self.qa_chain(question, session_id)
        
        if citations:
            ref_str = "\n\n【来源】\n" + "\n".join(
                f"- {c['source']} ({c['type']} #{c['index']+1})"
                for c in citations
            )
            answer += ref_str

        SESSION_HISTORY[session_id].append({"role": "user", "content": question})
        SESSION_HISTORY[session_id].append({"role": "assistant", "content": answer})

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": []
        }