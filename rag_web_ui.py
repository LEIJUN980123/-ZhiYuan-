# rag_web_ui.py —— 混合问答系统（RAG + 通用大模型 fallback）
import os
import uuid
import gradio as gr
from datetime import datetime
from rag_langchain import LangChainRAGWithMemory, SESSION_HISTORY
from qwen_client import stream_call_qwen

document_path = os.getenv("RAG_DOCUMENT_PATH", "output/structured_docs.json")
host = os.getenv("WEB_UI_HOST", "127.0.0.1")
port = int(os.getenv("WEB_UI_PORT", 7860))

RAG_INSTANCES = {}


def get_rag(session_id: str):
    if session_id not in RAG_INSTANCES:
        RAG_INSTANCES[session_id] = LangChainRAGWithMemory(document_path)
    return RAG_INSTANCES[session_id]


def clear_history(session_id: str):
    SESSION_HISTORY.pop(session_id, None)
    RAG_INSTANCES.pop(session_id, None)
    new_sid = str(uuid.uuid4())
    return [], new_sid


def _stream_general_qwen(question: str):
    """通用问答：直接调用 Qwen，不带上下文"""
    print(f"🌍 [DEBUG] 通用问答模式 - 问题: {question}")

    # === 特殊问题预处理（工具增强）===
    lower_q = question.lower()
    if any(kw in lower_q for kw in ["时间", "几点", "现在", "当前时间"]):
        now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        yield f"当前系统时间是：{now}"
        return
    elif any(kw in lower_q for kw in ["天气", "气温", "温度", "下雨", "晴天"]):
        yield "我无法获取实时天气信息。建议您使用中国天气网、墨迹天气或手机天气应用查看最新预报。"
        return

    # === 通用大模型问答 ===
    general_prompt = f"你是一个 helpful AI 助手，请直接、简洁地回答以下问题：\n\n{question}"
    
    answer = ""
    try:
        for i, token in enumerate(stream_call_qwen(general_prompt)):
            answer += token
            if i == 0:
                print(f"✅ [通用] 首个 token: {repr(token)}")
            yield answer
        print(f"✅ [通用] 完整输出: {repr(answer)}")
    except Exception as e:
        print(f"❌ 通用问答出错: {e}")
        yield "抱歉，我暂时无法回答这个问题。"


def predict(message: str, history: list, session_id: str):
    print(f"\n🔍 [DEBUG] 收到用户消息: {message}")
    
    # === 第一步：快速拦截问候语和极短消息 ===
    stripped = message.strip()
    GREETINGS = {"你好", "hi", "hello", "hey", "谢谢", "好的", "收到", "ok", "嗯", "啊", "哈", "您好"}
    if len(stripped) < 3 or stripped in GREETINGS:
        yield "你好！请问有什么具体问题我可以帮您解答吗？"
        return

    # === 第二步：获取 RAG 实例并判断是否为知识性问题 ===
    rag = get_rag(session_id)
    if not rag._is_knowledge_question(message):  # 👈 关键：提前判断！
        print("💬 [INFO] 判定为非知识性问题，切换到通用问答模式")
        yield from _stream_general_qwen(message)
        return

    # === 第三步：构建带历史的问题（仅用于知识性问题）===
    history_text = ""
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        clean_content = content.split('【来源】')[0].strip()
        if role == "user":
            history_text += f"用户: {clean_content}\n"
        elif role == "assistant":
            history_text += f"助手: {clean_content}\n"
    
    actual_question = (
        f"【对话历史】\n{history_text}\n\n【当前问题】\n{message}"
        if history_text.strip()
        else message
    )
    print(f"📝 [DEBUG] 实际检索问题:\n{actual_question}")

    # === 第四步：执行 RAG 检索 ===
    docs = rag._retrieve_with_hyde(actual_question)
    print(f"📚 [DEBUG] 检索到 {len(docs)} 个文档片段")

    context, citations = rag.format_docs_with_citation(docs)
    print(f"📎 [DEBUG] 引用数量: {len(citations)}")

    if not docs or not context.strip():
        print("🔄 [INFO] 未检索到相关文档，切换到通用问答模式")
        yield from _stream_general_qwen(message)
        return

    # 构建 RAG prompt
    prompt_template = """你是一个严谨的 AI 助手，请严格根据以下【上下文】回答问题。
上下文可能包含：
- 普通段落
- Markdown 表格（用 | 分隔，第一行为列名）
- 标题（以 # 开头）
- 图片 OCR 结果（标记为【...OCR结果】）

请遵守：
1. 如果问题涉及表格，请从表格中提取**精确数据**
2. 如果上下文无相关信息，回答：“根据现有资料无法确定”
3. 回答应简洁

上下文：
{context}

问题：
{question}

【回答】
"""
    prompt = prompt_template.format(context=context, question=message)
    print(f"🤖 [DEBUG] 发送给模型的 prompt 长度: {len(prompt)} 字符")

    # 流式生成 RAG 答案
    answer = ""
    try:
        for i, token in enumerate(stream_call_qwen(prompt)):
            answer += token
            if i == 0:
                print(f"✅ [DEBUG] 首个 token: {repr(token)}")
            yield answer
        print(f"✅ [DEBUG] 模型完整输出: {repr(answer)}")
    except Exception as e:
        error_msg = f"❌ 模型调用出错: {str(e)}"
        print(error_msg)
        yield "抱歉，系统处理请求时发生错误。"

    # === 第五步：判断是否 fallback ===
    final_answer = answer.strip()
    if final_answer.startswith("根据现有资料无法确定"):
        print("🔄 [INFO] RAG 无法回答，切换到通用问答模式")
        yield from _stream_general_qwen(message)
    else:
        # 正常 RAG 回答，附带来源（如果有）
        if citations and len(citations) > 0:
            ref_str = "\n\n【来源】\n" + "\n".join(
                f"- {c['source']} ({c['type']} #{c['index']+1})"
                for c in citations
            )
            print(f"🔖 [DEBUG] 添加引用:\n{ref_str}")
            yield answer + ref_str
        else:
            yield answer


def respond(message: str, history: list, sid: str):
    if not message or not message.strip():
        yield history, ""
        return

    new_history = history + [{"role": "user", "content": message}]
    bot_response = ""

    for partial_answer in predict(message, new_history, sid):
        bot_response = partial_answer
        current_history = new_history + [{"role": "assistant", "content": bot_response}]
        yield current_history, ""


# ==============================
# UI (保持不变)
# ==============================
with gr.Blocks(
    title="企业知识问答系统",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="gray",
    ).set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_border_width="1px",
        block_border_color="*neutral_200",
        block_radius="8px",
        shadow_drop="0 2px 8px rgba(0,0,0,0.08)",
    ),
    css="""
    .chatbot-container { max-width: 900px; margin: 0 auto; padding: 20px; }
    .header-title { text-align: center; margin-bottom: 16px; color: #1e40af; font-weight: 700; font-size: 28px; }
    .subtitle { text-align: center; color: #64748b; margin-bottom: 24px; font-size: 16px; }
    .input-box textarea { border-radius: 12px !important; border: 1px solid #cbd5e1 !important; padding: 12px !important; font-size: 16px !important; box-shadow: inset 0 1px 2px rgba(0,0,0,0.05); }
    .clear-btn { background: #f8fafc !important; border: 1px solid #e2e8f0 !important; color: #334155 !important; border-radius: 12px !important; padding: 8px 16px !important; font-weight: 600 !important; transition: all 0.2s ease; }
    .clear-btn:hover { background: #f1f5f9 !important; transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.06); }
    footer { visibility: hidden; }
    """
) as demo:
    session_id = gr.State("")
    
    with gr.Column(elem_classes="chatbot-container"):
        gr.Markdown("## 🧠 企业知识库智能问答", elem_classes="header-title")
        gr.Markdown("基于内部文档的精准问答系统，支持表格、流程图与 OCR 内容理解。", elem_classes="subtitle")
        chatbot = gr.Chatbot(height=500, avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/4712/4712159.png"))
        
        with gr.Row():
            msg = gr.Textbox(label="", placeholder="请输入您的问题...", container=False, scale=9, elem_classes="input-box")
            clear = gr.Button("🗑️ 新建对话", variant="secondary", elem_classes="clear-btn")

    msg.submit(respond, [msg, chatbot, session_id], [chatbot, msg])
    clear.click(clear_history, [session_id], [chatbot, session_id], queue=False)
    demo.load(lambda: str(uuid.uuid4()), None, session_id)


if __name__ == "__main__":
    print("🚀 启动混合问答服务中...")
    demo.launch(server_name=host, server_port=port, share=False)