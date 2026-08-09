"""
WebSocket 聊天端点 — 对接 LangGraph Agent
"""

import asyncio
import traceback

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage

import core

router = APIRouter()


@router.websocket("/chat")
async def websocket_chat(websocket: WebSocket, session_id: str = "default"):
    """
    WebSocket 聊天端点。

    消息协议：
      客户端 → 服务端: {"type": "query", "content": "用户问题"}
      服务端 → 客户端:
        {"type": "status", "stage": "routing|retrieving|generating"}
        {"type": "answer", "content": "完整回答", "sources": [...]}
        {"type": "error", "message": "错误信息"}

    方案 A（伪流式）：用 asyncio.to_thread 跑 app.invoke()，
    完成后一次性推送结果。后续可升级为 astream_events 真流式。
    """
    await websocket.accept()

    if core.langgraph_app is None:
        await websocket.send_json({
            "type": "error",
            "message": "系统未就绪，请稍后重试"
        })
        await websocket.close()
        return

    config = {"configurable": {"thread_id": session_id}}

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type != "query":
                await websocket.send_json({
                    "type": "error",
                    "message": f"未知消息类型: {msg_type}"
                })
                continue

            content = data.get("content", "").strip()
            if not content:
                await websocket.send_json({
                    "type": "error",
                    "message": "消息内容为空"
                })
                continue

            # 阶段通知：路由中
            await websocket.send_json({"type": "status", "stage": "routing"})

            try:
                # 在独立线程中运行 LangGraph（避免阻塞事件循环）
                result = await asyncio.to_thread(
                    core.langgraph_app.invoke,
                    {
                        "query": content,
                        "messages": [HumanMessage(content=content)]
                    },
                    config=config
                )
            except Exception:
                traceback.print_exc()
                await websocket.send_json({
                    "type": "error",
                    "message": "处理请求时出错，请稍后重试"
                })
                continue

            answer = result.get("answer", "")
            # 从最终消息中提取可能的来源引用（search_knowledge_tool 的输出）
            sources = _extract_sources(answer)

            await websocket.send_json({
                "type": "answer",
                "content": answer,
                "sources": sources
            })

    except WebSocketDisconnect:
        pass


def _extract_sources(text: str) -> list[dict]:
    """
    从回答文本中提取 [^N] 格式引用 + 「参考来源」区块。
    返回: [{index, doc_name, doc_type, snippet}, ...]
    """
    import re
    sources = []

    # 1. 解析「参考来源」区块：提取文档名、类型、原文片段
    ref_block = re.search(r'\*\*参考来源[：:]\*\*\n(.*?)(?:\n\n|\Z)', text, re.DOTALL)
    if ref_block:
        ref_lines = ref_block.group(1).strip().split('\n')
        for line in ref_lines:
            m = re.match(
                r'\[\^(\d+)\]\s*文档名[：:]\s*(.+?)\s*\|\s*类型[：:]\s*(\w+)\s*\|\s*原文片段[：:]\s*["“](.+?)["”]',
                line
            )
            if m:
                sources.append({
                    "index": int(m.group(1)),
                    "doc_name": m.group(2).strip(),
                    "doc_type": m.group(3).strip(),
                    "snippet": m.group(4).strip()
                })
                continue
            # 宽松匹配
            m2 = re.match(r'\[\^(\d+)\]\s*(.+)', line)
            if m2:
                sources.append({
                    "index": int(m2.group(1)),
                    "doc_name": "",
                    "doc_type": "",
                    "snippet": m2.group(2).strip()
                })

    # 2. 如果没解析到，回退到旧格式 [来源N]
    if not sources:
        seen = set()
        for match in re.finditer(r'\[来源(\d+)\]', text):
            idx = match.group(1)
            if idx not in seen:
                seen.add(idx)
                sources.append({"index": int(idx), "doc_name": "", "doc_type": "", "snippet": ""})

    return sources
