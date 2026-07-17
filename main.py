import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_core.messages import HumanMessage
from agent_graph import app

# ===================== 调用测试 =====================
if __name__ == "__main__":
    print("对话已启动，输入「退出」结束对话\n")
    # 固定会话线程ID，MemorySaver以此区分对话
    config = {"configurable": {"thread_id": "user-123"}}

    while True:
        # 接收用户输入
        user_input = input("用户：")
        # 退出判断
        if user_input.strip() == "退出":
            print("对话结束")
            break
        # 调用LangGraph执行流程（同时写入query和HumanMessage）
        res = app.invoke(
            {
                "query": user_input,
                "messages": [HumanMessage(content=user_input)]
            },
            config=config
        )
        # 打印AI回复
        print(f"AI：{res['answer']}\n")
