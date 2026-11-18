from dotenv import load_dotenv

load_dotenv()  # load .env before anything that needs GROQ_API_KEY

from uuid import uuid4

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from models import model
from prompts import agent_system_prompt
from tools import (
    calculate_right_refund,
    check_current_date,
    get_user_purchase_history,
    issue_refund_ticket,
)

# The agent finishes its turn with a plain-text reply. A structured-output
# tool (ToolStrategy) was dropped: the Groq models tried kept failing to call
# it, and Groq does not allow JSON-schema output together with tool calling.
agent = create_agent(
    model=model,
    system_prompt=agent_system_prompt,
    tools=[get_user_purchase_history, check_current_date, calculate_right_refund, issue_refund_ticket],
    checkpointer=InMemorySaver(),
)


def final_reply(response: dict) -> str | None:
    """The agent's reply to the customer, or None if its turn ended without one."""
    last = response["messages"][-1]
    if last.type == "ai" and not last.tool_calls and last.text:
        return last.text
    return None


def run_conversation():
    config = {"configurable": {"thread_id": str(uuid4())}}
    print("Chat with the refund agent. Press Enter on an empty line (or type 'exit') to quit.\n")

    while True:
        user_input = input("you: ").strip()
        if user_input.lower() in ("", "exit", "quit"):
            break
        response = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        print(f"agent: {final_reply(response) or '(no reply, please try again)'}\n")

    print("The job is done!")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_conversation()
