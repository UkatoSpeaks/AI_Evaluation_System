from dotenv import load_dotenv

load_dotenv()  # load .env before anything that needs GROQ_API_KEY

from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from models import model
from objects import final_answer
from prompts import agent_system_prompt
from tools import (
    calculate_right_refund,
    check_current_date,
    get_user_purchase_history,
    issue_refund_ticket,
)

agent = create_agent(
    model=model,
    system_prompt=agent_system_prompt,
    tools=[get_user_purchase_history, check_current_date, calculate_right_refund, issue_refund_ticket],
    response_format=ToolStrategy(final_answer),
    checkpointer=InMemorySaver(),
)


def run_conversation():
    config = {"configurable": {"thread_id": str(uuid4())}}

    question = HumanMessage(content="I would like to request a refund")
    conversation = True

    while conversation:
        response = agent.invoke({"messages": [question]}, config=config)
        structured_response = response["structured_response"]

        print(structured_response.answer)
        conversation = structured_response.conversation

        if conversation:
            user_input = input("reply:\n")
            question = HumanMessage(content=user_input)

    print("The job is done!")


if __name__ == "__main__":
    run_conversation()