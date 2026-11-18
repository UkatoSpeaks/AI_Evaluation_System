from pydantic import BaseModel, Field


# The class name is kept lowercase on purpose: ToolStrategy uses it as the
# structured-output tool name, which the system prompt and the evaluation
# scripts refer to as "final_answer".
class final_answer(BaseModel):  # noqa: N801
    answer: str = Field(description="The answer to the user's question")
    conversation: bool = Field(
        description=(
            "Indicates whether the conversation should continue. "
            "Set to False once the agent has made a final decision about the refund. "
            "Set to True if more information or discussion is still needed."
        ),
    )


class PurchaseItem(BaseModel):
    name: str = Field(description="name of the item purchased")
