from pydantic import BaseModel, Field


class final_answer(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    conversation: bool = Field(
        default=True,
        description=(
            "Indicates whether the conversation should continue. "
            "Set to False once the agent has made a final decision about the refund. "
            "Set to True if more information or discussion is still needed."
        ),
    )


class Purchase_Item(BaseModel):
    name: str = Field(description="name of the item purchased")