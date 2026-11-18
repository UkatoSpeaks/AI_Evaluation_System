from pydantic import BaseModel, Field


class PurchaseItem(BaseModel):
    name: str = Field(description="name of the item purchased")
