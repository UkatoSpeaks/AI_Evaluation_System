import logging
from langchain.tools import tool
from data import users_data

logging.basicConfig(level=logging.INFO)


@tool
def get_user_purchase_history(user: str) -> dict:
    """
    Get the purchase history of a customer by their username.
    Use this first to identify the customer before checking refund eligibility.
    Returns the customer's purchases, or an error if the username does not exist.
    """
    logging.info(f"get_user_purchase_history called for user: {user}")

    if user in users_data:
        return users_data[user]
    return {"error": f"No customer found with username '{user}'"}