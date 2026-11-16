import logging
from datetime import date
from langchain.tools import tool
from data import users_data
from objects import Purchase_Item

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


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


@tool
def calculate_right_refund(user: str, items: list[Purchase_Item]) -> dict:
    """
    Calculate the refund amount for specific items a customer wants to return.
    Takes the customer's username and the list of items to refund.
    Only call this after confirming the items are eligible for a refund.
    """
    logging.info(f"calculate_right_refund called for user: {user}")

    if user not in users_data:
        return {"error": f"No customer found with username '{user}'"}

    total_refund = 0
    not_found = []

    for item in items:
        if item.name in users_data[user]:
            total_refund += users_data[user][item.name]["price"]
        else:
            not_found.append(item.name)

    result = {"refund_amount": total_refund}
    if not_found:
        result["items_not_purchased"] = not_found
    return result


@tool
def issue_refund_ticket(user: str, refund_amount: float) -> str:
    """
    Issue a refund ticket for a customer with the given refund amount.
    Only call this when the customer has explicitly asked for a refund AND
    the items are eligible. Never call it for information-only questions.
    """
    logging.info(f"issue_refund_ticket called for user: {user}, amount: {refund_amount}")
    return f"A refund ticket has been issued successfully for the amount of {refund_amount}"


@tool
def check_current_date() -> str:
    """Get today's date in YYYY-MM-DD format. Use this before checking the 30-day refund window."""
    logging.info("check_current_date called")
    return "2026-10-28"  # fixed date so evals are reproducible


if __name__ == "__main__":
    # 1. Single item refund → expect 23
    print(calculate_right_refund.invoke({
        "user": "user123",
        "items": [{"name": "book"}]
    }))

    # 2. Multiple items → expect 23 + 7 = 30
    print(calculate_right_refund.invoke({
        "user": "user123",
        "items": [{"name": "book"}, {"name": "pen"}]
    }))

    # 3. Item never purchased → expect 0 + items_not_purchased: ["laptop"]
    print(calculate_right_refund.invoke({
        "user": "user123",
        "items": [{"name": "laptop"}]
    }))

    # 4. Mixed: one real, one fake → expect 45 + items_not_purchased: ["laptop"]
    print(calculate_right_refund.invoke({
        "user": "user126",
        "items": [{"name": "backpack"}, {"name": "laptop"}]
    }))

    # 5. Unknown user → expect error
    print(calculate_right_refund.invoke({
        "user": "user999",
        "items": [{"name": "book"}]
    }))