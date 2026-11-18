import logging
from datetime import date

from langchain.tools import tool

from data import users_data
from objects import PurchaseItem

logger = logging.getLogger(__name__)

TODAY = date(2026, 10, 28)  # fixed date so evals are reproducible
REFUND_WINDOW_DAYS = 30


def _ineligibility_reason(purchase_date: str) -> str | None:
    """Return None if a purchase is refundable today, otherwise the reason it is not."""
    days_ago = (TODAY - date.fromisoformat(purchase_date)).days
    if days_ago < 0:
        return f"the purchase date {purchase_date} is in the future, so the purchase is invalid"
    if days_ago > REFUND_WINDOW_DAYS:
        return (
            f"it was purchased {days_ago} days ago, outside the "
            f"{REFUND_WINDOW_DAYS}-day refund window"
        )
    return None


@tool
def get_user_purchase_history(user: str) -> dict:
    """
    Get the purchase history of a customer by their username.
    Use this first to identify the customer before checking refund eligibility.
    Returns the customer's purchases, or an error if the username does not exist.
    """
    logger.info("get_user_purchase_history called for user: %s", user)

    if user in users_data:
        return users_data[user]
    return {"error": f"No customer found with username '{user}'"}


@tool
def calculate_right_refund(user: str, items: list[PurchaseItem]) -> dict:
    """
    Calculate the refund amount for specific items a customer wants to return.
    Takes the customer's username and the list of items to refund.
    Enforces the refund policy: only items the customer purchased, within the
    last 30 days and not dated in the future, are counted. Returns the amount
    plus any items that were not purchased or are not eligible, with the reason.
    """
    logger.info("calculate_right_refund called for user: %s", user)

    if user not in users_data:
        return {"error": f"No customer found with username '{user}'"}

    purchases = users_data[user]
    total_refund = 0
    refunded_items = []
    ineligible_items = {}
    not_found = []

    for item in items:
        purchase = purchases.get(item.name)
        if purchase is None:
            not_found.append(item.name)
            continue
        reason = _ineligibility_reason(purchase["date"])
        if reason:
            ineligible_items[item.name] = reason
            continue
        total_refund += purchase["price"]
        refunded_items.append(item.name)

    result = {"refund_amount": total_refund, "refunded_items": refunded_items}
    if ineligible_items:
        result["ineligible_items"] = ineligible_items
    if not_found:
        result["items_not_purchased"] = not_found
    return result


@tool
def issue_refund_ticket(user: str, refund_amount: float) -> str:
    """
    Issue a refund ticket for a customer with the given refund amount.
    Only call this when the customer has explicitly asked for a refund AND
    calculate_right_refund returned a refund_amount greater than 0.
    Never call it for information-only questions.
    """
    logger.info("issue_refund_ticket called for user: %s, amount: %s", user, refund_amount)

    if user not in users_data:
        return f"Error: no customer found with username '{user}', no ticket issued"
    if refund_amount <= 0:
        return "Error: the refund amount must be greater than 0, no ticket issued"
    return f"A refund ticket has been issued successfully for the amount of {refund_amount}"


@tool
def check_current_date() -> str:
    """Get today's date in YYYY-MM-DD format. Use this before checking the 30-day refund window."""
    logger.info("check_current_date called")
    return TODAY.isoformat()


if __name__ == "__main__":
    # 1. Single eligible item -> 23
    print(calculate_right_refund.invoke({"user": "user123", "items": [{"name": "book"}]}))

    # 2. Multiple eligible items -> 23 + 7 = 30
    print(calculate_right_refund.invoke(
        {"user": "user123", "items": [{"name": "book"}, {"name": "pen"}]}
    ))

    # 3. Item never purchased -> 0 + items_not_purchased: ["laptop"]
    print(calculate_right_refund.invoke({"user": "user123", "items": [{"name": "laptop"}]}))

    # 4. Mixed: eligible + outside window + never purchased -> 45
    print(calculate_right_refund.invoke({
        "user": "user126",
        "items": [{"name": "backpack"}, {"name": "water_bottle"}, {"name": "laptop"}],
    }))

    # 5. Boundary + future: planner (30 days) -> 15, marker (31 days) and tablet (future) ineligible
    print(calculate_right_refund.invoke({
        "user": "user133",
        "items": [{"name": "planner"}, {"name": "marker"}, {"name": "tablet"}],
    }))

    # 6. Unknown user -> error
    print(calculate_right_refund.invoke({"user": "user999", "items": [{"name": "book"}]}))

    # 7. Zero-amount ticket -> refused
    print(issue_refund_ticket.invoke({"user": "user123", "refund_amount": 0}))
