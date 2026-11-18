from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import ToolCall

dataset = EvaluationDataset(
    goldens=[
        # 1. Eligible, multiple items
        Golden(
            input="I am user123, I would like a refund for the book and the pen",
            expected_output="A refund ticket for $30 has been issued successfully.",
            expected_tools=[
                ToolCall(name="get_user_purchase_history"),
                ToolCall(name="check_current_date"),
                ToolCall(name="calculate_right_refund"),
                ToolCall(name="issue_refund_ticket"),
            ],
        ),
        # 2. Eligible, one specific item
        Golden(
            input="I am user123, I would like a refund for the book.",
            expected_output="A refund ticket for $23 has been issued successfully.",
            expected_tools=[
                ToolCall(name="get_user_purchase_history"),
                ToolCall(name="check_current_date"),
                ToolCall(name="calculate_right_refund"),
                ToolCall(name="issue_refund_ticket"),
            ],
        ),
        # 3. Information only, no ticket
        Golden(
            input="I am user123, I do not want a refund, I just want to know what I purchased.",
            expected_output="No refund ticket was issued. The customer is told they purchased a book and a pen.",
            expected_tools=[
                ToolCall(name="get_user_purchase_history"),
            ],
        ),
        # 4. Unknown customer
        Golden(
            input="I am user999, I would like a refund for the book.",
            expected_output="No customer was found with the username user999, so no refund can be issued.",
            expected_tools=[
                ToolCall(name="get_user_purchase_history"),
            ],
        ),
        # 5. Outside the 30-day window
        Golden(
            input="I am user126, I would like a refund for the water_bottle.",
            expected_output="No refund will be issued because the water_bottle was purchased more than 30 days ago.",
            expected_tools=[
                ToolCall(name="get_user_purchase_history"),
                ToolCall(name="check_current_date"),
            ],
        ),
        # 6. Mixed: one eligible, one not
        Golden(
            input="I am user126, I would like a refund for the water_bottle and the backpack.",
            expected_output="A refund ticket for $45 has been issued only for the backpack. The water_bottle is not eligible because it was purchased more than 30 days ago.",
            expected_tools=[
                ToolCall(name="get_user_purchase_history"),
                ToolCall(name="check_current_date"),
                ToolCall(name="calculate_right_refund"),
                ToolCall(name="issue_refund_ticket"),
            ],
        ),
        # 7. Item the customer never purchased
        Golden(
            input="I am user123, I would like a refund for the sneakers.",
            expected_output="No refund can be issued because the customer did not purchase the sneakers.",
            expected_tools=[
                ToolCall(name="get_user_purchase_history"),
            ],
        ),
        # 8. Missing username
        Golden(
            input="I would like a refund for the book.",
            expected_output="The agent asks the customer for their username before doing anything else.",
            expected_tools=[],
        ),
        # 9. Purchase dated in the future
        Golden(
            input="I am user133, I would like a refund for the tablet.",
            expected_output="No refund can be issued because the tablet's purchase date is in the future, so the purchase is invalid.",
            expected_tools=[
                ToolCall(name="get_user_purchase_history"),
                ToolCall(name="check_current_date"),
            ],
        ),
        # 10. 30-day boundary: planner is exactly 30 days old (eligible), marker is 31 (not)
        Golden(
            input="I am user133, I would like a refund for the planner and the marker.",
            expected_output="A refund ticket for $15 has been issued only for the planner. The marker is not eligible because it was purchased more than 30 days ago.",
            expected_tools=[
                ToolCall(name="get_user_purchase_history"),
                ToolCall(name="check_current_date"),
                ToolCall(name="calculate_right_refund"),
                ToolCall(name="issue_refund_ticket"),
            ],
        ),
    ]
)