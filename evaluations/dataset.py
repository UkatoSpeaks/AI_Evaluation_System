from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import ToolCall


dataset=  EvaluationDataset(
    goldens=[
        Golden(
            input="I am user123, I would like refund for the book and the pen",
            expected_output="A refund ticket for 30$ has been issued successfully",
            expected_tools=[
                ToolCall(name="get_user_purchase_history"),
                ToolCall(name="calculate_right_refund"),
                ToolCall(name="issue_refund_ticket"),
                ToolCall(name="check_current_date")
            ]
        ),
        Golden(
            input="i am user123, I would like a refund for the book.",
            expected_output="A refund ticket of $23 has been issued successfully",
            expected_tools=[
                 ToolCall(name="get_user_purchase_history"),
                 ToolCall(name="calculate_right_refund"),
                 ToolCall(name="issue_refund_ticket"),
                 ToolCall(name="check_current_date")
            ]
        ),
         Golden(
             input="i am user123, I would like no refund, I just want to know what I purchased.",
             expected_output="No refund tickets were issued only a list of what they bought.",
             expected_tools=[
                  ToolCall(name="get_user_purchase_history"),
           ]
       ),
        Golden(
           input="i am user123, I would like a refund for the book.",
           expected_output="A refund ticket of $23 has been issued successfully",
           expected_tools=[
                 ToolCall(name="get_user_purchase_history"),
                 ToolCall(name="calculate_right_refund"),
                 ToolCall(name="issue_refund_ticket"),
                 ToolCall(name="check_current_date")
          ]
      ),
    ]
)