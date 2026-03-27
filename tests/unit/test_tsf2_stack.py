import aws_cdk as core
import aws_cdk.assertions as assertions

from tsf2.tsf2_stack import Tsf2Stack

# example tests. To run these tests, uncomment this file along with the example
# resource in tsf2/tsf2_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = Tsf2Stack(app, "tsf2")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
