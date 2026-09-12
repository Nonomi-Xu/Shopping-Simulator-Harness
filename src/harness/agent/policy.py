"""Model decision policy, separated from execution and runtime state ownership."""

from ..providers import ModelProvider


class ModelPolicy:
    version = "shopping-model-policy-v1"

    def __init__(self, provider: ModelProvider, tool_choice, parameters):
        self.provider = provider
        self.tool_choice = tool_choice
        self.parameters = dict(parameters or {})

    def decide(self, messages, definitions):
        return self.provider.complete(
            tuple(messages),
            tools=tuple(definitions),
            tool_choice=self.tool_choice,
            parameters=self.parameters,
        )
