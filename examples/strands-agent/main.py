"""Custom-tool Strands example; model requests may incur Bedrock charges."""

from strands import Agent, tool
from strands.models import BedrockModel


@tool
def character_count(text: str) -> int:
    """Count characters in text.

    Args:
        text: Text to count.
    """
    return len(text)


def main() -> None:
    model = BedrockModel(model_id="amazon.nova-micro-v1:0")
    agent = Agent(model=model, tools=[character_count])
    print(agent("Use the character_count tool to count the characters in 'AWS AI'."))


if __name__ == "__main__":
    main()
