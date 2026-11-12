# Strands Agents

Install the optional extra with `python -m pip install -e ".[strands]"`, then run `python examples/strands-agent/main.py`. The example gives a custom Python function to `Agent` as a tool decorated with Strands `@tool`. The agent may decide to invoke it based on the prompt; model requests use Bedrock and may incur charges. See the [official custom tools guide](https://strandsagents.com/docs/user-guide/concepts/tools/python-tools/) for current SDK details.
