"""Parse checked-in GitHub Actions, issue form, and AWS SAM YAML files."""

from pathlib import Path

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

ROOT = Path(__file__).resolve().parents[1]
YAML_PATHS = (
    ROOT / ".github/workflows/ci.yml",
    ROOT / ".github/workflows/security.yml",
    ROOT / ".github/ISSUE_TEMPLATE/bug_report.md",
    ROOT / ".github/ISSUE_TEMPLATE/feature_request.md",
    ROOT / "infrastructure/sam/template.yaml",
)


def construct_cloudformation_tag(loader: yaml.SafeLoader, tag: str, node: yaml.Node):
    """Treat CloudFormation intrinsics as opaque parsed YAML values."""
    if isinstance(node, ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, MappingNode):
        return loader.construct_mapping(node)
    raise TypeError(f"Unsupported YAML node for {tag}")


def load_yaml(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        parts = text.split("---", 2)
        if len(parts) != 3:
            raise ValueError(f"Missing YAML front matter in {path}")
        text = parts[1]
    loader = type("CloudFormationSafeLoader", (yaml.SafeLoader,), {})
    loader.add_multi_constructor("!", construct_cloudformation_tag)
    return yaml.load(text, Loader=loader)


def main() -> None:
    for path in YAML_PATHS:
        load_yaml(path)
        print(f"YAML syntax valid: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
