"""Generate OpenAPI docs as markdown."""

import sys
from pathlib import Path

# Add backend to path so we can import app
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "apps" / "backend" / "src")
)


def generate_markdown(schema: dict) -> str:
    """Build markdown documentation from an OpenAPI schema dictionary.

    Args:
        schema: OpenAPI schema as returned by ``app.openapi()``.

    Returns:
        A markdown string containing endpoint, parameter, and response sections.
    """
    # The note is emitted here rather than hand-added to the output file:
    # this script overwrites that file on every docs build, so anything
    # written into it by hand is silently lost on the next run.
    lines = [
        "# Backend API Reference",
        "",
        "This is an OpenAPI-derived reference for the backend API.",
        "",
        ":::{note}",
        "This page is generated from the live FastAPI app by",
        "`scripts/generate_openapi_docs.py`, which the `docs.yml` workflow runs",
        "before Sphinx on every build, so the published site always reflects the",
        "running app. Do not edit it by hand — regeneration discards the edit.",
        "",
        "Descriptions are the first paragraph of each route's docstring. The full",
        "docstrings, including `Args`, `Returns` and `Raises`, are in the",
        "autoapi reference.",
        ":::",
        "",
    ]
    paths = schema.get("paths", {})
    for path, methods in paths.items():
        for method, operation in methods.items():
            lines.append(f"## {method.upper()} `{path}`")
            lines.append("")
            if "summary" in operation:
                lines.append(f"**Summary:** {operation['summary']}  ")
            if "description" in operation:
                desc = operation["description"].strip()
                # Take just the first paragraph/line to avoid Args/Returns
                # blocks rendering as code.
                if "\n\n" in desc:
                    summary = desc.split("\n\n")[0]
                else:
                    summary = desc.split("\n")[0]
                lines.append(f"**Description:** {summary}  ")
            lines.append("")

            if "parameters" in operation:
                lines.append("### Parameters")
                for param in operation["parameters"]:
                    req = "*(Required)*" if param.get("required") else "*(Optional)*"
                    lines.append(
                        f"- **`{param['name']}`** {req} ({param['in']}): "
                        f"{param.get('description', '')}"
                    )
                lines.append("")

            if "responses" in operation:
                lines.append("### Responses")
                for status, resp in operation["responses"].items():
                    lines.append(f"- **{status}**: {resp.get('description', '')}")
                lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Import after path setup above so local backend package resolution works.
    from api.main import app

    schema = app.openapi()
    md = generate_markdown(schema)
    out_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "source"
        / "reference"
        / "backend-api.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Generated {out_path}")
