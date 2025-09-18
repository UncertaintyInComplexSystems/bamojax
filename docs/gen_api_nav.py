import mkdocs_gen_files as gen
from pathlib import Path

pkg = "bamojax"
modules = []

for path in sorted(Path(pkg).rglob("*.py")):
    if path.name == "__init__.py":
        continue
    mod = ".".join(path.with_suffix("").parts)
    modules.append(mod)

with gen.open("api/index.md", "w") as f:
    print("# API reference\n", file=f)
    for mod in modules:
        print(f"- [{mod}](./{mod}.md)", file=f)

for mod in modules:
    out_path = Path("api", f"{mod}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gen.open(out_path, "w") as f:
        print(f"# `{mod}`\n", file=f)
        print(f"::: {mod}\n", file=f)
