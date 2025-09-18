# docs/gen_api_nav.py
import mkdocs_gen_files as gen
from pathlib import Path

pkg = "bamojax"
mods = []

for path in sorted(Path(pkg).rglob("*.py")):
    if path.name == "__init__.py":
        continue
    mods.append(".".join(path.with_suffix("").parts))

# Landing page for API
with gen.open("api/index.md", "w") as f:
    print("# API reference\n", file=f)
    for m in mods:
        print(f"- [`{m}`](./{m}.md)", file=f)

# One page per module
for m in mods:
    with gen.open(f"api/{m}.md", "w") as f:
        print(f"# `{m}`\n", file=f)
        print(f"::: {m}\n", file=f)
