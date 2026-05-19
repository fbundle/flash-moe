#!/usr/bin/env python3
"""Generate shaders.h from shaders.metal — called by setup.py before each build."""

import os


def generate(src_path, dst_path):
    with open(src_path) as f:
        lines = f.readlines()

    parts = []
    parts.append("// Auto-generated from shaders.metal — do not edit.\n")
    parts.append("#ifndef SHADERS_H\n")
    parts.append("#define SHADERS_H\n")
    parts.append("\n")
    parts.append("static const char *g_shader_source =\n")

    for line in lines:
        # Escape backslashes and double-quotes, then wrap
        escaped = line.rstrip("\n").replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'"{escaped}\\n"\n')

    parts.append(";\n")
    parts.append("\n")
    parts.append("#endif // SHADERS_H\n")

    with open(dst_path, "w") as f:
        f.writelines(parts)

    print(f"[gen_shaders] {os.path.basename(dst_path)} "
          f"({len(lines)} lines)")


if __name__ == "__main__":
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    core_src = os.path.join(repo_dir, "moe_infer_mlx", "core_src")
    generate(
        os.path.join(core_src, "shaders.metal"),
        os.path.join(core_src, "shaders.h"),
    )
