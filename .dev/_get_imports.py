# Re-import after code reset
import importlib.metadata
import os
import ast
import sys
import re
import pandas as pd
import pathlib
from importlib.metadata import distributions
import tomllib


def extract_third_party_imports(root_dir: str) -> pd.DataFrame:
    """
    Walk codebase and collect third-party root import names.
    """
    stdlib = (
        set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()
    )
    rows = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            full_path = os.path.join(dirpath, filename)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=full_path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            name = alias.name.split(".")[0]
                            if name not in stdlib:
                                rows.append((full_path, name))
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            name = node.module.split(".")[0]
                            if name not in stdlib:
                                rows.append((full_path, name))
            except (SyntaxError, UnicodeDecodeError):
                continue

    return pd.DataFrame(rows, columns=["filename", "root_package"]).drop_duplicates()


def _extract_pkg_name(dep: str) -> str:
    return re.split(r"[<>=~! ]", dep, 1)[0].strip().lower()


def _get_import_names(declared: set[str]) -> set[str]:
    """
    Given a set of declared package names, return the set of all import names
    associated with those packages based on installed distributions.
    """
    dist_map = importlib.metadata.packages_distributions()
    import_names = set()

    for dep in declared:
        dep_matches = {k for k, v in dist_map.items() if dep in v}
        if dep_matches:
            import_names.update(dep_matches)
        else:
            import_names.add(dep.lower().replace("-", "_"))

    return import_names


def find_missing_dependencies(project_path: os.PathLike) -> set[str]:
    """
    Compare third-party imports with declared dependencies in pyproject.toml.

    Parameters
    ----------
    project_path : str
        Base path of the Python project

    Returns
    -------
    pd.DataFrame
        Subset of `imports_df` where the root_package is used but not declared
        in pyproject.toml.
    """
    project_path = pathlib.Path(project_path)
    with open(project_path / "pyproject.toml", "rb") as f:
        toml = tomllib.load(f)

    declared = {_extract_pkg_name(d) for d in toml["project"]["dependencies"]}
    optional = toml["project"].get("optional-dependencies", {})
    for group in optional.values():
        declared.update(_extract_pkg_name(d) for d in group)
    project_name = toml["project"]["name"].replace("-", "_").lower()
    declared.add(project_name)
    declared = _get_import_names(declared)

    imports = extract_third_party_imports(project_path / "src")

    return imports[~imports["root_package"].isin(declared)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find undeclared third-party imports.")
    parser.add_argument(
        "--base",
        type=str,
        help="Path to the base Python package or source root.",
        default="./",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="List of package names to exclude from the check.",
    )

    args = parser.parse_args()
    missing_df = find_missing_dependencies(pathlib.Path(args.base))

    if args.exclude:
        missing_df = missing_df[~missing_df["root_package"].isin(args.exclude)]

    if missing_df.empty:
        print("No missing dependencies found.")
    else:
        print("Missing dependencies:")
        print(missing_df.sort_values("root_package").to_string(index=False))
