"""lafa.__version__ is the single version source; every published copy must agree with it."""
import ast
import email
import re
import tomllib
from pathlib import Path
import pytest
import lafa

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    """Text of a project file; a source distribution ships only part of the repository."""
    path = ROOT/name
    if not path.is_file():
        pytest.skip(f'{name} is not part of this tree')
    return path.read_text(encoding='utf-8')


def yaml_scalars(text):
    """Dotted mapping paths to their scalar values, for the plain block YAML used here."""
    stack, found = [], {}
    for line in text.splitlines():
        match = re.fullmatch(r'( *)([A-Za-z_][\w-]*):(?:\s+(.*?))?\s*', line)
        if match is None:
            continue
        indent, key, value = len(match[1]), match[2], match[3]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        if value:
            found.setdefault('.'.join(name for _, name in stack), []).append(value.strip('\'"'))
    return found


COPIES = {
    'CITATION.cff': lambda text: yaml_scalars(text).get('version'),
    'mkdocs.yml': lambda text: yaml_scalars(text).get('extra.software.version'),
    'README.md': lambda text: re.findall(r'\*\*Version:\*\*\s*(\S+)', text),
    'docs/index.md': lambda text: re.findall(r'\*\*Version\s+(\S+)', text),
}


def test_yaml_scalars_separates_nested_and_similar_keys():
    text = 'cff-version: 1.2.0\nversion: "0.3"\nextra:\n  social:\n    - icon: x\n      version: 9\n  software:\n    version: 0.4\n'
    found = yaml_scalars(text)
    assert found['version'] == ['0.3'] and found['cff-version'] == ['1.2.0']
    assert found['extra.software.version'] == ['0.4']


@pytest.mark.parametrize('name', COPIES)
def test_published_version_copy_matches_the_package(name):
    assert COPIES[name](read(name)) == [lafa.__version__]


def test_pyproject_takes_the_version_from_the_package():
    data = tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))
    assert 'version' not in data['project'] and 'version' in data['project']['dynamic']
    assert data['tool']['setuptools']['dynamic']['version'] == {'attr': 'lafa.__version__'}


def test_version_is_a_single_string_literal():
    # setuptools reads a literal without importing lafa; anything else would need torch to build.
    tree = ast.parse((ROOT/'src'/'lafa'/'__init__.py').read_text(encoding='utf-8'))
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == '__version__' for target in node.targets)]
    assert len(values) == 1 and isinstance(values[0], ast.Constant)
    assert isinstance(values[0].value, str) and values[0].value == lafa.__version__


def test_source_distribution_metadata_matches_the_package():
    # PKG-INFO exists at the root of an extracted source distribution only.
    assert email.message_from_string(read('PKG-INFO'))['Version'] == lafa.__version__


def test_source_distribution_ships_the_thread_pinning_conftest():
    assert (ROOT/'tests'/'conftest.py').is_file()
    assert 'include tests/conftest.py' in [line.strip() for line in read('MANIFEST.in').splitlines()]
