import os
import re
import io
import setuptools

ROOT_DIR = os.path.dirname(__file__)


def parse_readme(readme: str) -> str:
    """Parse the README.md file to be pypi compatible."""
    # Replace the footnotes.
    readme = readme.replace('<!-- Footnote -->', '#')
    footnote_re = re.compile(r'\[\^([0-9]+)\]')
    readme = footnote_re.sub(r'<sup>[\1]</sup>', readme)

    # Remove the dark mode switcher
    mode_re = re.compile(
        r'<picture>[\n ]*<source media=.*>[\n ]*<img(.*)>[\n ]*</picture>',
        re.MULTILINE)
    readme = mode_re.sub(r'<img\1>', readme)
    return readme


long_description = ''
readme_filepath='README.md'
if os.path.exists(readme_filepath):
    long_description = io.open(readme_filepath, 'r', encoding='utf-8').read()
    long_description = parse_readme(long_description)

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name='internmanip',
    version='0.1.0',
    packages=setuptools.find_packages(),
    author='OpenRobotLab',
    author_email='OpenRobotLab@pjlab.org.cn',
    license='Apache 2.0',
    readme='README.md',
    description='InternManip: A comprehensive framework with trainer and evaluator for embodied manipulation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    include_package_data=True,
)
