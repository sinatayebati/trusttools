from setuptools import setup, find_packages

setup(
    name='TrustTools',
    version='1.0.0',
    description='An effective and easy-to-use agentic framework with extendable tools for complex reasoning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='',
    author_email='',
    url='https://trusttools.github.io',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)