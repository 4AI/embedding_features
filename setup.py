import io
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

__version__ = None
exec(open('embedding_features/version.py').read())

short_description = 'extract embedding features from text'

# Get the long description from the README file
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    'numpy >= 1.14',
    'gensim >= 3.7.0',
    'scikit-learn >= 0.20.0'
]

extras_requires = {
    'tests': [
        'pytest-cov >= 2.4.0',
        'flake8 >= 3.6.0'],
}


setup(
    name="embedding_features",
    version=__version__,
    author="SeanLee@4AI",
    author_email="xmlee97@gmail.com",
    description=(short_description),
    license="MIT",
    keywords="embedding features, word2vec, fasttext",
    url="https://github.com/4AI/embedding_features",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.6'
    ],
    install_requires=install_requires,
    extras_require=extras_requires,
    package_data={'': ['*.md', '*.txt']},
    include_package_data=True,
)
