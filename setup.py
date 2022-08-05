from setuptools import setup, find_packages
from flowvae import __name__, __version__

# with open('README.md') as f:
#       long_description = f.read()

setup(name=__name__,
      version=__version__,
      description='vae for flowfield reconstruction and prediction',
      keywords=['CFD', 'machine learning'],
      # download_url='https://github.com/swayli94/cfdpost/',
      license='MIT',
      author='Aerolab',
      author_email='yyj980401@126.com',
      packages=find_packages(),
      install_requires=['numpy', 'torch'],
      classifiers=[
            'Programming Language :: Python :: 3'
      ]
)

