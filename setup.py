from setuptools import setup

setup(name='archs',
      version='0.1',
      description='Neural Network Architectures',
      url='https://gitlab.idiap.ch/whe/archs',
      author='Weipeng He',
      author_email='weipeng.he@idiap.ch',
      license='BSDv3',
      packages=['archs'],
      install_requires=['numpy', 'torch'],
      include_package_data=False,
      zip_safe=False)

