from setuptools import setup, find_packages

setup(name='prosecco_nmr',
      version='2.0',
      description='Sequence based NMR chemical shift prediction for proteins',
      long_description='PROSECCO: PROtein SEquence and Chemical shift COrrelations.',
      keywords='NMR chemical shift protein chemistry',
      url='https://github.com/maximosanz/Prosecco-NMR',
      author='Maximo Sanz-Hernandez',
      author_email='maximo.sanzh@gmail.com',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.17.3',
      ],
      include_package_data=True,
      zip_safe=False)