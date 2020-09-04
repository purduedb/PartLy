from setuptools import setup, find_packages

setup(name='PartLy',
      version='0.0.1',
      description='DRL Data Partitioner',
      url='https://github.com/purduedb/PartLy',
      author='Ahmed S. Abdelhamid',
      author_email='samy@purdue.edu',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
