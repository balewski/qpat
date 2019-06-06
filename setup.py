import setuptools

setuptools.setup(
                 name='qpat',
                 version='0.0.2',
                 description='Quantum Program Analysis Tools',
                 url='https://github.com/edyounis/qpat',
                 author='Ed Younis',
                 author_email='edyounis@berkeley.edu',
                 packages=['qpat'],
                 zip_safe=False,
                 install_requires=[ 'qiskit', 'qiskit-terra' ]
                )
