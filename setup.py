#!/nfs/cae/Ferramentas/EXEC/PYTHON/intelpython2/bin/python

"""Semi-analytical models for plates, shells, stiffened panels, single- and
multi-domain

"""
import sys
import os
import subprocess
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Intended Audience :: Stress Analysis Engineers
License :: 3-Clause BSD Approved
Programming Language :: Python
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: Unix

"""

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


def write_version_py(filename='panels/version.py'):
    cnt = """# This file is generated automatically by the setup.py
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
svn_revision = '%(svn_revision)s'
isreleased = %(isreleased)s
if isreleased:
    __version__ = version
else:
    __version__ = full_version

if not isreleased:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'svn_revision': GIT_REVISION,
                       'isreleased': str(ISRELEASED)})
    finally:
        a.close()


def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "Unknown"

    return git_revision


def get_version_info():
    FULLVERSION = VERSION
    GIT_REVISION = ''
    if not ISRELEASED:
        GIT_REVISION = git_version()
        FULLVERSION = VERSION + 'rc' + GIT_REVISION
    return FULLVERSION, GIT_REVISION

if os.name == 'nt':
    compile_args = ['/openmp']
    link_args = []
else:
    compile_args = ['-fopenmp', '-static', '-static-libgcc', '-static-libstdc++']
    link_args = ['-fopenmp', '-static-libgcc', '-static-libstdc++']

include_dirs = [
            r'./panels/core/include',
            np.get_include(),
            ]
extensions = [
    Extension('panels.models.clpt_bardell_field',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/core/src/legendre_gauss_quadrature.cpp',
            './panels/models/clpt_bardell_field.pyx',
            ],
        include_dirs=include_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language='c++'),
    #Extension('panels.models.coneshell_clpt_donnell_bardell_num',
        #sources=[
            #'./panels/core/src/bardell_functions.cpp',
            #'./panels/core/src/legendre_gauss_quadrature.cpp',
            #'./panels/models/coneshell_clpt_donnell_bardell_num.pyx',
            #],
        #include_dirs=include_dirs,
        #extra_compile_args=compile_args,
        #extra_link_args=link_args,
        #language='c++'),
    Extension('panels.models.cylshell_clpt_donnell_bardell_num',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/core/src/legendre_gauss_quadrature.cpp',
            './panels/models/cylshell_clpt_donnell_bardell_num.pyx',
            ],
        include_dirs=include_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language='c++'),
    Extension('panels.models.plate_clpt_donnell_bardell_num',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/core/src/legendre_gauss_quadrature.cpp',
            './panels/models/plate_clpt_donnell_bardell_num.pyx',
            ],
        include_dirs=include_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language='c++'),
    ]


FULLVERSION, GIT_REVISION = get_version_info()
build_requires = []
write_version_py()
ext_modules = cythonize(extensions)
for e in ext_modules:
    e.cython_directives = {'embedsignature': True}
setup(
    name='panels',
    maintainer='Saullo G. P. Castro',
    maintainer_email='castrosaullo@gmail.com',
    version=FULLVERSION,
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    url='https://github.com/compmech/panels',
    download_url='https://github.com/compmech/panels',
    license='Copyrighted',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms=['Windows', 'Linux'],
    setup_requires=build_requires,
    install_requires=build_requires,
    ext_modules = ext_modules,
    entry_points={
        'console_scripts': [
            ],
        },
    packages=find_packages(),
)
