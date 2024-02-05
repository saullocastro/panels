"""Semi-analytical models for plates, shells, stiffened panels
"""
import platform
import os
import inspect
import subprocess
from setuptools import setup, find_packages
from setuptools.extension import Extension

from Cython.Build import cythonize


DOCLINES = __doc__.split("\n")

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    setupdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return open(os.path.join(setupdir, fname)).read()

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
Intended Audience :: Education
Intended Audience :: End Users/Desktop
Topic :: Scientific/Engineering
Topic :: Education
Topic :: Software Development
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: POSIX :: BSD
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
License :: OSI Approved :: BSD License

"""

MAJOR = 0
MINOR = 4
MICRO = 1
ISRELEASED = True
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

if platform.system() == 'Windows':
    compile_args = ['/openmp']
    link_args = []
elif platform.system() == 'Linux':
    compile_args = ['-fopenmp']
    link_args = ['-fopenmp']
else: # MAC-OS
    compile_args = []
    link_args = []

include_dirs = [
    r'./panels/core/include',
    r'C:\Users\natha\Documents\GitHub\panels\panels\core\include',
            ]

extensions = [
# Bardell functions
    Extension('panels.bardell',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/bardell.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),
# numerical integration
    Extension('panels.legendre_gauss_quadrature',
        sources=[
            './panels/core/src/legendre_gauss_quadrature.cpp',
            './panels/legendre_gauss_quadrature.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),
# field calculation
    Extension('panels.models.clpt_bardell_field',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/core/src/legendre_gauss_quadrature.cpp',
            './panels/models/clpt_bardell_field.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),
# shell models
    Extension('panels.models.plate_clpt_donnell_bardell',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/models/plate_clpt_donnell_bardell.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),
    Extension('panels.models.plate_clpt_donnell_bardell_num',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/core/src/legendre_gauss_quadrature.cpp',
            './panels/models/plate_clpt_donnell_bardell_num.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),
    Extension('panels.models.cylshell_clpt_donnell_bardell',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/models/cylshell_clpt_donnell_bardell.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),
    Extension('panels.models.cylshell_clpt_donnell_bardell_num',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/core/src/legendre_gauss_quadrature.cpp',
            './panels/models/cylshell_clpt_donnell_bardell_num.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),
# stiffener models
    Extension('panels.stiffener.models.bladestiff1d_clt_donnell_bardell',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/stiffener/models/bladestiff1d_clt_donnell_bardell.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),
    Extension('panels.stiffener.models.bladestiff2d_clt_donnell_bardell',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/stiffener/models/bladestiff2d_clt_donnell_bardell.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),

# multi-domain connections
    Extension('panels.multidomain.connections.kCBFxcte',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/multidomain/connections/kCBFxcte.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),
    Extension('panels.multidomain.connections.kCBFycte',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/multidomain/connections/kCBFycte.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),
    Extension('panels.multidomain.connections.kCSB',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/multidomain/connections/kCSB.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),
    Extension('panels.multidomain.connections.kCSSxcte',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/multidomain/connections/kCSSxcte.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),
    Extension('panels.multidomain.connections.kCSSycte',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/multidomain/connections/kCSSycte.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),
# multi-domain connections
    Extension('panels.multidomain.connections.kCpd',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/multidomain/connections/kCpd.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, language='c++'),

    ]


FULLVERSION, GIT_REVISION = get_version_info()
install_requires = [
    'numpy',
    'scipy',
    'matplotlib',
    'composites',
    'structsolve',
    'cython',
]

write_version_py()

ext_modules = cythonize(extensions,
                        compiler_directives={'linetrace': True},
                        language_level=3,
                        )

data_files = [('', [
        'README.md',
        'LICENSE',
        ])]

package_data = {
        'panels': ['*.py', '*.pxd', '*.pyx',
                   'core/include/*.hpp',
                   'core/src/*.cpp',
                   'models/*.pyx', 'models/*.pxd'
                   'tests/tests_shell/*.py'
                   ],
        }

setup(
    name = 'panels',
    version = FULLVERSION,
    author = 'Saullo G. P. Castro',
    author_email = 'S.G.P.Castro@tudelft.nl',
    description = '\n'.join(DOCLINES),
    long_description = read('README.md'),
    long_description_content_type = 'text/markdown',
    license = '3-Clause BSD',
    url = 'https://github.com/saullocastro/panels',
    package_data = package_data,
    data_files = data_files,
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
    install_requires = install_requires,
    ext_modules = ext_modules,
    packages = find_packages(),
)
