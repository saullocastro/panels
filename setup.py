"""Semi-analytical models for plates, shells, stiffened panels
"""
import platform
import os
import sys
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
Intended Audience :: Education
Intended Audience :: Science/Research
Intended Audience :: Developers
Intended Audience :: End Users/Desktop
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Mathematics
Topic :: Education
Topic :: Software Development
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: POSIX :: BSD
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Programming Language :: Python :: 3.12
Programming Language :: Python :: 3.13
Programming Language :: Python :: 3.14

"""

MAJOR = 0
MINOR = 9
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
YEAR = '2026'


def write_version_py(filename='panels/version.py'):
    cnt = """# This file is generated automatically by the setup.py
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
svn_revision = '%(svn_revision)s'
isreleased = %(isreleased)s
__year__ = '%(year)s'
if isreleased:
    __version__ = version
else:
    __version__ = full_version
    version = full_version

"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'svn_revision': GIT_REVISION,
                       'isreleased': str(ISRELEASED),
                       'year': YEAR,
                       })
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

# NOTE a coverage build, see .github/workflows/coverage.yml, requested with
#      CYTHON_TRACE_NOGIL in the environment or with --define CYTHON_TRACE...
trace = ('CYTHON_TRACE_NOGIL' in os.environ.keys()
         or any('CYTHON_TRACE' in arg for arg in sys.argv))

# NOTE flags for speed. OpenMP stays, since panels/models/clpt_field.pyx uses
#      prange. GCC and Clang get -O3 explicitly, because the level inherited
#      from the Python build is not guaranteed, and -fno-math-errno, which
#      lets sqrt() compile to a single instruction and changes no result.
#      MSVC is already at its fastest standard-conforming setting with the
#      /O2 and /GL that setuptools passes. Flags that change floating-point
#      results, such as /fp:fast or -ffast-math, and flags that tie a wheel to
#      the CPU that built it, such as -march=native, are deliberately left out
define_macros = []
if platform.system() == 'Windows':
    compile_args = ['/O2', '/openmp']
    link_args = []
elif platform.system() == 'Linux':
    compile_args = ['-O3', '-fno-math-errno', '-fopenmp']
    link_args = ['-fopenmp', '-static-libgcc', '-static-libstdc++']
else: # MAC-OS
    compile_args = ['-O3', '-fno-math-errno']
    link_args = []

if trace:
    # NOTE unoptimized, so that every traced line maps to code. Since Python
    #      3.12 Cython traces through sys.monitoring by default, which the
    #      Cython.Coverage plugin cannot follow, hence the legacy tracing
    if os.name == 'nt': # Windows
        compile_args = ['/Od']
    else: # MAC-OS or Linux
        compile_args = ['-O0']
    link_args = []
    define_macros = [('CYTHON_TRACE_NOGIL', '1'),
                     ('CYTHON_USE_SYS_MONITORING', '0')]

include_dirs = [
    r'./panels/core/include',
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
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
# field calculation
    Extension('panels.models.clpt_field',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/models/clpt_field.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
# shell models
    Extension('panels.models.plate_clpt_donnell',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/models/plate_clpt_donnell.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.plate_clpt_donnell_num',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/models/plate_clpt_donnell_num.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.fsdt_tsdt_field',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/models/fsdt_tsdt_field.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.plate_fsdt_donnell',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/models/plate_fsdt_donnell.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.plate_fsdt_donnell_num',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/models/plate_fsdt_donnell_num.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.plate_tsdt_donnell',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/models/plate_tsdt_donnell.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.plate_tsdt_donnell_num',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/models/plate_tsdt_donnell_num.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.cylshell_fsdt_donnell',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/models/cylshell_fsdt_donnell.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.cylshell_fsdt_donnell_num',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/models/cylshell_fsdt_donnell_num.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.cylshell_fsdt_sanders',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/models/cylshell_fsdt_sanders.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.cylshell_fsdt_sanders_num',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/models/cylshell_fsdt_sanders_num.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.cylshell_tsdt_donnell',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/models/cylshell_tsdt_donnell.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.cylshell_tsdt_donnell_num',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/models/cylshell_tsdt_donnell_num.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.cylshell_tsdt_sanders',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/models/cylshell_tsdt_sanders.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.cylshell_tsdt_sanders_num',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/models/cylshell_tsdt_sanders_num.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.cylshell_clpt_donnell',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/models/cylshell_clpt_donnell.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.cylshell_clpt_donnell_num',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/models/cylshell_clpt_donnell_num.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.cylshell_clpt_sanders',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/models/cylshell_clpt_sanders.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.models.cylshell_clpt_sanders_num',
        sources=[
            './panels/core/src/bardell_functions.cpp',
            './panels/models/cylshell_clpt_sanders_num.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
# stiffener models
    Extension('panels.stiffener.models.bladestiff1d_clt_donnell',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/stiffener/models/bladestiff1d_clt_donnell.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.stiffener.models.bladestiff2d_clt_donnell',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/stiffener/models/bladestiff2d_clt_donnell.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),

# multi-domain connections
    Extension('panels.multidomain.connections.kCBFxcte',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/multidomain/connections/kCBFxcte.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.multidomain.connections.kCBFycte',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/multidomain/connections/kCBFycte.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.multidomain.connections.kCSB',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/multidomain/connections/kCSB.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.multidomain.connections.kCSSxcte',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/multidomain/connections/kCSSxcte.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.multidomain.connections.kCSSycte',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/multidomain/connections/kCSSycte.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
# multi-domain connections
    Extension('panels.multidomain.connections.kCpd',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/multidomain/connections/kCpd.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),
    Extension('panels.multidomain.connections.kCSB_dmg',
        sources=[
            './panels/core/src/bardell.cpp',
            './panels/core/src/bardell_functions.cpp',
            './panels/multidomain/connections/kCSB_dmg.pyx',
            ],
        include_dirs=include_dirs, extra_compile_args=compile_args,
              extra_link_args=link_args, define_macros=define_macros,
              language='c++'),

    ]


FULLVERSION, GIT_REVISION = get_version_info()

write_version_py()

def generated_with_other_trace_mode(ext):
    r"""Whether the C++ file generated from a Cython source of ``ext`` was
    generated with, or without, line tracing, the opposite of this build

    cythonize regenerates a C++ file only when its source is newer, so
    switching between a coverage build and a normal one would otherwise
    reuse the C++ file of the other mode without notice.
    """
    for source in ext.sources:
        if not source.endswith('.pyx'):
            continue
        cpp = os.path.splitext(source)[0] + '.cpp'
        if os.path.isfile(cpp):
            with open(cpp, encoding='utf-8', errors='ignore') as f:
                if ('__Pyx_TraceLine(' in f.read()) != trace:
                    return True
    return False

# NOTE line tracing only for a coverage build, since the profiling hooks it
#      generates otherwise stay active in every function call
ext_modules = cythonize(extensions,
                        compiler_directives={'linetrace': trace},
                        language_level=3,
                        force=any(generated_with_other_trace_mode(ext)
                                  for ext in extensions),
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

keywords = [
            'Ritz method',
            'semi-analytical'
            'Energy-based method',
            'structural analysis',
            'structural optimization',
            'static analysis',
            'buckling',
            'vibration',
            'panel flutter',
            'structural dynamics',
            'implicit time integration',
            'explicit time integration',
            ]

setup(
    name = 'panels',
    version = FULLVERSION,
    author = "Saullo G. P. Castro, Nathan D'Souza",
    author_email = 'S.G.P.Castro@tudelft.nl',
    description = ("Semi-analytical models for plates, shells and stiffened panels"),
    long_description = read('README.md'),
    long_description_content_type = 'text/markdown',
    license = '3-Clause BSD',
    keywords = keywords,
    url = 'https://github.com/saullocastro/panels',
    package_data = package_data,
    data_files = data_files,
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
    ext_modules = ext_modules,
    include_package_data = True,
    packages = find_packages(),
)
