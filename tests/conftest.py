import os
import sys

# NOTE in Pyodide (WebAssembly), see .github/workflows/pyodide.yml, the tests
#      run in Node.js, without the browser of the default backend of
#      matplotlib; MPLBACKEND is also set there, this is a fallback
if sys.platform == 'emscripten':
    os.environ['MPLBACKEND'] = 'Agg'
