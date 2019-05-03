python setup.py build_ext --inplace
python setup.py install
coverage run --source panels -m py.test
coverage report
