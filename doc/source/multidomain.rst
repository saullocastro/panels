Multi-domain assemblies (:mod:`panels.multidomain`)
===================================================

.. autoclass:: panels.multidomain.MultiDomain
    :members:
    :private-members: _kC_conn_sdt, _gauss_field_operators, _tsl_operators,
                      _kC_TSL


Helper functions
----------------

Conventions of the connections, used by :meth:`.MultiDomain.get_kC_conn`.

.. autofunction:: panels.multidomain.multidomain._sb_top_bottom

.. autofunction:: panels.multidomain.multidomain._bf_base_flange

.. autofunction:: panels.multidomain.multidomain._rinv_sanders

Number of terms across a joint, checked by
:meth:`.MultiDomain.check_conn_terms`, see the Notes of
:class:`.MultiDomain`.

.. autofunction:: panels.multidomain.multidomain._terms_linear

.. autoclass:: panels.multidomain.multidomain.ConnectionTermsWarning
