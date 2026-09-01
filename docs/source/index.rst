CV Pipeline Documentation
=========================

Welcome to the CV Pipeline documentation. The full content is organised
along the `Diátaxis framework <https://diataxis.fr/>`_ — see the
:doc:`landing page <landing>` for a guided overview, or use the
navigation tree below.

.. note::

   This page is the root of the documentation index and exposes the
   API surface via ``automodule`` directives. For a guided tour of
   the package, start at the :doc:`landing page <landing>` instead.


.. toctree::
   :maxdepth: 2
   :caption: Contents

   landing
   tutorials/quickstart
   how-to/index
   reference/index
   reference/specification
   explanation/index


API surface (via automodule)
----------------------------

The following ``automodule`` directives generate the public API
reference for the ``cv_pipeline`` package and the ``api`` backend
directly from the source-code docstrings.

cv_pipeline
~~~~~~~~~~~

.. automodule:: cv_pipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: cv_pipeline.schema
   :members:
   :undoc-members:
   :no-index:

.. automodule:: cv_pipeline.validation
   :members:
   :undoc-members:
   :no-index:

.. automodule:: cv_pipeline.weights
   :members:
   :undoc-members:
   :no-index:

.. automodule:: cv_pipeline.cli
   :members:
   :undoc-members:
   :no-index:

api backend
~~~~~~~~~~~

.. automodule:: api.routers.infer
   :members:
   :undoc-members:
   :no-index:

.. automodule:: api.routers.health
   :members:
   :undoc-members:
   :no-index:

.. automodule:: api.auth.api_key
   :members:
   :undoc-members:
   :no-index:

.. automodule:: api.middleware.request_id
   :members:
   :undoc-members:
   :no-index:

.. automodule:: api.schemas.infer
   :members:
   :undoc-members:
   :no-index:

.. automodule:: api.schemas.health
   :members:
   :undoc-members:
   :no-index:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
