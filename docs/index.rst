Overview
========

The following workflows are available:

QuasiStatic
-----------

Quasistatic, event-driven, simulations.

1.  Generate realisations

    :ref:`QuasiStatic_Generate <QuasiStatic_Generate>`

2.  Run event-driven quasistatic simulations of one realisation

    :ref:`QuasiStatic_Run <QuasiStatic_Run>`

3.  Collect basic output data of ensemble of realisations

    :ref:`QuasiStatic_EnsembleInfo <QuasiStatic_EnsembleInfo>`

QuasiStatic (post-process)
::::::::::::::::::::::::::

-   Structure factor after system spanning events [ensemble]

    :ref:`QuasiStatic_StructureAfterSystemSpanning <QuasiStatic_StructureAfterSystemSpanning>`

-   Activation barriers after system spanning events [ensemble]

    :ref:`QuasiStatic_StateAfterSystemSpanning <QuasiStatic_StateAfterSystemSpanning>`

-   Basic plotting [realisation or ensemble]

    :ref:`QuasiStatic_Plot <QuasiStatic_Plot>`

-   Basic visualisation using Paraview [realisation]

    :ref:`QuasiStatic_Paraview <QuasiStatic_Paraview>`

Dynamics
::::::::

Analyse dynamics of one event

1.  Re-run event, store dynamics [realisation]

    :ref:`Dynamics_Run <Dynamics_Run>`

2.  Collect averages at fixed 'time' of several events [ensemble]

    :ref:`Dynamics_AverageSystemSpanning <Dynamics_AverageSystemSpanning>`

EventMap
::::::::

Analyse the sequences failures of one event

1.  Re-run event, store time and position of each failure [realisation]

    :ref:`EventMap_Run <EventMap_Run>`

2.  Extract basic info [ensemble]

    :ref:`EventMap_Info <EventMap_Info>`

3.  Other post-processing:

    -   Basic plotting [realisation]

        :ref:`EventMap_Plot <EventMap_Plot>`

    -   Basic visualisation using Paraview [realisation]

        :ref:`EventMap_Paraview <EventMap_Paraview>`

Relaxation
::::::::::

Analyse the relaxation of a system spanning event

1.  Re-run event, store output [realisation]

    :ref:`Relaxation_Run <Relaxation_Run>`

2.  Compute average rheology [ensemble]

    :ref:`Relaxation_EnsembleInfo <Relaxation_EnsembleInfo>`

Trigger
:::::::

Branch to trigger at different forces

1.  Branch quasistatic simulations [ensemble]

    :ref:`Trigger_Generate <Trigger_Generate>`

2.  Trigger and minimise [realisation]

    :ref:`Trigger_Run <Trigger_Run>`

3.  Collect basic output data [ensemble]

    :ref:`Trigger_EnsembleInfo <Trigger_EnsembleInfo>`

Flow
----

Drive at constant velocity

1.  Generate realisations

    :ref:`Flow_Generate <Flow_Generate>`

2.  Run a single realisation

    :ref:`Flow_Run <Flow_Run>`

3.  Extract basic output of several realisations

    :ref:`Flow_EnsemblePack <Flow_EnsemblePack>`

Documentation
=============

.. toctree::
   :maxdepth: 1

   module.rst
   cli.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
