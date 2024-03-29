QuasiStatic
-----------

Stating point
:::::::::::::

Quasistatic, event-driven, simulations.

1.  Generate realisations

    :ref:`QuasiStatic_Generate`

2.  Run event-driven quasistatic simulations of one realisation

    :ref:`QuasiStatic_Run`

3.  Collect basic output data of ensemble of realisations

    :ref:`QuasiStatic_EnsembleInfo`

Post-process
::::::::::::

Post-process the output of the quasistatic simulations:

-   Structure factor after system spanning events [ensemble]

    :ref:`QuasiStatic_StructureAfterSystemSpanning`

-   Activation barriers after system spanning events [ensemble]

    :ref:`QuasiStatic_StateAfterSystemSpanning`

-   Basic plotting [realisation or ensemble]

    :ref:`QuasiStatic_Plot`

-   Basic visualisation using Paraview [realisation]

    :ref:`QuasiStatic_Paraview`

Dynamics
::::::::

Analyse dynamics of one event

1.  Re-run event, store dynamics [realisation]

    :ref:`Dynamics_Run`

2.  Collect averages at fixed 'time' of several events [ensemble]

    :ref:`Dynamics_AverageSystemSpanning`

EventMap
::::::::

Analyse the sequences failures of one event

1.  Re-run event, store time and position of each failure [realisation]

    :ref:`EventMap_Run`

2.  Extract basic info [ensemble]

    :ref:`EventMap_Info`

3.  Other post-processing:

    -   Basic plotting [realisation]

        :ref:`EventMap_Plot`

    -   Basic visualisation using Paraview [realisation]

        :ref:`EventMap_Paraview`

Relaxation
::::::::::

Analyse the relaxation of a system spanning event

1.  Re-run event, store output [realisation]

    :ref:`Relaxation_Run`

2.  Compute average rheology [ensemble]

    :ref:`Relaxation_EnsembleInfo`

Trigger
:::::::

Branch to trigger at different forces

1.  Branch quasistatic simulations [ensemble]

    :ref:`Trigger_Generate`

2.  Trigger and minimise [realisation]

    :ref:`Trigger_Run`

3.  Collect basic output data [ensemble]

    :ref:`Trigger_EnsembleInfo`
