# Changelog

<!--next-version-placeholder-->

## v0.8.0 (2021-06-10)
### Feature
* **tutorial_7_generic_metrological_agent:** Adds notebook tutorial for the default metrological agent ([`34236bd`](https://github.com/Met4FoF/agentMET4FOF/commit/34236bdbd770004b77bb53266dd1d79b93b351a3))
* **tutorial_default_generator:** Add a tutorial for a generic metrologically enabled agent ([`7bebef1`](https://github.com/Met4FoF/agentMET4FOF/commit/7bebef1c56004c532cdc658548db2a82df6590de))
* **MetrologicalGeneratorAgent:** Rewrote to default metrological generator agent for both streams ([`52606d7`](https://github.com/Met4FoF/agentMET4FOF/commit/52606d7583166c397594bc12585635825352dd1f))
* **SineGeneratorAgent:** Added default sine generator agent to metrological_agents.py ([`54f4035`](https://github.com/Met4FoF/agentMET4FOF/commit/54f403541ccf713a666bec9764dcc265ff5c66e4))
* **multiwave_generator:** All instance attributes initialized in __init__ ([`09a4e08`](https://github.com/Met4FoF/agentMET4FOF/commit/09a4e0813d184852513c8b63a139915dc194f764))
* **multiwave_generator:** Value_unc calculated from exp_unc in init ([`caff216`](https://github.com/Met4FoF/agentMET4FOF/commit/caff216f39c71c6a3bd9d51b1289664df53bedde))
* **multiwave_generator:** Expanded uncertainty added to datastream. ([`348c813`](https://github.com/Met4FoF/agentMET4FOF/commit/348c813039ec6eec232c2152cf8f9624811940d2))
* **multiwave_generator:** Multiwave generator added to metrological_streams.py ([`682a5ff`](https://github.com/Met4FoF/agentMET4FOF/commit/682a5ff2379c36a57ffea610aa7f113489e72e95))

### Fix
* **metrological_agents:** Merges latest version of metrological_agents.py from develop ([`bb4734e`](https://github.com/Met4FoF/agentMET4FOF/commit/bb4734edbe0256a56a8a4114cb92404788147d34))
* **metrological_agents:** Updated docstring for MetrologicalGeneratorAgent ([`ac5f6e2`](https://github.com/Met4FoF/agentMET4FOF/commit/ac5f6e2fed6ea198f19316c618ead9dc97b9b411))
* **metrological_streams:**  fix the switched variables in the MultiwaveGenerator for-loop ([`62d786d`](https://github.com/Met4FoF/agentMET4FOF/commit/62d786d1fd97ffc43d4b83eefdd69ad9105d0c24))

### Documentation
* **tutorial_7:** Introduce tutorial 7 into ReadTheDocs ([`01d7e44`](https://github.com/Met4FoF/agentMET4FOF/commit/01d7e448a280e0653e49d5e5a1ecdef1fa0dfcb9))
* **tutorial_7:** Introduce expected output ([`fac4486`](https://github.com/Met4FoF/agentMET4FOF/commit/fac4486384926841b01cc2d3851d96e065797c0f))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.7.0...v0.8.0)**

## v0.7.0 (2021-06-04)
### Feature
* **enhance_ui:** Button to export agent network display as png ([`3b41610`](https://github.com/Met4FoF/agentMET4FOF/commit/3b41610b0673e17811fff28ec71defbf94c8815e))
* **enhance_subscribe:** Allow special channels for requests ([`a88a817`](https://github.com/Met4FoF/agentMET4FOF/commit/a88a817f2c9dcde8fae072349bdeb611c09249ba))

### Fix
* **enhance_UI:** Fix bug on callback of `agents-network.generateImage` ([`d9125ab`](https://github.com/Met4FoF/agentMET4FOF/commit/d9125ab3dec00b5c0edeb689cecce3a116b565fc))
* **enhance_UI:** Fix importing name for UI example ([`58b990b`](https://github.com/Met4FoF/agentMET4FOF/commit/58b990b39e6f32bca5f1730bf24ac4e6f3216550))
* **enhance_ui:** Required import visdcc ([`f02429e`](https://github.com/Met4FoF/agentMET4FOF/commit/f02429e2af3c99a61bfa13a718dbf8494b0a8499))
* **enhance_subscribe:** Specify the plot channel for tests ([`1d293e7`](https://github.com/Met4FoF/agentMET4FOF/commit/1d293e79d37552bfc17949b2e65e659447a84db4))

### Documentation
* **enhance_UI:** Refactor agent_type to agent_name_filter ([`a6f8da9`](https://github.com/Met4FoF/agentMET4FOF/commit/a6f8da90a43d6225140212c7ac3ad106569b5600))
* **enhance_UI:** Add docstring for create_edges_cytoscape method ([`b66fcb3`](https://github.com/Met4FoF/agentMET4FOF/commit/b66fcb3869d04495591daef992ea11b5823b7abb))
* **enhance_UI:** Intuitive name for toast message function in UI ([`c155138`](https://github.com/Met4FoF/agentMET4FOF/commit/c1551387c270f32053b76399500fb27b963db9dd))
* **enhance_UI:** Rename and add description to UI example ([`3a03428`](https://github.com/Met4FoF/agentMET4FOF/commit/3a0342875c4ea46582b7dab49cd8a7b10aa83b37))
* **enhance_UI:** Refactor tutorial folder on UI example ([`360f6da`](https://github.com/Met4FoF/agentMET4FOF/commit/360f6da39d249ce09a3384e867867d1fa91cc757))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.6.4...v0.7.0)**

## v0.6.4 (2021-05-06)
### Fix
* **streams:** Remove unnecessary instantiation of sine_freq in method sine_wave_function ([`192a6c9`](https://github.com/Met4FoF/agentMET4FOF/commit/192a6c9f662b014bf5ef21ec07ed8ba5638697c6))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.6.3...v0.6.4)**

## v0.6.3 (2021-04-28)
### Fix
* **time_series_buffer:** Resolve storing nan values ([`bbc1ae9`](https://github.com/Met4FoF/agentMET4FOF/commit/bbc1ae925d221fdd79715a46e48c91059b2d4cd6))

### Documentation
* **metrological_agents:** Reformat two of our docstrings to properly display thos example values in the docs ([`c56cca9`](https://github.com/Met4FoF/agentMET4FOF/commit/c56cca993502a35f1e39dd14df9b82fa00ba32c4))
* **metrological_agents:** Correct docstrings for `_concatenate()` ([`eb6a20b`](https://github.com/Met4FoF/agentMET4FOF/commit/eb6a20bd4a053d05064dd73b55c42554dcce41da))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.6.2...v0.6.3)**

## v0.6.2 (2021-03-19)
### Fix
* **dashboard:** Finally resolve #186 by cleanly shutting down dashboard ([`9903d15`](https://github.com/Met4FoF/agentMET4FOF/commit/9903d15b61033ab5e8536deb9318936ff89e4249))

### Documentation
* **CHANGELOG:** Include CHANGELOG into docs ([`4ba20d3`](https://github.com/Met4FoF/agentMET4FOF/commit/4ba20d376381ad5299853d112135204104165e6e))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.6.1...v0.6.2)**
