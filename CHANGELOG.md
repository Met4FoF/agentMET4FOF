# Changelog

<!--next-version-placeholder-->

## v0.13.2 (2022-02-03)
### Fix
* **#266:** Correct Mesa agents' names if only one space ([`263b07a`](https://github.com/Met4FoF/agentMET4FOF/commit/263b07ace484a482e2ba2f48ead210391c2f3d13))

### Refactor
* **Backend:** Switch to Backend enums to avoid errors with wrong initialization ([`08f769c`](https://github.com/Met4FoF/agentMET4FOF/commit/08f769cc475334e3250d0514e471dad835125aa1))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.13.1...v0.13.2)**

## v0.13.1 (2022-02-01)
### Fix
* **tutorial2:** Repair math agent in tutorial 2 ([`8cdc12e`](https://github.com/Met4FoF/agentMET4FOF/commit/8cdc12ebdc5d4a1792d9b1c1b3a72d9c721cbf3c))

### Refactor
* **signal_streams:** Turn wave methods into static methods in SineGenerator and CosineGenerator ([`9d24259`](https://github.com/Met4FoF/agentMET4FOF/commit/9d24259b9d9e240a2ca90f9a07542cef0527db1d))
* **signal_streams:** Remove unneeded parenthesis in SineGenerator and CosineGenerator ([`dce0d86`](https://github.com/Met4FoF/agentMET4FOF/commit/dce0d864629a254338729f02916bb37926fa696e))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.13.0...v0.13.1)**

## v0.13.0 (2021-11-12)
### Feature
* **Docker:** Transfer image to Docker Hub for more convenient usage ([`14aff5f`](https://github.com/Met4FoF/agentMET4FOF/commit/14aff5f49487a30bc0743da89613256054e8eac5))

### Documentation
* **Docker:** Update documentation to reflect new image location ([`753f461`](https://github.com/Met4FoF/agentMET4FOF/commit/753f46138af7dd1400fa8ef16ed0d04f58481f77))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.12.2...v0.13.0)**

## v0.12.1 (2021-08-27)
### Fix
* **DataStreamAgent:** Insert missing legacy import for agents.DataStreamAgent ([`302f61a`](https://github.com/Met4FoF/agentMET4FOF/commit/302f61ac7ac1ce976df3308999cafe6cde8a5643))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.12.0...v0.12.1)**

## v0.12.0 (2021-07-30)
### Feature
* **NoiseJitterRemovalAgent:** Introduce an agent to reduce the amount of noise and jitter in streamed sensor values ([`4349548`](https://github.com/Met4FoF/agentMET4FOF/commit/4349548fd0402c7cdbc661ba5a440ba14f7ee312))
* **SineWithJitter:** Introduce a streaming data stream of a sine signal with jitter and the corresponding agent ([`b6ca636`](https://github.com/Met4FoF/agentMET4FOF/commit/b6ca636fc7eeae693819bd82893328d3e838fdd0))

### Refactor
* **Dashboard:** Rename `Dashboard.is_port_available` to `Dashboard.is_port_at_ip_available` ([`25a9700`](https://github.com/Met4FoF/agentMET4FOF/commit/25a970013e9a08fc1b9a154562754300e03c6a98))

### Documentation
* **README:** Introduce link and note about noise and jitter removal tutorial ([`59cc55e`](https://github.com/Met4FoF/agentMET4FOF/commit/59cc55e04302f0302ff931eb59285f7b066411c8))
* **signal_streams:** Improve docstrings and type hints ([`a7e0007`](https://github.com/Met4FoF/agentMET4FOF/commit/a7e0007fcd460458d629546c99d68ef1f2079c11))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.11.0...v0.12.0)**

## v0.11.0 (2021-07-29)
### Feature
* **Plots:** Adapt dashboard such that all agents of types `MonitorAgent` and `MetrologicalMonitorAgent` get plotted regardless of their name ([`67a2544`](https://github.com/Met4FoF/agentMET4FOF/commit/67a25448f1709fbf8a5f9a4b6991d0c1fccb740f))
* **network:** Introduce a method `agents_by_type` to query for agents of a given type ([`e43a5b9`](https://github.com/Met4FoF/agentMET4FOF/commit/e43a5b9a7e2fffe53b90c2de6baae74265a400c2))

### Fix
* Allow spaces in agents' names which get replaced by underscores for the _osBrain_ backend ([`2d8437d`](https://github.com/Met4FoF/agentMET4FOF/commit/2d8437daeb04ff19a52c771d31ca5ec16b5c3e68))

### Refactor
* **AgentNetwork:** Change some of the internal variable ([`06c9a36`](https://github.com/Met4FoF/agentMET4FOF/commit/06c9a3661c6f9eab788c49b8044bbdc61b009893))
* **SineGeneratorAgent:** Improve data sending by including actually computed time ([`87c3e34`](https://github.com/Met4FoF/agentMET4FOF/commit/87c3e341ff1ca3974b4967974c4f45ef04fb92aa))
* **signal_agents:** Introduce "official" _sine_stream instance variable for SineGeneratorAgent ([`98d0f19`](https://github.com/Met4FoF/agentMET4FOF/commit/98d0f1995003aea5e843fef547d4e09cfe8f6aa4))
* **signal_agents:** Reintroduce NoiseAgent placeholder after module refactorization ([`f97f4d9`](https://github.com/Met4FoF/agentMET4FOF/commit/f97f4d9d281379502ea0e9c90649d3c59d0b224b))

### Documentation
* **network:** Introduce some more type hints ([`ee4b158`](https://github.com/Met4FoF/agentMET4FOF/commit/ee4b158664397362b403d044f30084560cd71b79))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.10.1...v0.11.0)**

## v0.10.1 (2021-07-23)
### Fix
* **CODE_OF_CONDUCT:** Finally add a code of conduct ([`56ee503`](https://github.com/Met4FoF/agentMET4FOF/commit/56ee503d09157f6b83da926c24b841171ded5ffd))

### Refactor
* **CONTRIBUTING:** Change heading and location of _CONTRIBUTING.md_ to make it more visible ([`1ff173f`](https://github.com/Met4FoF/agentMET4FOF/commit/1ff173f6b63de8cff9e06fd3e5fad3edc1c60b18))

### Documentation
* **README:** Introduce key facts ([`0e2d936`](https://github.com/Met4FoF/agentMET4FOF/commit/0e2d936c5d44fd26a26cec979c7d3bd32247d68e))
* **README:** Introduce Tutorials section ([`e1f41e9`](https://github.com/Met4FoF/agentMET4FOF/commit/e1f41e9a745c4635d1749ae37274c7fa89096cf1))
* **README:** Introduce Contributing section ([`85e8d9f`](https://github.com/Met4FoF/agentMET4FOF/commit/85e8d9f0fe7013eea8a761feb6662dc06ecb9c2c))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.10.0...v0.10.1)**

## v0.10.0 (2021-07-21)
### Feature
* **Docker:** Create container image for every release ([`1a40eaf`](https://github.com/Met4FoF/agentMET4FOF/commit/1a40eaf5724a0f956074d7cbee69dcee449c009b))

### Refactor
* **network:** Change delivery ip to 0.0.0.0, i.e. all interfaces instead of 127.0.0.1 to enable docker deployment ([`2472826`](https://github.com/Met4FoF/agentMET4FOF/commit/24728269a1485376ebaf2492d3849c232f88c05a))

### Documentation
* **ReadTheDocs:** Include Docker guide ([`7c35bba`](https://github.com/Met4FoF/agentMET4FOF/commit/7c35bba0aa45635c782830389d0c53c0c23d9cb7))
* **CONTRIBUTING:** Mention commit type refactor in contributing guide ([`a4891e7`](https://github.com/Met4FoF/agentMET4FOF/commit/a4891e7a5d4267955f8a1d582c2faf3bc369cc09))
* **metrological_redundancy_agents:** Introduce redundancy agent into docs ([`cb493cb`](https://github.com/Met4FoF/agentMET4FOF/commit/cb493cb24aa1c6cc499446552e952b5ec053a8a7))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.9.0...v0.10.0)**

## v0.9.0 (2021-07-08)
### Feature
* **agents:** Adds `ampl` and `phase_ini` parameters to SineGeneratorAgent ([`774a91b`](https://github.com/Met4FoF/agentMET4FOF/commit/774a91b454aac2f9caa1c3ce5b6577ccea27dfdb))
* **streams:** Adds `ampl` and `phase_ini` parameters to SineGenerator and CosineGenerator ([`998b158`](https://github.com/Met4FoF/agentMET4FOF/commit/998b158840c6cfdb0c264e4a381099c332a3cdc7))
* **metrological_streams:** Adds `ampl` and `phase_ini` parameters to MetrologicalSineGenerator ([`fc9901e`](https://github.com/Met4FoF/agentMET4FOF/commit/fc9901edcc8b5c2c0d033d04509913336edb8036))
* **redundancy agent:** Adds redundancy agent class to metrological_agents.py ([`7fbe934`](https://github.com/Met4FoF/agentMET4FOF/commit/7fbe9349d4688b8c7e424525a4499aaefced4daf))
* **MetrologicalGeneratorAgent:** Add an agent to handle generic metrological data streams ([`3334762`](https://github.com/Met4FoF/agentMET4FOF/commit/33347624e690d40233392ccf0073c55a424c6040))
* **MET4FoF redundancy:** Method init_parameters1 removed. removed unnecessary comments ([`afe47c3`](https://github.com/Met4FoF/agentMET4FOF/commit/afe47c3c81d5b6c77b2ef28851815a4aa6d6f429))
* **MET4FoF redundancy:** Second redundancy tutorial integrated ([`c527f33`](https://github.com/Met4FoF/agentMET4FOF/commit/c527f33a527aeaa9cc02f767687f3c735c6403b4))
* **MET4FoF redundancy:** Tutorial 7 added ([`caa62ca`](https://github.com/Met4FoF/agentMET4FOF/commit/caa62ca67ab8538809cf413b1bf71ee19234a236))
* **MET4FoF redundancy:** Met4fof redundancy modules added ([`bb3cb6a`](https://github.com/Met4FoF/agentMET4FOF/commit/bb3cb6ae9b1f5713c227fdec6a483c68ab2f51d2))
* **multiwave:** Added multiwave generator to metrological_streams.py ([`7b0bf4e`](https://github.com/Met4FoF/agentMET4FOF/commit/7b0bf4eb97a10719248ab6166ce300786cdd3503))
* **multiwave:** Added multiwave generator to metrological_streams.py ([`035d7c9`](https://github.com/Met4FoF/agentMET4FOF/commit/035d7c976e96ff428fb68a720d99ee2f3e0c474b))

### Fix
* **tutorial 3:** Replace one of the generator agents' streams by the cosine stream ([`2f20183`](https://github.com/Met4FoF/agentMET4FOF/commit/2f2018375766407a7590e55d8efb317fb315876f))
* **metrological_agents:** AgentBuffer import was accidentally removed ([`cbec978`](https://github.com/Met4FoF/agentMET4FOF/commit/cbec978558919a40271c3e40324e47fea51bac19))

### Refactor
* **agents and metrological_streams and streams:** Renames 'ampl' to 'amplitude' and 'phase_ini' to 'initial_phase' ([`d0d8271`](https://github.com/Met4FoF/agentMET4FOF/commit/d0d82719a1055054c135c899dfb13cb4ac554ec1))
* **metrological_agents:** Refactor metrological_agents.py into a package metrological_agents and include classes into agents ([`5558008`](https://github.com/Met4FoF/agentMET4FOF/commit/55580085cfb3e9182606e78b88e14db3c75d3f8f))
* **metrological_streams:** Refactor metrological_streams.py into a package metrological_streams and include classes into streams ([`9da3744`](https://github.com/Met4FoF/agentMET4FOF/commit/9da3744972737c8c1a863e4712c85385348b3d0e))
* **base_streams:** Refactor base_streams.py by applying black ([`34ff2d0`](https://github.com/Met4FoF/agentMET4FOF/commit/34ff2d0841bb477cb72f514fd23236adf38e2238))
* **signal_streams:** Introduce __all__ variable into signal_streams.py ([`8c62972`](https://github.com/Met4FoF/agentMET4FOF/commit/8c62972a5798e1220aae735bf3352530cb627de4))
* **signal_streams:** Refactor signal_streams.py by applying black ([`e604418`](https://github.com/Met4FoF/agentMET4FOF/commit/e604418516ee6adb62bb881d9cbd211d441e57a2))
* **streams:** Refactor streams.py into a package streams with the modules base_streams and signal_streams ([`11368ee`](https://github.com/Met4FoF/agentMET4FOF/commit/11368eeeb95f162508627e614daebb0a977e6dfb))
* **signal_agents:** Introduce __all__ variable into signal_agents.py ([`364702f`](https://github.com/Met4FoF/agentMET4FOF/commit/364702f493dba8c8a727b3140180badb9b4c7554))
* **signal_agents:** Refactor signal_agents.py by applying black ([`396a95e`](https://github.com/Met4FoF/agentMET4FOF/commit/396a95ed59f9ffa04252f390a7acd70af1692e36))
* **base_agents:** Introduce __all__ variable into base_agents.py ([`34f626b`](https://github.com/Met4FoF/agentMET4FOF/commit/34f626b7dcf4d5d45c2dbdccb244411c55bcaf5f))
* **base_agents:** Refactor base_agents.py by applying black and shortening line lengths ([`8173324`](https://github.com/Met4FoF/agentMET4FOF/commit/817332411f06da14bebfb983c1890a07bd88a4db))
* **agents:** Refactor agents.py into a package agents with the modules base_agents and signal_agents ([`c7bdfae`](https://github.com/Met4FoF/agentMET4FOF/commit/c7bdfae443d203ce07820bcebceb41decc096cb5))
* **multi_generator:** Replaces `amplitude` with `value` in variable names in generator classes ([`ee45e1c`](https://github.com/Met4FoF/agentMET4FOF/commit/ee45e1c4144bbe35b7724fbe40496ef62becdf44))
* **simple_generator:** Replaces `amplitude` with `value` in variable names in generator classes ([`f7475ba`](https://github.com/Met4FoF/agentMET4FOF/commit/f7475ba530ccb076075ecbff0ca7d6d34919d253))
* **metrological agents and streams:** Make imports relative ([`907cdb2`](https://github.com/Met4FoF/agentMET4FOF/commit/907cdb2638869e10532913842615644058f46d97))
* **streams:** Replaces `amplitude` with `value` in variable names in generator classes ([`177b231`](https://github.com/Met4FoF/agentMET4FOF/commit/177b231f797461e052a6adaeddb391eae8cd2210))
* **metrological_streams:** Replaces `amplitude` with `value` in variable names ([`b4a0118`](https://github.com/Met4FoF/agentMET4FOF/commit/b4a0118ec52f87fced47a9164fa028fc7b81d729))
* **metrological_streams:** Reorder parameters ([`b0c465f`](https://github.com/Met4FoF/agentMET4FOF/commit/b0c465feb4867d0792b1ea31538cfed5879f8a69))
* **MetrologicalDataStreamMet4FoF:** Refine MetrologicalDataStreamMet4FoF docstring ([`1bfc2a6`](https://github.com/Met4FoF/agentMET4FOF/commit/1bfc2a6df225e762b56f8dbe3f54a98d887a315d))

### Documentation
* **ReadTheDocs:** Adapt documentation to new package and module structure ([`ab207c4`](https://github.com/Met4FoF/agentMET4FOF/commit/ab207c41a0efea5ce205eac3c2249f5bd379f8d5))
* **tutorial 3:** Update fixed notebook with most recent output ([`5511079`](https://github.com/Met4FoF/agentMET4FOF/commit/5511079132b067caef77cca01d6ef968d20893f5))
* **CONTRIBUTING:** Change commit log examples to `develop`'s commits ([`9ee020f`](https://github.com/Met4FoF/agentMET4FOF/commit/9ee020fcecf7f0731f44d31c698c8376fe17ad8e))
* **CHANGELOG:** Insert actually current changelog into docs ([`aff2f6a`](https://github.com/Met4FoF/agentMET4FOF/commit/aff2f6a85f1c07f50c0dd718de85cad420dd6ee4))

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.8.1...v0.9.0)**

## v0.8.1 (2021-06-10)


**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/v0.8.0...v0.8.1)**

## v0.8.0 (2021-06-10)
### Feature
* **MetrologicalGeneratorAgent:** Rewrote to default metrological generator agent for both streams ([`52606d7`](https://github.com/Met4FoF/agentMET4FOF/commit/52606d7583166c397594bc12585635825352dd1f))
* **SineGeneratorAgent:** Added default sine generator agent to metrological_agents.py ([`54f4035`](https://github.com/Met4FoF/agentMET4FOF/commit/54f403541ccf713a666bec9764dcc265ff5c66e4))
* **multiwave_generator:** Multiwave generator added to metrological_streams.py ([`682a5ff`](https://github.com/Met4FoF/agentMET4FOF/commit/682a5ff2379c36a57ffea610aa7f113489e72e95))

### Documentation
* **tutorial_7:** Introduce tutorial 7 into ReadTheDocs ([`01d7e44`](https://github.com/Met4FoF/agentMET4FOF/commit/01d7e448a280e0653e49d5e5a1ecdef1fa0dfcb9))
* **tutorial_default_generator:** Add a tutorial for a generic metrologically enabled agent ([`7bebef1`](https://github.com/Met4FoF/agentMET4FOF/commit/7bebef1c56004c532cdc658548db2a82df6590de))

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

## v0.6.1 (2021-03-12)
### Fix
* **metrological_streams:** Fixed computational error for batches of size more than one ([`686bad5`](https://github.com/Met4FoF/agentMET4FOF/commit/686bad58ee02b216ffb3c663c9492d7dd7aaf6df))
* **agents:** Fixed `buffer.clear` method, which did not work anymore after moving from `memory` to `buffer`

### Documentation
* **README:** Include the Zenodo DOI banner and add the Citation section ([`4a57dd8`](https://github.com/Met4FoF/agentMET4FOF/commit/4a57dd8402590b9439ecf3bc47fbf950c6d44c7d))
* **docstrings:** Seriously improve docstrings and type hinting in `agents.py` and `dashboard/LayoutHelper.py`

**[See all commits in this version](https://github.com/Met4FoF/agentMET4FOF/compare/0.6.0...v0.6.1)**