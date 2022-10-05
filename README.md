# tdsurv

Reproducibility package for the paper:

> Lucas Maystre, Daniel Russo. [_Temporally-Consistent Survival
> Analysis_](https://research.atspotify.com/publications/temporally-consistent-survival-analysis/).
> Advances in Neural Information Processing Systems 35 (NeurIPS 2022).

This repository contains

- a reference implementation of the algorithms presented in the paper, and
- Jupyter notebooks enabling the reproduction of some of the experiments.

The paper and the libary address the problem of learning survival models from
sequential observations (also known as the _dynamic_ setting). For an
accessible overview of the main idea, you can read our [blog
post](https://research.atspotify.com/2022/11/survival-analysis-meets-reinforcement-learning/).

## Getting Started

To get started, follow these steps:

- Clone the repo locally with: `git clone
  https://github.com/spotify-research/tdsurv.git`
- Move to the repository: `cd tdsurv`
- Install the dependencies: `pip install -r requirements.txt`
- Install the package: `pip install -e lib/`
- Move to the notebook folder: `cd notebooks`
- Start a notebook server: `jupyter notebok`

To reproduce some of the experimental results, you will need to download the
relevant datasets. You can find further instructions under `data/README.md`.

Our codebase was tested with Python 3.9.7. The following libraries are required
(and installed automatically via the first `pip` command above):

- jax (tested with version 0.3.4)
- jaxlib (tested with version 0.3.2)
- jupyter (tested with version 6.4)
- lifelines (tested with version 0.27.0)
- matplotlib (tested with version 3.5.1)
- numpy (tested with version 1.21.2)
- pandas (tested with version 1.4.1)
- scipy (tested with version 1.73)

## Support

Create a [new issue](https://github.com/spotify-research/tdsurv/issues/new)

## Contributing

We feel that a welcoming community is important and we ask that you follow Spotify's
[Open Source Code of Conduct](https://github.com/spotify/code-of-conduct/blob/main/code-of-conduct.md)
in all interactions with the community.

## Author

[Lucas Maystre](mailto:lucasm@spotify.com)

A full list of [contributors](https://github.com/spotify-research/tdsurv/graphs/contributors?type=a) can be found on GHC

Follow [@SpotifyResearch](https://twitter.com/SpotifyResearch) on Twitter for updates.

## License

Copyright 2022 Spotify, Inc.

Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

## Security Issues?

Please report sensitive security issues via Spotify's bug-bounty program (https://hackerone.com/spotify) rather than GitHub.
