# Copyright 2022 Spotify AB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib

from matplotlib.backends.backend_pgf import FigureCanvasPgf


# See: <https://matplotlib.org/3.2.1/tutorials/text/pgf.html>
RCPARAMS = {
    "figure.figsize": (5.5, 2.0),       # Single column 5.5 in wide.
    "figure.dpi": 150,                  # Displays figures nicely in notebooks.
    "axes.linewidth": 0.5,              # Matplotlib's current default is 0.8.
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",             # use serif rather than sans-serif
    "font.serif": "Times",
    "font.size": 9,
    "axes.titlesize": 9,                # LaTeX default is 10pt font.
    "axes.labelsize": 8,                # LaTeX default is 10pt font.
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "legend.frameon": False,            # Remove the black frame around the legend
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.0,
}


def setup_plotting():
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
    matplotlib.rcParams.update(RCPARAMS)
    print("Plotting settings loaded!")
