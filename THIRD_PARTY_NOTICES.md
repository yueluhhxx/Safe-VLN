# Third-party notices

The implementation builds on [ETPNav](https://github.com/MarSaKi/ETPNav).
Its MIT license and original copyright notice are preserved in [LICENSE](LICENSE).

Occupancy sensing follows the simulated planar laser setup in
[Sim2Sim-VLNCE](https://github.com/jacobkrantz/Sim2Sim-VLNCE).
Its notice is retained in [licenses/Sim2Sim-MIT.txt](licenses/Sim2Sim-MIT.txt).

The waypoint predictor includes Hugging Face / Google / NVIDIA transformer
components under Apache-2.0. Their source headers are preserved; the license is
included in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
`vlnce_baselines/common/transformer.py` also credits the Facebook DETR project.
Microsoft waypoint-model code retains its original MIT header.

Other dependencies include [VLN-CE](https://github.com/jacobkrantz/VLN-CE),
[Habitat-Lab](https://github.com/facebookresearch/habitat-lab),
[Habitat-Sim](https://github.com/facebookresearch/habitat-sim),
[CLIP](https://github.com/openai/CLIP), and
[DUET](https://github.com/cshizhe/VLN-DUET). Their respective licenses apply.
The Habitat-Lab notice is included in
[licenses/Habitat-Lab-MIT.txt](licenses/Habitat-Lab-MIT.txt).

The README displays Figure 2 from [Safe-VLN](https://arxiv.org/abs/2311.02817)
by linking to the image hosted on arXiv. The figure is not covered by this
repository's code license; copyright remains with its rightsholders.
The [arXiv distribution license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/)
is not a general open-source license for the illustration.

Datasets, Matterport3D scenes and externally downloaded pretrained weights are
not included in the Git repository. Obtain them from their original providers
and follow their applicable terms. The code license does not replace those terms.
