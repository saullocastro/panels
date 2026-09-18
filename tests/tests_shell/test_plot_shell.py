import sys
sys.path.append('../..')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from panels.shell import Shell
from panels.plot_shell import plot_shell


@pytest.fixture
def shell_c():
    s = Shell(a=1., b=0.5, r=None, stack=[0, 90], plyt=1e-3,
              laminaprop=(71e9, 0.33), model='plate_clpt_donnell', m=5, n=5)
    rng = np.random.default_rng(0)
    c = rng.random(s.get_size())*1e-3
    return s, c


@pytest.mark.parametrize('vec', ['w', 'exx', 'Nxx'])
def test_plot_shell_fields(shell_c, vec):
    s, c = shell_c
    fig, ax = plt.subplots()
    out = plot_shell(s, c, vec=vec, ax=ax, gridx=8, gridy=6, num_levels=5)
    assert out is ax
    assert len(ax.collections) > 0
    plt.close(fig)


def test_plot_shell_restores_fields(shell_c):
    s, c = shell_c
    s.uvw(c, gridx=4, gridy=3)
    w = s.fields['w'].copy()
    fig, ax = plt.subplots()
    plot_shell(s, c, vec='u', ax=ax, gridx=8, gridy=6, num_levels=5)
    plt.close(fig)
    assert np.array_equal(s.fields['w'], w)


def test_plot_shell_options(shell_c, tmp_path):
    s, c = shell_c
    filename = tmp_path / 'shell.png'
    ax = plot_shell(s, c, vec='Mxx', deform_u=True, deform_u_sf=10.,
                    colorbar=True, cbar_title='Mxx', invert_y=True,
                    title='title', clean=False, colormap=1,
                    texts=[dict(x=0.1, y=0.1, s='text')],
                    filename=str(filename), dpi=20,
                    gridx=8, gridy=6, num_levels=5)
    assert filename.exists()
    assert ax.get_title() == 'title'
    assert ax.yaxis_inverted()


def test_plot_shell_invalid_colormap(shell_c):
    s, c = shell_c
    fig, ax = plt.subplots()
    plot_shell(s, c, ax=ax, colormap='not_a_colormap', gridx=8, gridy=6,
               num_levels=5)
    plt.close(fig)


def test_plot_shell_invalid_input(shell_c):
    s, c = shell_c
    with pytest.raises(ValueError, match='not a valid vec'):
        plot_shell(s, c, vec='xyz')
    with pytest.raises(ValueError, match='Axes'):
        plot_shell(s, c, ax='not an axes', gridx=8, gridy=6)
    plt.close('all')
