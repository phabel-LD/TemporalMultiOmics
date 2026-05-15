# tests/test_imports.py
def test_import_tmo():
    import tmo
    assert tmo.__version__ is not None

def test_import_submodules():
    from tmo.data import loader
    from tmo.ccf import local_ccf
    from tmo.models import TMOModel
    from tmo.training import PhaseATrainer
    from tmo.validation import compute_lcs
    from tmo.plotting import plot_delta_tau_heatmap