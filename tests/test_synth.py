import pytest

from attune.concordance_engine.engine import Engine
from attune.packs.pcos import PCOS_PACK
from attune.packs.veteran import VETERAN_PACK
from attune.synth import generate, load_memory, save

# generate() plants the flare over [days-12, days-7); at days=90 that is [78, 83).
FLARE_DAY = 80
CALM_DAY = 50


def test_generate_is_deterministic():
    assert generate(VETERAN_PACK, seed=7).signals == generate(VETERAN_PACK, seed=7).signals


@pytest.mark.parametrize("pack", [PCOS_PACK, VETERAN_PACK])
def test_planted_flare_is_concordant_and_calm_days_are_quiet(pack):
    eng = Engine(pack, generate(pack, days=90, seed=2))
    assert not eng.reflect(day=CALM_DAY).concordant
    assert eng.reflect(day=FLARE_DAY).concordant


def test_roundtrip_preserves_signals(tmp_path):
    mem = generate(PCOS_PACK, seed=3)
    path = tmp_path / "pcos.json"
    save(mem, path)
    assert load_memory(path).signals == mem.signals
