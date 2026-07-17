from attune.concordance_engine.engine import Engine
from attune.demo import first_concordant_day
from attune.packs.veteran import VETERAN_PACK
from attune.synth import flare_window, generate

WINDOW = flare_window(90)


def test_demo_finds_first_concordant_day_in_the_flare_window():
    eng = Engine(VETERAN_PACK, generate(VETERAN_PACK, days=90, seed=2))
    day = first_concordant_day(eng, WINDOW.onset - 4, WINDOW.end + 1)
    assert day is not None
    assert WINDOW.onset <= day <= WINDOW.end


def test_demo_preflare_window_is_clean():
    # the days shown just before onset must be quiet, or "caught as the drift begins" is a lie
    eng = Engine(VETERAN_PACK, generate(VETERAN_PACK, days=90, seed=2))
    assert first_concordant_day(eng, WINDOW.onset - 4, WINDOW.onset - 1) is None
