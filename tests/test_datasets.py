from attune.datasets import (
    DEMO_DATASETS,
    datasets_for_modality,
    real_dataset_catalog,
)


def test_real_dataset_catalog_covers_core_modalities():
    catalog = real_dataset_catalog()
    modalities = {modality for dataset in catalog for modality in dataset.modalities}

    assert {"wearable", "voice", "image", "video", "metabolic", "context"}.issubset(
        modalities
    )


def test_demo_datasets_include_public_wearable_voice_and_metabolic_sources():
    names = {dataset.name for dataset in DEMO_DATASETS}

    assert "BIDSleep" in names
    assert "WESAD" in names
    assert "CGMacros" in names
    assert "Bridge2AI-Voice" in names


def test_dataset_lookup_filters_by_modality():
    image_sets = datasets_for_modality("image")
    wearable_sets = datasets_for_modality("wearable")

    assert any(dataset.name == "DDI" for dataset in image_sets)
    assert any(dataset.name == "BIDSleep" for dataset in wearable_sets)
    assert all("image" in dataset.modalities for dataset in image_sets)
