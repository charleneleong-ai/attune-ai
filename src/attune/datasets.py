from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetStub:
    name: str
    url: str
    modalities: tuple[str, ...]
    heads: tuple[str, ...]
    note: str


DATASET_CATALOG: tuple[DatasetStub, ...] = (
    DatasetStub(
        name="BIDSleep",
        url="https://physionet.org/content/bidsleep-dataset/1.0.0/",
        modalities=("wearable", "sleep", "context"),
        heads=("recovery", "sleep", "fatigue"),
        note="Apple Watch HR and accelerometry aligned to EEG sleep-stage labels.",
    ),
    DatasetStub(
        name="Real-world Smartwatch HRV",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12375003/",
        modalities=("wearable", "sleep", "context"),
        heads=("recovery", "stress", "fatigue"),
        note="Four-week Samsung watch HRV/motion data with sleep diaries and mental-health questionnaires.",
    ),
    DatasetStub(
        name="WESAD",
        url="https://complexity.cecs.ucf.edu/wesad-wearable-stress-and-affect-detection/",
        modalities=("wearable", "physiology", "context"),
        heads=("stress", "autonomic_load", "anomaly"),
        note="Wrist and chest physiology with stress and affect labels.",
    ),
    DatasetStub(
        name="ExtraSensory",
        url="https://extrasensory.ucsd.edu/",
        modalities=("wearable", "phone", "context"),
        heads=("work_burden", "activity", "context"),
        note="In-the-wild phone and smartwatch sensors with labels such as computer work, at work, sleeping, and talking.",
    ),
    DatasetStub(
        name="PAMAP2",
        url="https://huggingface.co/datasets/monster-monash/PAMAP2",
        modalities=("wearable", "activity", "video"),
        heads=("activity", "mobility", "exertion"),
        note="IMU and heart-rate activity benchmark used as a stand-in for movement/video-derived functional capacity.",
    ),
    DatasetStub(
        name="SSAQS",
        url="https://zenodo.org/records/18706837",
        modalities=("wearable", "sleep", "context"),
        heads=("stress", "anxiety", "fatigue"),
        note="Fitbit HRV, sleep, SpO2, activity, and daily stress/anxiety ratings.",
    ),
    DatasetStub(
        name="CGMacros",
        url="https://www.physionet.org/content/cgmacros/1.0.0/",
        modalities=("metabolic", "wearable", "image", "context"),
        heads=("diet_response", "metabolic", "glucose_response"),
        note="CGM, food macros, food photos, activity, labs, BMI, and diabetes/prediabetes status.",
    ),
    DatasetStub(
        name="HUPA-UCM",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11214197/",
        modalities=("metabolic", "wearable", "sleep", "context"),
        heads=("glucose_response", "diet_response", "medication_response"),
        note="CGM, insulin, meals, steps, calories, heart rate, and sleep for people with type 1 diabetes.",
    ),
    DatasetStub(
        name="Bridge2AI-Voice",
        url="https://physionet.org/content/b2ai-voice/1.1/",
        modalities=("voice", "clinical", "context"),
        heads=("voice_checkin", "respiratory", "mood", "neurological"),
        note="Voice-derived features linked to clinical information across voice-relevant conditions.",
    ),
    DatasetStub(
        name="DDI",
        url="https://ddi-dataset.github.io/",
        modalities=("image", "clinical"),
        heads=("visible_change", "skin"),
        note="Diverse dermatology images for skin lesion triage research.",
    ),
    DatasetStub(
        name="Lower-limb Wound",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC13090948/",
        modalities=("image", "clinical"),
        heads=("visible_change", "wound"),
        note="Lower-limb and foot wound images with clinically significant wound categories.",
    ),
    DatasetStub(
        name="SNAPMe",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10708545/",
        modalities=("image", "nutrition", "context"),
        heads=("diet_response", "food_photo"),
        note="Free-living food photos linked to detailed dietary records.",
    ),
)

DEMO_DATASET_NAMES = frozenset(
    {"BIDSleep", "WESAD", "CGMacros", "Bridge2AI-Voice", "DDI", "PAMAP2"}
)

DEMO_DATASETS: tuple[DatasetStub, ...] = tuple(
    dataset for dataset in DATASET_CATALOG if dataset.name in DEMO_DATASET_NAMES
)


def real_dataset_catalog() -> tuple[DatasetStub, ...]:
    return DATASET_CATALOG


def datasets_for_modality(modality: str) -> tuple[DatasetStub, ...]:
    return tuple(
        dataset for dataset in DATASET_CATALOG if modality in dataset.modalities
    )
