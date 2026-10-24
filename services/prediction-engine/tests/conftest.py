import sys, pathlib, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
RAW = pathlib.Path(__file__).resolve().parents[3] / "data" / "raw"
ART = pathlib.Path(__file__).resolve().parents[3] / "artifacts"
@pytest.fixture(scope="session")
def dataset():
    from app.data import build_dataset
    return build_dataset(data_dir=RAW)
