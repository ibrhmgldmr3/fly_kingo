from flyhire.db.models import Base


def test_expected_tables():
    assert set(Base.metadata.tables) == {
        "jobs",
        "job_features",
        "decisions",
        "labels",
        "applications",
    }
