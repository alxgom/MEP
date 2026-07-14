from mep_routing.data_sources.catalog import DwellingCase, dwelling_cases_from_rows


def test_dwelling_catalog_normalizes_deduplicates_and_sorts_database_rows():
    rows = [
        {"project_guid": "project-b", "execution": "run-2", "dwelling_id": "home-1"},
        {"project_guid": "project-a", "execution": "run-1", "dwelling_id": "home-2"},
        {"project_guid": "project-a", "execution": "run-1", "dwelling_id": "home-2"},
        {"project_guid": "project-a", "execution": "run-1", "dwelling_id": "home-1"},
    ]

    assert dwelling_cases_from_rows(rows) == (
        DwellingCase("project-a", "run-1", "home-1"),
        DwellingCase("project-a", "run-1", "home-2"),
        DwellingCase("project-b", "run-2", "home-1"),
    )
    assert dwelling_cases_from_rows(rows)[0].label == "run-1 / home-1"
