from ft.utils.run_diagnostics import source_provenance


def test_source_provenance_hashes_config_and_identity_sources():
    first = source_provenance({"identity": {"global_team_jersey_owner": True}})
    repeated = source_provenance({"identity": {"global_team_jersey_owner": True}})
    changed = source_provenance({"identity": {"global_team_jersey_owner": False}})

    assert first == repeated
    assert first["config_sha256"] != changed["config_sha256"]
    assert all(item["exists"] for item in first["sources"].values())
    assert all(len(item["sha256"]) == 64 for item in first["sources"].values())
