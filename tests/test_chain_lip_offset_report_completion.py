from scripts.chain_tests.report_completion import g4_person_status


def test_g4_registered_status_uses_only_four_and_eight_frame_conditions():
    rows = {
        "W1": {
            "-8": {"G4_pass": True}, "-4": {"G4_pass": True},
            "-2": {"G4_pass": False}, "2": {"G4_pass": False},
            "4": {"G4_pass": True}, "8": {"G4_pass": True},
        },
        "W2": {
            "-8": {"G4_pass": True}, "-4": {"G4_pass": True},
            "-2": {"G4_pass": True}, "2": {"G4_pass": True},
            "4": {"G4_pass": True}, "8": {"G4_pass": True},
        },
    }
    assert g4_person_status(rows) == "pass"


def test_g4_registered_status_marks_any_indeterminate_condition_undetermined():
    rows = {window: {str(k): {"G4_pass": True} for k in (-8, -4, -2, 2, 4, 8)} for window in ("W1", "W2")}
    rows["W1"]["-4"]["G4_pass"] = None
    assert g4_person_status(rows) == "undetermined"
