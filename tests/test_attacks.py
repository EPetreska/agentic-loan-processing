"""Tests for the attack list (evals/attacks.json)."""

from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from loans.evals.attacks import AttackList, InjectionPoint, load_attacks
from loans.evals.case import load_case

EVALS = Path(__file__).parent.parent / "evals"


def test_attack_list_loads():
    attacks = load_attacks(EVALS / "attacks.json")
    assert len(attacks.attacks) == 4


def test_base_and_victim_cases_exist():
    attacks = load_attacks(EVALS / "attacks.json")
    base = load_case(EVALS / "cases" / attacks.base_case)
    victim = load_case(EVALS / "cases" / attacks.victim_case)
    # The victim's canary values must not already appear in the base case,
    # otherwise a "leak" could not be told apart from the base case's own data.
    assert victim.application.ssn not in base.application.model_dump_json()
    assert victim.application.full_name not in base.application.model_dump_json()


def test_four_attacks_in_six_places_make_twenty_four_cases():
    attacks = load_attacks(EVALS / "attacks.json")
    assert len(attacks.attacks) * len(get_args(InjectionPoint)) == 24


def test_duplicate_ids_are_rejected():
    attack = {"id": "a", "text": "x", "goal": "leak_data", "categories": ["prompt_injection"]}
    with pytest.raises(ValidationError, match="duplicate"):
        AttackList(base_case="b", victim_case="v", attacks=[attack, attack])


def test_unknown_key_is_rejected():
    attack = {"id": "a", "text": "x", "goal": "leak_data", "categories": ["prompt_injection"], "gaol": "x"}
    with pytest.raises(ValidationError, match="gaol"):
        AttackList(base_case="b", victim_case="v", attacks=[attack])


def test_unknown_goal_is_rejected():
    attack = {"id": "a", "text": "x", "goal": "steal_money", "categories": ["prompt_injection"]}
    with pytest.raises(ValidationError):
        AttackList(base_case="b", victim_case="v", attacks=[attack])
