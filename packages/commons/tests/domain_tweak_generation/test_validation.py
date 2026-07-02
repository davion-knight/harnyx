from __future__ import annotations

import json

from harnyx_commons.domain_tweak_generation.validation import (
    validate_form_review_output,
    validate_question_generation_output,
    validate_reference_answer_output,
)


def test_question_validation_accepts_candidate_and_rejects_reference_answer_field() -> None:
    valid = validate_question_generation_output(
        json.dumps(
            {
                "question": "Which players meet all constraints?",
                "short_answer": "Ada Example; Ben Example",
                "solution_plan": "- Find candidate pool\n- Intersect constraints",
            }
        )
    )
    invalid = validate_question_generation_output(
        json.dumps(
            {
                "question": "Which players meet all constraints?",
                "short_answer": "Ada Example; Ben Example",
                "solution_plan": "- Find candidate pool",
                "reference_answer": {"text": "Ada Example and Ben Example."},
            }
        )
    )

    assert valid.ok is True
    assert valid.terminal_status == "validated"
    assert invalid.ok is False
    assert invalid.terminal_status == "validation_failed"


def test_question_validation_accepts_no_generate_terminal_status() -> None:
    outcome = validate_question_generation_output(
        json.dumps(
            {
                "no_generate": True,
                "reason": "The source domain does not support the original aggregation form.",
                "retry_recommended": False,
            }
        )
    )

    assert outcome.ok is True
    assert outcome.terminal_status == "no_generate"
    assert outcome.parsed_output is None


def test_form_review_validation_converts_mismatch_to_form_rejected() -> None:
    outcome = validate_form_review_output(
        json.dumps(
            {
                "form_match": False,
                "false_premise_status": "none",
                "reviewer_feedback": "The generated question lost the aggregation step.",
                "retry_recommended": True,
            }
        )
    )

    assert outcome.ok is True
    assert outcome.terminal_status == "form_rejected"
    assert outcome.feedback == ("The generated question lost the aggregation step.",)


def test_reference_answer_validation_requires_claim_bearing_citations() -> None:
    invalid = validate_reference_answer_output(
        json.dumps(
            {
                "question": "Which players meet all constraints?",
                "premise_assessment": "The premise is supported.",
                "reference_answer": {
                    "text": "Ada Example and Ben Example meet all constraints.",
                    "citations": [],
                },
            }
        )
    )
    valid = validate_reference_answer_output(
        json.dumps(
            {
                "question": "Which players meet all constraints?",
                "premise_assessment": "The premise is supported by the official table.",
                "reference_answer": {
                    "text": "Ada Example and Ben Example meet all constraints.",
                    "citations": [
                        {
                            "url": "https://example.com/table",
                            "title": "Official table",
                            "note": "Lists Ada Example and Ben Example with all required constraints.",
                        }
                    ],
                },
            }
        )
    )

    assert invalid.ok is False
    assert "citations" in invalid.feedback[0]
    assert valid.ok is True
    assert valid.terminal_status == "validated"
