"""
Diagnostic script to analyze JSON parsing failures from Bedrock batch output.

This script loads actual batch results and tests the parsing logic to identify
failure patterns and provide detailed error analysis.
"""

import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel, ValidationError, create_model, Field


# ============================================================================
# UNICODE QUOTE MAPPINGS (from main script)
# ============================================================================

UNICODE_DOUBLE_QUOTES = {
    "\u201c": "'",  # " → '
    "\u201d": "'",  # " → '
    "\u201e": "'",  # „ → '
    "\u201f": "'",  # ‟ → '
}

UNICODE_SINGLE_QUOTES = {
    "\u2018": "'",  # ' → '
    "\u2019": "'",  # ' → '
    "\u201a": "'",  # ‚ → '
    "\u201b": "'",  # ‛ → '
}

GERMAN_OPEN_QUOTE_PATTERN = re.compile(r'„([^""\u201c\u201d]*)["\u201c\u201d]')


# ============================================================================
# PARSING FUNCTIONS (from main script)
# ============================================================================


def normalize_unicode_quotes(text: str) -> str:
    """Normalize Unicode quotation marks to ASCII equivalents."""
    for bad, repl in UNICODE_DOUBLE_QUOTES.items():
        text = text.replace(bad, repl)
    for bad, repl in UNICODE_SINGLE_QUOTES.items():
        text = text.replace(bad, repl)
    return text


def repair_json(text: str) -> str:
    """Repair Unicode/fancy quotes in JSON responses."""
    text = GERMAN_OPEN_QUOTE_PATTERN.sub(r'\\"\1\\"', text)
    text = normalize_unicode_quotes(text)
    return text


def extract_json_candidate(response_text: str) -> str:
    """
    Extract the first complete JSON object using intelligent brace counting.
    This is the NEW implementation that fixes the brace imbalance issue.
    """
    start = response_text.find("{")
    if start == -1:
        return response_text.strip()

    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(start, len(response_text)):
        char = response_text[i]

        # Handle escape sequences
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        # Track string boundaries (braces inside strings don't count)
        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        # Count braces only outside strings
        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                # Found first complete JSON object when count returns to 0
                if brace_count == 0:
                    return response_text[start : i + 1]

    # Fallback: no complete object found, return from first brace onwards
    return response_text[start:].strip()


def extract_valid_json_smart(text: str) -> str:
    """
    Intelligent JSON extractor with brace counting.
    Finds first complete valid JSON object.
    """
    start = text.find("{")
    if start == -1:
        return text.strip()

    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start : i + 1]

    return text[start:].strip()


# ============================================================================
# VALIDATION SCHEMA (from output_format.json)
# ============================================================================


def create_response_model() -> type:
    """Create Pydantic model from validation schema."""
    # Simplified version based on the schema
    fields = {
        "category": (str, Field(..., description="Category classification")),
        "confidence_score": (float, Field(..., ge=0.0, le=1.0)),
        "key_evidence": (dict, Field(..., description="Evidence object")),
        "reasoning": (dict, Field(..., description="Reasoning object")),
    }

    return create_model("BedrockResponse", **fields)


# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================


def analyze_json_structure(json_str: str) -> Dict[str, Any]:
    """Analyze structural properties of JSON string."""
    return {
        "length": len(json_str),
        "starts_with_brace": json_str.strip().startswith("{"),
        "ends_with_brace": json_str.strip().endswith("}"),
        "opening_braces": json_str.count("{"),
        "closing_braces": json_str.count("}"),
        "brace_balance": json_str.count("{") - json_str.count("}"),
        "has_german_quotes": bool(re.search(r"„", json_str)),
        "has_unicode_quotes": bool(re.search(r"[\u201c\u201d\u201e\u201f]", json_str)),
        "has_trailing_chars": not json_str.strip().endswith("}")
        if json_str.strip().startswith("{")
        else None,
    }


def test_parsing_pipeline(
    response_text: str, response_model_class: type
) -> Dict[str, Any]:
    """Test the full parsing pipeline and collect diagnostic info."""
    result = {
        "original_length": len(response_text),
        "extraction_method": "none",
        "repair_applied": False,
        "parse_success": False,
        "validation_success": False,
        "error_type": None,
        "error_message": None,
        "original_structure": None,
        "extracted_structure": None,
        "repaired_structure": None,
    }

    try:
        result["original_structure"] = analyze_json_structure(response_text)

        # STEP 0: Handle assistant prefilling BEFORE extraction (CRITICAL FIX)
        # Prepend { BEFORE extraction to avoid grabbing nested objects
        if not response_text.strip().startswith("{"):
            response_text = "{" + response_text
            result["prepended_brace"] = True

        # STEP 1: Extract JSON candidate with smart brace counting
        complete_json = extract_json_candidate(response_text)
        result["extraction_method"] = "extract_json_candidate"
        result["extracted_length"] = len(complete_json)
        result["extracted_structure"] = analyze_json_structure(complete_json)

        # STEP 2: Try parsing as-is
        try:
            validated_response = response_model_class.model_validate_json(complete_json)
            result["parse_success"] = True
            result["validation_success"] = True
            return result
        except (ValidationError, json.JSONDecodeError) as e:
            result["error_type"] = type(e).__name__
            result["error_message"] = str(e)

            # Step 2: Try repair
            repaired_json = repair_json(complete_json)
            result["repair_applied"] = True
            result["repaired_structure"] = analyze_json_structure(repaired_json)

            try:
                validated_response = response_model_class.model_validate_json(
                    repaired_json
                )
                result["parse_success"] = True
                result["validation_success"] = True
                return result
            except (ValidationError, json.JSONDecodeError) as e2:
                result["error_type"] = type(e2).__name__
                result["error_message"] = str(e2)
                result["original_error"] = str(e)
                result["repair_error"] = str(e2)

                # Try smart extraction as fallback
                smart_json = extract_valid_json_smart(response_text)
                if smart_json != complete_json:
                    result["tried_smart_extraction"] = True
                    result["smart_extracted_structure"] = analyze_json_structure(
                        smart_json
                    )

                    try:
                        validated_response = response_model_class.model_validate_json(
                            smart_json
                        )
                        result["parse_success"] = True
                        result["validation_success"] = True
                        result["extraction_method"] = "smart_extraction"
                        return result
                    except:
                        pass

    except Exception as e:
        result["error_type"] = "Exception"
        result["error_message"] = str(e)

    return result


def load_batch_results(file_path: Path) -> List[Dict[str, Any]]:
    """Load batch results from JSONL output file."""
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    record = json.loads(line)
                    results.append(
                        {
                            "line_num": line_num,
                            "record_id": record.get("recordId"),
                            "record": record,
                        }
                    )
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
    return results


def analyze_batch_results(
    batch_results: List[Dict[str, Any]],
    response_model_class: type,
    max_failures_to_show: int = 10,
) -> Dict[str, Any]:
    """Analyze all batch results and generate comprehensive report."""

    report = {
        "total_records": len(batch_results),
        "successful_parses": 0,
        "failed_parses": 0,
        "repairs_successful": 0,
        "repairs_failed": 0,
        "error_types": {},
        "structural_issues": {
            "brace_imbalance": 0,
            "german_quotes": 0,
            "unicode_quotes": 0,
            "trailing_characters": 0,
        },
        "sample_failures": [],
    }

    for idx, batch_item in enumerate(batch_results):
        record = batch_item["record"]

        if "modelOutput" not in record:
            report["failed_parses"] += 1
            continue

        # Extract response text
        model_output = record["modelOutput"]
        if "content" in model_output and len(model_output["content"]) > 0:
            response_text = model_output["content"][0].get("text", "")
        else:
            report["failed_parses"] += 1
            continue

        # Test parsing
        test_result = test_parsing_pipeline(response_text, response_model_class)

        if test_result["validation_success"]:
            report["successful_parses"] += 1
            if test_result["repair_applied"]:
                report["repairs_successful"] += 1
        else:
            report["failed_parses"] += 1
            if test_result["repair_applied"]:
                report["repairs_failed"] += 1

            # Track error types
            error_type = test_result["error_type"]
            if error_type:
                report["error_types"][error_type] = (
                    report["error_types"].get(error_type, 0) + 1
                )

            # Track structural issues
            if test_result["extracted_structure"]:
                struct = test_result["extracted_structure"]
                if struct["brace_balance"] != 0:
                    report["structural_issues"]["brace_imbalance"] += 1
                if struct["has_german_quotes"]:
                    report["structural_issues"]["german_quotes"] += 1
                if struct["has_unicode_quotes"]:
                    report["structural_issues"]["unicode_quotes"] += 1
                if struct["has_trailing_chars"]:
                    report["structural_issues"]["trailing_characters"] += 1

            # Save sample failures
            if len(report["sample_failures"]) < max_failures_to_show:
                report["sample_failures"].append(
                    {
                        "record_id": batch_item["record_id"],
                        "line_num": batch_item["line_num"],
                        "response_preview": response_text[:500],
                        "test_result": test_result,
                    }
                )

    # Calculate rates
    report["success_rate"] = (
        report["successful_parses"] / report["total_records"]
        if report["total_records"] > 0
        else 0
    )
    report["repair_success_rate"] = (
        report["repairs_successful"]
        / (report["repairs_successful"] + report["repairs_failed"])
        if (report["repairs_successful"] + report["repairs_failed"]) > 0
        else 0
    )

    return report


def print_report(report: Dict[str, Any]):
    """Print formatted diagnostic report."""
    print("=" * 80)
    print("BEDROCK BATCH JSON PARSING DIAGNOSTIC REPORT")
    print("=" * 80)
    print()

    print(f"Total Records: {report['total_records']}")
    print(
        f"Successful Parses: {report['successful_parses']} ({report['success_rate']:.2%})"
    )
    print(
        f"Failed Parses: {report['failed_parses']} ({1 - report['success_rate']:.2%})"
    )
    print()

    print("Repair Statistics:")
    print(f"  Repairs Successful: {report['repairs_successful']}")
    print(f"  Repairs Failed: {report['repairs_failed']}")
    if report["repairs_successful"] + report["repairs_failed"] > 0:
        print(f"  Repair Success Rate: {report['repair_success_rate']:.2%}")
    print()

    print("Error Type Breakdown:")
    for error_type, count in sorted(
        report["error_types"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {error_type}: {count}")
    print()

    print("Structural Issues:")
    for issue, count in report["structural_issues"].items():
        if count > 0:
            print(f"  {issue}: {count}")
    print()

    print("=" * 80)
    print("SAMPLE FAILURE CASES")
    print("=" * 80)

    for i, failure in enumerate(report["sample_failures"], 1):
        print(
            f"\n--- Failure {i} (Record: {failure['record_id']}, Line: {failure['line_num']}) ---"
        )
        print(
            f"Error: {failure['test_result']['error_type']} - {failure['test_result']['error_message'][:200]}"
        )

        if failure["test_result"]["extracted_structure"]:
            struct = failure["test_result"]["extracted_structure"]
            print(
                f"Structure: Braces={struct['opening_braces']}/{struct['closing_braces']} (balance={struct['brace_balance']})"
            )
            print(
                f"           German quotes={struct['has_german_quotes']}, Unicode quotes={struct['has_unicode_quotes']}"
            )

        print(f"Response preview (first 500 chars):")
        print(failure["response_preview"])
        print()


def main():
    """Main diagnostic function."""
    # File paths - use absolute path or relative from script location
    batch_output_file = Path("input_20251110_061009.jsonl.out")

    if not batch_output_file.exists():
        print(f"Error: Batch output file not found: {batch_output_file}")
        print(f"Current working directory: {Path.cwd()}")
        print(f"Script directory: {Path(__file__).parent}")
        return

    print(f"Loading batch results from: {batch_output_file}")
    batch_results = load_batch_results(batch_output_file)
    print(f"Loaded {len(batch_results)} batch results")
    print()

    # Create response model
    response_model_class = create_response_model()
    print("Created Pydantic response model")
    print()

    # Analyze results
    print("Analyzing batch results...")
    report = analyze_batch_results(
        batch_results, response_model_class, max_failures_to_show=20
    )

    # Print report
    print_report(report)

    # Save report to file
    output_file = Path("json_parsing_diagnostic_report.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {output_file}")


if __name__ == "__main__":
    main()
