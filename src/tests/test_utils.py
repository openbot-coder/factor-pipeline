"""Comprehensive tests for data/utils.py — norm_code and related utilities.

Covers:
- SSE codes (60xxxx, 68xxxx, 90xxxx)
- SZSE codes (00xxxx, 30xxxx, 20xxxx)
- BSE codes (43xxxx, 83xxxx, 87xxxx, 92xxxx)
- Unknown / unrecognized code prefixes
- Non-string inputs (int, float, None, bool)
- Boundary values (already-padded, short, long, empty strings)
- Whitespace handling
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from factor_pipeline.data.utils import norm_code


# =============================================================================
# SSE (Shanghai Stock Exchange) Tests
# =============================================================================


class TestNormCodeSSE:
    """Tests for Shanghai Stock Exchange code normalization."""

    @pytest.mark.parametrize(
        "code, expected",
        [
            ("600000", "600000.SSE"),
            ("601398", "601398.SSE"),
            ("688001", "688001.SSE"),
            ("688981", "688981.SSE"),
            ("900901", "900901.SSE"),  # B-share
        ],
        ids=["60xxxx", "601xxx", "68xxxx", "688xxx", "90xxxx-bshare"],
    )
    def test_sse_codes(self, code, expected):
        """Positive: All SSE prefix codes map to .SSE."""
        assert norm_code(code) == expected

    def test_sse_needs_padding(self):
        """Edge: Short SSE code gets zero-padded, changing prefix to '00' → SZSE.

        NOTE: zfill(6) pads with leading zeros, so "600" → "000600",
        which starts with "00" and becomes SZSE. This is a known quirk
        of the implementation — users must pass 6-digit codes.
        """
        assert norm_code("600") == "000600.SZSE"

    def test_sse_already_padded(self):
        """Boundary: Already 6-digit SSE code is returned as-is with suffix."""
        assert norm_code("600000") == "600000.SSE"


# =============================================================================
# SZSE (Shenzhen Stock Exchange) Tests
# =============================================================================


class TestNormCodeSZSE:
    """Tests for Shenzhen Stock Exchange code normalization."""

    @pytest.mark.parametrize(
        "code, expected",
        [
            ("000001", "000001.SZSE"),
            ("002594", "002594.SZSE"),
            ("300750", "300750.SZSE"),  # ChiNext
            ("301269", "301269.SZSE"),
            ("200002", "200002.SZSE"),  # B-share
        ],
        ids=["000xxx", "002xxx", "300xxx-chi-next", "301xxx", "20xxxx-bshare"],
    )
    def test_szse_codes(self, code, expected):
        """Positive: All SZSE prefix codes map to .SZSE."""
        assert norm_code(code) == expected

    def test_szse_needs_padding(self):
        """Positive: Short SZSE code gets zero-padded."""
        assert norm_code("1") == "000001.SZSE"

    def test_szse_zero_padded_input(self):
        """Boundary: Input string with leading zeros already correct."""
        assert norm_code("000001") == "000001.SZSE"


# =============================================================================
# BSE (Beijing Stock Exchange) Tests
# =============================================================================


class TestNormCodeBSE:
    """Tests for Beijing Stock Exchange code normalization."""

    @pytest.mark.parametrize(
        "code, expected",
        [
            ("430047", "430047.BSE"),
            ("830779", "830779.BSE"),
            ("870299", "870299.BSE"),
            ("920066", "920066.BSE"),
        ],
        ids=["43xxxx", "83xxxx", "87xxxx", "92xxxx"],
    )
    def test_bse_codes(self, code, expected):
        """Positive: All BSE prefix codes map to .BSE."""
        assert norm_code(code) == expected

    def test_bse_needs_padding(self):
        """Edge: Short BSE code gets zero-padded, changing prefix to '00' → SZSE.

        NOTE: zfill(6) pads with leading zeros, so "430" → "000430",
        which starts with "00" and becomes SZSE.
        """
        assert norm_code("430") == "000430.SZSE"


# =============================================================================
# Unknown / Unrecognized Codes
# =============================================================================


class TestNormCodeUnknown:
    """Tests for unrecognized code prefixes returning None."""

    @pytest.mark.parametrize(
        "code",
        ["110000", "990001", "500000", "010000", "770000"],
        ids=["11xxxx", "99xxxx", "50xxxx", "01xxxx", "77xxxx"],
    )
    def test_unknown_prefix_returns_none(self, code):
        """Negative: Unrecognized prefixes return None."""
        assert norm_code(code) is None

    def test_borderline_43_is_bse(self):
        """Boundary: 43 is the lowest BSE prefix — maps to BSE."""
        assert norm_code("430000") == "430000.BSE"

    def test_borderline_39_returns_none(self):
        """Boundary: 39 is not a recognized prefix — returns None."""
        assert norm_code("390000") is None

    def test_borderline_91_returns_none(self):
        """Boundary: 91 is not a recognized prefix — returns None."""
        assert norm_code("910000") is None

    def test_borderline_60_is_sse(self):
        """Boundary: 60 is the lowest SSE prefix — maps to SSE."""
        assert norm_code("600000") == "600000.SSE"

    def test_borderline_30_is_szse(self):
        """Boundary: 30 is a recognized SZSE prefix."""
        assert norm_code("300000") == "300000.SZSE"

    def test_borderline_20_is_szse(self):
        """Boundary: 20 is a recognized SZSE prefix."""
        assert norm_code("200000") == "200000.SZSE"

    def test_borderline_10_returns_none(self):
        """Boundary: 10 is not a recognized prefix."""
        assert norm_code("100000") is None


# =============================================================================
# Non-String Input Tests
# =============================================================================


class TestNormCodeNonString:
    """Tests for non-string inputs — norm_code calls str() internally."""

    def test_integer_input(self):
        """Positive: Integer input is converted via str()."""
        assert norm_code(600000) == "600000.SSE"

    def test_integer_short(self):
        """Positive: Integer shorter than 6 digits gets padded."""
        assert norm_code(1) == "000001.SZSE"

    def test_integer_zero(self):
        """Boundary: Integer 0 — str(0).zfill(6) = '000000' → starts with '00' → SZSE."""
        assert norm_code(0) == "000000.SZSE"

    def test_float_input(self):
        """Boundary: Float input is stringified then processed.
        str(600000.0) = '600000.0' → starts with '60' → SSE.
        """
        assert norm_code(600000.0) == "600000.0.SSE"

    def test_none_input(self):
        """Boundary: None input — str(None) = 'None' → zfill(6) = '000None'."""
        # str(None) gives 'None', which .zfill(6) pads to '000None'.
        # This starts with '00', mapping to SZSE.
        result = norm_code(None)
        assert result is not None
        assert ".SZSE" in result

    def test_bool_input(self):
        """Boundary: bool input — str(True) = 'True' → zfill(6) = '000True' → starts with '00'."""
        # This documents current behavior: booleans are stringified.
        result = norm_code(True)
        assert result is not None  # '000True' starts with '00'
        assert ".SZSE" in result

    def test_empty_string(self):
        """Boundary: Empty string gets zero-padded to '000000' → SZSE."""
        assert norm_code("") == "000000.SZSE"

    def test_whitespace_only_string(self):
        """Boundary: Whitespace-only string is stripped then zero-padded."""
        assert norm_code("   ") == "000000.SZSE"


# =============================================================================
# Whitespace and Formatting Edge Cases
# =============================================================================


class TestNormCodeWhitespace:
    """Tests for leading/trailing whitespace handling."""

    def test_leading_whitespace(self):
        """Boundary: Leading whitespace is stripped."""
        assert norm_code("  600000") == "600000.SSE"

    def test_trailing_whitespace(self):
        """Boundary: Trailing whitespace is stripped."""
        assert norm_code("600000  ") == "600000.SSE"

    def test_both_whitespace(self):
        """Boundary: Both leading and trailing whitespace stripped."""
        assert norm_code("  600000  ") == "600000.SSE"

    def test_tab_whitespace(self):
        """Boundary: Tab characters are stripped."""
        assert norm_code("\t600000\t") == "600000.SSE"


# =============================================================================
# Length Edge Cases
# =============================================================================


class TestNormCodeLength:
    """Tests for varying code lengths."""

    def test_single_digit(self):
        """Boundary: Single digit code gets padded to 6 digits."""
        assert norm_code("1") == "000001.SZSE"

    def test_three_digit(self):
        """Boundary: Three digit code gets padded, becoming SZSE.

        "600.zfill(6)" = "000600" → starts with '00' → SZSE.
        """
        assert norm_code("600") == "000600.SZSE"

    def test_exactly_six_digits(self):
        """Boundary: Exactly 6 digits — no padding needed."""
        assert norm_code("600000") == "600000.SSE"

    def test_seven_digits(self):
        """Boundary: 7-digit code — zfill(6) is a no-op since len >= 6.
        '1600000' doesn't start with recognized prefix.
        """
        assert norm_code("1600000") is None

    def test_very_long_numeric(self):
        """Boundary: Very long numeric string still starts with prefix."""
        # "600000123456" starts with "60" → SSE
        assert norm_code("600000123456") == "600000123456.SSE"

    def test_all_zeros(self):
        """Boundary: '000000' starts with '00' → SZSE."""
        assert norm_code("000000") == "000000.SZSE"


# =============================================================================
# Prefix Priority Edge Cases
# =============================================================================


class TestNormCodePriority:
    """Tests for prefix matching order/priority."""

    def test_30_takes_szse_over_unknown(self):
        """Positive: 30xxxx matches the SZSE '30' prefix, not an unknown."""
        assert norm_code("300000") == "300000.SZSE"

    def test_90_takes_sse_over_unknown(self):
        """Positive: 90xxxx matches the SSE '90' prefix."""
        assert norm_code("900000") == "900000.SSE"

    def test_87_takes_bse_over_unknown(self):
        """Positive: 87xxxx matches the BSE '87' prefix."""
        assert norm_code("870000") == "870000.BSE"

    def test_20_takes_szse_not_bse(self):
        """Positive: 20xxxx matches SZSE '20', not BSE."""
        assert norm_code("200000") == "200000.SZSE"
