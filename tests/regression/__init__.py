"""Regression tests that lock current reported numbers.

These tests load the JSON result artifacts produced by the existing evaluate/
scripts and assert that their headline numbers are unchanged. They do NOT
re-run inference; they are a cheap CI gate that catches accidental edits to
`evaluate/results/*.json`.
"""
