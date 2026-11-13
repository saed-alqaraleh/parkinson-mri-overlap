"""Compatibility command for the released patient-level analysis.

The archived project command is retained for existing workflows; all required
prediction evidence is included in the repository by default.
"""
from scripts.reproduce_patient_analysis import main


if __name__ == "__main__":
    main()
