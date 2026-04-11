"""
safety.py - Safety checks and interlocks for spray actuation.
"""
from loguru import logger

from ..types import ActuationPlan, SafetyCheckResult, SafetyReport, SafetyState


class SafetyChecker:
    """Verify safety conditions before actuation."""

    def check_actuation_safety(
        self, actuation_plan: ActuationPlan, dry_run: bool = True, test_fluid_only: bool = True
    ) -> SafetyReport:
        """
        Perform comprehensive safety checks.

        Args:
            actuation_plan: Plan to check
            dry_run: Require DRY_RUN mode
            test_fluid_only: Require TEST_FLUID_ONLY mode

        Returns:
            SafetyReport with all check results
        """
        report = SafetyReport(mission_id=actuation_plan.mission_id)

        # Check 1: DRY RUN mode
        if dry_run:
            check = SafetyCheckResult(
                check_name="dry_run_enabled",
                passed=actuation_plan.dry_run,
                severity="error" if not actuation_plan.dry_run else "info",
                message="DRY_RUN mode must be enabled" if not actuation_plan.dry_run else "DRY_RUN enabled",
            )
            report.add_check(check)
            report.dry_run_enabled = actuation_plan.dry_run

        # Check 2: TEST FLUID ONLY
        if test_fluid_only:
            check = SafetyCheckResult(
                check_name="test_fluid_only",
                passed=actuation_plan.test_fluid_only,
                severity="error" if not actuation_plan.test_fluid_only else "info",
                message="TEST_FLUID_ONLY must be enabled",
            )
            report.add_check(check)
            report.test_fluid_only_enabled = actuation_plan.test_fluid_only

        # Check 3: Human review required
        check = SafetyCheckResult(
            check_name="human_review",
            passed=actuation_plan.requires_human_review,
            severity="warning" if not actuation_plan.requires_human_review else "info",
            message="Human review required before actuation",
        )
        report.add_check(check)

        # Check 4: Approval status
        check = SafetyCheckResult(
            check_name="approval_status",
            passed=actuation_plan.approval_status == "approved",
            severity="error" if actuation_plan.approval_status != "approved" else "info",
            message=f"Plan status: {actuation_plan.approval_status}",
        )
        report.add_check(check)

        # Finalize report
        report.finalize()
        logger.info(f"Safety check complete: {report.overall_state}")

        return report

    def is_safe_to_actuate(self, report: SafetyReport) -> bool:
        """Check if report passes all safety checks."""
        return report.overall_state == SafetyState.SAFE and report.all_checks_passed
