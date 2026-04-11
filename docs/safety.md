"""
Safety Guidelines and Interlocks
=================================

## Critical Safety Rules (Non-Negotiable)

This system **must not** be used for real pesticide deployment without explicit
regulatory approval and proper field testing infrastructure.

### Rule 1: Test Fluid Only
- All spray demonstrations use **water or harmless colored test fluid ONLY**
- `SAFE_TEST_FLUID_ONLY=true` must be set in `.env` for any hardware actuation
- Never load real pesticides into the system without explicit field protocol approval

### Rule 2: Dry Run Mode
- Spray actuation disabled by default unless explicitly enabled
- `DRY_RUN=true` by default in `.env`
- Physical spraying only possible in controlled test environment with supervisor present

### Rule 3: Deterministic Decisions
- No spray decision based solely on LLM or black-box model output
- All actuation must pass through deterministic rule-based validation
- Every decision logged with confidence, thresholds, and reason codes

### Rule 4: Human Review Required
- Operator must review prescription maps before any spray
- Human approval step cannot be skipped in production mode
- Dashboard shows why each zone was flagged for treatment

### Rule 5: Audit Trail
- Complete logging of:
  - Input image and mission ID
  - Detection model version and thresholds
  - Confidence scores and uncertainty
  - Environmental conditions
  - Prescription reasoning
  - Actuation events and timing
  - All errors and warnings
- Logs persisted in `outputs/logs/`

---

## Safety Interlock Architecture

```
Application Start
    ↓
[Load Config]
    ↓
[Check DRY_RUN and TEST_FLUID_ONLY flags]
    ↓
DRY_RUN=true? ─No→ [Warn and disable actuation anyway]
    ↓
TEST_FLUID_ONLY=true? ─No→ [Warn and disable actuation anyway]
    ↓
[Initialize Safe Mode]
    ↓
Detection Pipeline (Always OK)
    ↓
Prescription Generation (Always OK)
    ↓
Actuation Planning
    ↓
[Safety Validation]
    ├─ DRY_RUN enabled? ✓
    ├─ TEST_FLUID_ONLY enabled? ✓
    ├─ Human approval granted? ✓
    └─ Config checksum valid? ✓
    ↓
Pass All? ─No→ [Block actuation, log error]
    ↓
Yes→ [Execute with logging]
```

---

## Configuration-Based Safety

### .env Safety Settings

```env
# MANDATORY SAFETY FLAGS
DRY_RUN=true                    # Always start as true
SAFE_TEST_FLUID_ONLY=true      # Always true unless explicitly changed

# If either is false:
# - Warnings are logged
# - Actuation is still BLOCKED
# - Requires code change to deploy to hardware
```

### Prescription Safety Rules

In `configs/prescription.yaml`:

```yaml
prescription:
  safety:
    dry_run: true                    # Require dry-run mode
    test_fluid_only: true            # Require test fluid flag
    require_human_review: true       # Operator approval required
    fail_safe: true                  # Default to safe state
```

---

## Decision Chain with Safety Validation

Every prescription decision follows this sequence:

```
1. Detection Model
   └─ Input: Raw image
   └─ Output: Hotspot detections with confidence
   └─ Logging: Image ID, model version, detections count

2. Postprocessing
   └─ Remove overlaps, small artifacts
   └─ Logging: Filtering reason codes

3. Environmental Fusion
   └─ Attach temperature, humidity, wind context
   └─ Adjust severity based on conditions
   └─ Logging: Environmental modifiers applied

4. Prescription Engine (Deterministic)
   └─ Apply thresholds to severity scores
   └─ Compute spray rates
   └─ Logging: Decision rules applied, reason codes

5. Safety Validation
   ├─ Check DRY_RUN mode: Must be true to spray
   ├─ Check TEST_FLUID_ONLY: Must be true to spray
   ├─ Check human approval: Must be explicit
   ├─ Check mission state: Valid mission ID
   ├─ Check configuration: Version mismatches
   └─ Logging: All checks performed

6. Actuation Plan Creation
   └─ Create ActuationPlan with:
      - spray_zones: GridCell list
      - dry_run flag
      - test_fluid_only flag
      - safety_checks_passed: boolean
      - approval_status: pending/approved/rejected/executed

7. Human Review (Dashboard)
   └─ Operator reviews map
   └─ Sees reason for each recommendation
   └─ Can reject or modify zones
   └─ Approves plan before execution
   └─ Logging: Reviewer, timestamp, approval time

8. Actuation Execution
   ├─ Verify all safety checks passed
   ├─ Check DRY_RUN and TEST_FLUID_ONLY flags again
   ├─ Physically actuate pump/valve
   ├─ Log every spray event
   └─ Emergency stop button always active
```

---

## Code Safety Patterns

### Do NOT Do This

```python
# ❌ WRONG: Direct LLM spray control
response = llm.ask("Should we spray here?")
if "yes" in response.lower():
    sprayer.spray()  # UNSAFE - No determinism!

# ❌ WRONG: No logging
sprayer.spray(pump_rate)

# ❌ WRONG: Hardcoded thresholds
if confidence > 0.5:  # Changing this requires code edit!
    sprayer.spray()

# ❌ WRONG: Skipping safety checks
if dry_run == False:  # Should be ==, and should still check twice
    sprayer.spray()
```

### Do This Instead

```python
# ✅ CORRECT: Deterministic rule-based
severity_score = compute_severity(detection_confidence)
if severity_score >= config.get("prescription.thresholds.high"):
    cell.recommended_action = "spray"
    cell.reason_codes.append("high_severity")
    logger.info(f"Recommended spray for cell {cell.id}: {cell.reason_codes}")

# ✅ CORRECT: Explicit safety check
def can_actuate(mission_id: str) -> bool:
    dry_run = config.get_env().dry_run
    test_fluid_only = config.get_env().safe_test_fluid_only

    if not dry_run:
        logger.error("DRY_RUN not enabled - blocking actuation")
        return False
    if not test_fluid_only:
        logger.error("TEST_FLUID_ONLY not enabled - blocking actuation")
        return False
    return True

# ✅ CORRECT: Human approval required
if actuation_plan.approval_status != "approved":
    logger.warning("Actuation plan not approved - skipping spray")
    return

# ✅ CORRECT: Comprehensive logging
event = ActuationEvent(
    mission_id=mission_id,
    status=ActuationStatus.SPRAYING,
    pump_rate=pump_rate,
    dry_run=dry_run,
    safety_approved=True,
    reason_codes=["high_severity", "optimal_conditions"],
)
actuation_log.add_event(event)
```

---

## Emergency Procedures

### If Unexpected Spray Occurs

1. **Immediately activate Emergency Stop** (GPIO pin 22 or manual button)
2. Record the time and mission ID
3. Check logs: `tail -f outputs/logs/agridrone.log`
4. Look for safety flag status
5. Review what decision was made
6. Report to safety supervisor
7. Do NOT resume until root cause identified

### If Hardware Malfunction

1. Kill the process: `pkill -f agridrone`
2. Manually verify sprayer is OFF
3. Check GPIO states: `gpio readall` (Raspberry Pi)
4. Review recent events log
5. Run diagnostics before retry

---

## Field Deployment Checklist

Before any outdoor testing:

- [ ] `DRY_RUN=true` in `.env`
- [ ] `SAFE_TEST_FLUID_ONLY=true` in `.env`
- [ ] Safety supervisor present and trained
- [ ] Emergency stop button accessible
- [ ] Test fluid loaded (colored water for visibility)
- [ ] Pressure relief valve set correctly
- [ ] All logs accessible for review
- [ ] Operator trained on dashboard UI
- [ ] Safety procedures briefing completed
- [ ] Backup manual spray control available

---

## Research Ethics Commitment

This system is designed for **research purposes only** to advance site-specific crop protection technology.
We commit to:

- Never marketing this as production-ready pesticide automation
- Obtaining proper regulatory approval before field testing
- Prioritizing safety and environmental responsibility
- Publishing results transparently
- Supporting other researchers building on this work
- Maintaining open discussion of limitations and risks

---

## Questions or Safety Concerns?

If you identify a safety issue or have concerns:

1. Document the issue clearly
2. Email the research team immediately
3. Do NOT continue operation if uncertain
4. Follow local agricultural and environmental regulations

Safety first, research second. Always.
