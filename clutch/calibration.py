import math
from dataclasses import dataclass

from .inputs import InputError, Policy


@dataclass(frozen=True)
class StageCalibration:
    championship_final: float = 90.0
    direct_elimination: float = 82.0
    second_chance_qualifier: float = 70.0
    live_survival: float = 72.0
    material_consequence: float = 58.0
    ordinary_live: float = 44.0
    objective_settled: float = 22.0
    series_decider_baseline_discount: float = 0.07
    prestige_curvature: float = 0.5
    modifier_cap: float = 8.0
    opposition_strength_limit: float = 1.5
    head_to_head_limit: float = 0.5
    qualification_aware: bool = True
    late_qualification: float = 80.0
    playoff_aware: bool = True
    eliminator_bonus: float = 2.0
    final_berth_bonus: float = 3.0
    ipl_practical_points_target: int = 14

    def __post_init__(self):
        anchors = [
            self.objective_settled, self.ordinary_live, self.material_consequence,
            self.second_chance_qualifier, self.live_survival, self.late_qualification, self.direct_elimination,
            self.championship_final,
        ]
        if not all(isinstance(value, bool) for value in [self.qualification_aware, self.playoff_aware]):
            raise InputError("Calibration feature switches must be booleans.")
        if not self.qualification_aware or not self.playoff_aware:
            raise InputError("The current model requires qualification and playoff context.")
        if any(not math.isfinite(value) for value in self.__dict__.values()):
            raise InputError("Calibration parameters must be finite.")
        if any(not 0 <= value <= 100 for value in anchors):
            raise InputError("Calibration anchors must be within 0-100.")
        if anchors != sorted(set(anchors)):
            raise InputError("Calibration event anchors must be strictly increasing.")
        if not 0 <= self.eliminator_bonus <= self.final_berth_bonus:
            raise InputError("Knockout progression bonuses must be nonnegative and ordered.")
        if self.direct_elimination + self.final_berth_bonus >= self.championship_final:
            raise InputError("Final-berth baseline must remain below the championship-final baseline.")
        if type(self.ipl_practical_points_target) is not int or self.ipl_practical_points_target <= 0:
            raise InputError("The practical IPL target must be a positive integer, not an official cutoff.")
        if not 0 <= self.series_decider_baseline_discount < 1:
            raise InputError("Series-decider baseline discount must be in [0,1).")
        limits = [self.modifier_cap, self.opposition_strength_limit, self.head_to_head_limit]
        if not 0 <= self.prestige_curvature <= 1 or any(value < 0 for value in limits):
            raise InputError("Invalid prestige curve or modifier cap.")
        if self.championship_final + sum(limits) > 100:
            raise InputError("Team calibration would exceed 100 before individual adjustments.")
        minimum_base = min(
            self.objective_settled * (1 - self.prestige_curvature),
            self.direct_elimination * (1 - self.series_decider_baseline_discount))
        if self.qualification_aware:
            minimum_base = min(minimum_base, (
                self.live_survival - (self.direct_elimination - self.late_qualification)
            ) * (1 - self.prestige_curvature))
        if minimum_base < self.opposition_strength_limit + self.head_to_head_limit:
            raise InputError("Opposition adjustments could produce a negative team score.")

    def prestige(self, importance: float) -> float:
        return 1 - self.prestige_curvature * (1 - importance) ** 2

    @property
    def version(self) -> str:
        return "CLUTCH"


@dataclass(frozen=True)
class CalibratedScore:
    model: str
    tier: str
    baseline: float | None
    expectation_bonus: float
    form_bonus: float
    rivalry_bonus: float
    opposition_strength_points: float
    head_to_head_points: float
    score: float | None
    known_lower: float
    possible_upper: float
    practical_qualification_status: str = "not_applied"


def calibrate(
    policy: Policy, importance: float, stakes: float | None, expectation: float,
    form: float, rivalry: float, family: str, phase: str, reason: str,
    calibration: StageCalibration | None,
    *, opposition_strength: float = 0.0, head_to_head: float = 0.0,
    qualification_context: str = "",
    max_qualification_points: int | None = None,
    qualification_win_path_guaranteed: bool = False,
) -> CalibratedScore:
    for label, value in [("importance", importance), ("expectation", expectation),
                         ("form", form), ("rivalry", rivalry)]:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise InputError(f"Invalid {label} for calibration.")
    if stakes is not None and (not math.isfinite(stakes) or not 0 <= stakes <= 1):
        raise InputError("Invalid stakes for calibration.")
    for label, value in [("opposition_strength", opposition_strength), ("head_to_head", head_to_head)]:
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise InputError(f"Invalid {label} for calibration.")
    if qualification_context not in {
        "", "loss_eliminates", "late_loss_of_control", "late_external_dependency",
    }:
        raise InputError(f"Unknown qualification context: {qualification_context}")
    if max_qualification_points is not None and (
        type(max_qualification_points) is not int or max_qualification_points < 0
    ):
        raise InputError("Maximum qualification points must be a nonnegative integer.")
    if not isinstance(qualification_win_path_guaranteed, bool):
        raise InputError("Qualification win-path evidence must be a boolean.")
    weights = policy.weights
    if calibration is None:
        raise InputError("A current stage calibration is required.")
    components = {
        "expectation_points": weights["expectation_weight"] * expectation,
        "form_points": weights["recent_underperformance_weight"] * form,
        "rivalry_points": weights["rivalry_weight"] * rivalry,
    }

    total_modifier_weight = sum(weights[key] for key in [
        "expectation_weight", "recent_underperformance_weight", "rivalry_weight"])
    scale = calibration.modifier_cap / total_modifier_weight if total_modifier_weight else 0
    expectation_bonus = scale * components["expectation_points"]
    form_bonus = scale * components["form_points"]
    rivalry_bonus = scale * components["rivalry_points"]
    opposition_points = calibration.opposition_strength_limit * opposition_strength
    head_to_head_points = calibration.head_to_head_limit * head_to_head
    bonus = expectation_bonus + form_bonus + rivalry_bonus + opposition_points + head_to_head_points
    prestige = calibration.prestige(importance)

    if stakes is None:
        max_base = max(calibration.championship_final * prestige,
                       calibration.direct_elimination * (1 - calibration.series_decider_baseline_discount))
        return CalibratedScore(
            calibration.version, "unresolved", None, expectation_bonus, form_bonus, rivalry_bonus,
            opposition_points, head_to_head_points, None, max(0, bonus),
            min(100, max_base + bonus),
        )
    practical_status = "not_applied"
    if calibration.playoff_aware and family == "ipl" and phase == "league":
        if max_qualification_points is None:
            raise InputError("IPL practical-target scoring requires a pre-match maximum-points bound.")
        if reason.startswith("top_two_second_chance") or reason == "playoff_qualification_already_secured":
            practical_status = "qualification_secured"
        elif reason == "playoff_qualification_unreachable":
            practical_status = "mathematically_eliminated"
        elif not reason.startswith("playoff_qualification_"):
            raise InputError(f"Unexpected IPL league qualification reason: {reason}")
        elif max_qualification_points < calibration.ipl_practical_points_target:
            practical_status = (
                "controlled_path_below_target" if qualification_win_path_guaranteed
                else "below_practical_target")
        elif max_qualification_points - 2 < calibration.ipl_practical_points_target:
            practical_status = "must_win_for_practical_target"
        else:
            practical_status = "practical_target_reachable"
    series_decider = (
        reason == "series_decider"
        or (family == "senior_tri_series" and (phase == "Final" or stakes == 1))
    )
    if practical_status == "below_practical_target":
        tier = "below_practical_target"
        baseline = calibration.ordinary_live * prestige
    elif practical_status == "must_win_for_practical_target":
        tier = ("qualification_elimination" if qualification_context == "loss_eliminates"
                else "practical_target_must_win")
        baseline = calibration.direct_elimination * prestige
    elif series_decider:
        # Directly anchored to IPL elimination; do not discount the series twice.
        tier = "series_decider"
        baseline = calibration.direct_elimination * (1 - calibration.series_decider_baseline_discount)
    elif phase == "Final":
        tier = "championship_final"
        baseline = calibration.championship_final * prestige
    elif phase in {"Semi Final", "Eliminator", "Qualifier 2"}:
        tier = "direct_elimination"
        proximity = 0.0
        if calibration.playoff_aware:
            if phase == "Eliminator":
                tier, proximity = "playoff_eliminator", calibration.eliminator_bonus
            else:
                tier, proximity = "final_berth_knockout", calibration.final_berth_bonus
        baseline = (calibration.direct_elimination + proximity) * prestige
    elif phase == "Qualifier 1":
        tier = "second_chance_qualifier"
        baseline = calibration.second_chance_qualifier * prestige
    elif calibration.qualification_aware and qualification_context:
        elimination = (calibration.live_survival if family == "senior_tri_series"
                       else calibration.direct_elimination)
        if qualification_context == "loss_eliminates":
            tier = "qualification_elimination"
            baseline = elimination * prestige
        else:
            tier = "late_qualification"
            baseline = (elimination - (calibration.direct_elimination - calibration.late_qualification)) * prestige
    elif stakes == 0.9:
        tier = "live_survival"
        baseline = calibration.live_survival * prestige
    elif stakes == 0.6:
        tier = "material_consequence"
        baseline = calibration.material_consequence * prestige
    elif stakes == 0.35:
        tier = "ordinary_live"
        baseline = calibration.ordinary_live * prestige
    elif stakes == 0.1:
        tier = "objective_settled"
        baseline = calibration.objective_settled * prestige
    else:
        raise InputError(f"No {calibration.version} tier for {family}/{phase}/{reason}/{stakes}")
    score = baseline + bonus
    if not 0 <= score <= 100:
        raise InputError("Calibrated team score is outside 0-100.")
    return CalibratedScore(
        calibration.version, tier, baseline, expectation_bonus, form_bonus, rivalry_bonus,
        opposition_points, head_to_head_points, score, score, score,
        practical_qualification_status=practical_status,
    )
