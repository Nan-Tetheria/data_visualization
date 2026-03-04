#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Tuple

MOTOR_PULLEY_RADIUS = 9.000  # mm


# All coeffs are in mm/radian.
@dataclass
class FingerCoeffs:
    mcp_flex_coeff: float = 12.4912
    pip_coeff: float = 7.3211
    dip_coeff: float = 9.0000


@dataclass
class ThumbFlexCoeffs:
    cmc_abd_coeff: float = 2.5000
    cmc_flex_coeff: float = 12.4931


@dataclass
class ThumbIPCoeffs:
    cmc_abd_coeff: float = 2.5000
    cmc_flex_coeff: float = 2.5000
    mcp_coeff: float = 9.4372
    ip_coeff: float = 12.5000


class ActuationsToJointsModel:
    """
    Inverse model: convert actuator movements to joint positions.

    Assumptions:
      - For each finger: mcp_flex = pip = dip = q (one-DoF finger approximation).
      - For thumb: to make 3 actuations -> 4 joints solvable, assume mcp = ip (one scalar),
        and solve cmc_flex + (mcp=ip) from the two tendon equations.
    """

    def __init__(self) -> None:
        self.finger_coeffs = FingerCoeffs()
        self.thumb_flex_coeffs = ThumbFlexCoeffs()
        self.thumb_ip_coeffs = ThumbIPCoeffs()

    def finger_joints(self, actuation: float) -> Tuple[float, float, float]:
        """
        Given a finger tendon actuation (motor angle, rad), return (mcp_flex, pip, dip) in rad,
        under the assumption mcp_flex = pip = dip.
        """
        denom = (
            self.finger_coeffs.mcp_flex_coeff
            + self.finger_coeffs.pip_coeff
            + self.finger_coeffs.dip_coeff
        )
        if denom == 0.0:
            raise ZeroDivisionError("Finger coefficient sum is zero; cannot invert.")

        q = actuation * MOTOR_PULLEY_RADIUS / denom
        return q, q, q

    def thumb_joints(
        self, abd_act: float, cmc_flex_act: float, tendon_act: float
    ) -> Tuple[float, float, float, float]:
        """
        Given thumb actuations (abd motor, cmc_flex motor, tendon motor), return
        (cmc_abd, cmc_flex, mcp, ip) in rad.

        Inversion details:
          - cmc_abd = abd_act (direct mapping)
          - cmc_flex_act * R = abd_coeff*cmc_abd + flex_coeff*cmc_flex
          - tendon_act * R = abd_coeff*cmc_abd - flex_coeff*cmc_flex + mcp_coeff*mcp + ip_coeff*ip
          - Assume mcp = ip = m to make it solvable.
        """
        cmc_abd = abd_act

        # Solve cmc_flex from the CMC flex tendon equation
        flex_coeff = self.thumb_flex_coeffs.cmc_flex_coeff
        abd_coeff_flex = self.thumb_flex_coeffs.cmc_abd_coeff
        if flex_coeff == 0.0:
            raise ZeroDivisionError("thumb_flex_coeffs.cmc_flex_coeff is zero; cannot invert.")

        cmc_flex = (cmc_flex_act * MOTOR_PULLEY_RADIUS - abd_coeff_flex * cmc_abd) / flex_coeff

        # Solve mcp=ip=m from the IP tendon equation
        abd_coeff_ip = self.thumb_ip_coeffs.cmc_abd_coeff
        flex_coeff_ip = self.thumb_ip_coeffs.cmc_flex_coeff
        mcp_coeff = self.thumb_ip_coeffs.mcp_coeff
        ip_coeff = self.thumb_ip_coeffs.ip_coeff

        denom = mcp_coeff + ip_coeff
        if denom == 0.0:
            raise ZeroDivisionError("thumb_ip_coeffs.mcp_coeff + ip_coeff is zero; cannot invert.")

        # tendon_act * R = abd_coeff*abd - flex_coeff*flex + (mcp_coeff+ip_coeff)*m
        m = (
            tendon_act * MOTOR_PULLEY_RADIUS
            - abd_coeff_ip * cmc_abd
            + flex_coeff_ip * cmc_flex
        ) / denom

        mcp = m
        ip = m
        return cmc_abd, cmc_flex, mcp, ip

    def hand_joints(self, actuations: List[float]) -> List[float]:
        """
        Convert 7 actuator movements to 16 joint positions (all in rad).
        Actuations order must match JointsToActuationsModel.hand_actuations():
          [thumb_cmc_abd_act, thumb_cmc_flex_act, thumb_tendon_act, finger0, finger1, finger2, finger3]

        Returns joint_positions order matching your original:
          [thumb: cmc_abd, cmc_flex, mcp, ip, then 4 fingers each: (mcp_flex, pip, dip)]
        """
        if len(actuations) != 7:
            raise ValueError(f"Expected 7 actuations, got {len(actuations)}")

        joints: List[float] = []

        # Thumb (4 joints)
        joints += list(self.thumb_joints(actuations[0], actuations[1], actuations[2]))

        # Four fingers (12 joints)
        for i in range(4):
            joints += list(self.finger_joints(actuations[3 + i]))

        if len(joints) != 16:
            raise RuntimeError(f"Internal error: expected 16 joints, got {len(joints)}")

        return joints