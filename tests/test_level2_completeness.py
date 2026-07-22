"""SCI-003 invariants for multispectral product completeness."""

from __future__ import annotations

import pytest

from milgrau.level2.completeness import (
    Level2ProductContract,
    ProductCompleteness,
    ProductStatus,
    WavelengthFailureCode,
    WavelengthFailureDiagnostic,
    WavelengthFailureStage,
)


def _failure(wavelength_nm: int) -> WavelengthFailureDiagnostic:
    return WavelengthFailureDiagnostic(
        wavelength_nm=wavelength_nm,
        stage=WavelengthFailureStage.SELECTION_AND_BLOCKING,
        code=WavelengthFailureCode.CHANNEL_SELECTION_FAILED,
        message="synthetic missing channel",
        cause_summary="ValueError",
    )


def test_completeness_contract_canonicalizes_wavelength_order() -> None:
    contract = Level2ProductContract(
        requested_wavelengths=(532, 355),
        processed_wavelengths=(355,),
        failed_wavelengths=(532,),
        completeness=ProductCompleteness.PARTIAL,
        product_status=ProductStatus.PARTIAL_FAILURE,
        failure_diagnostics=(_failure(532),),
    )

    assert contract.requested_wavelengths == (355, 532)
    assert contract.processed_wavelengths == (355,)
    assert contract.failed_wavelengths == (532,)


@pytest.mark.parametrize(
    ("processed", "failed", "completeness", "status", "diagnostics", "message"),
    [
        (
            (355, 532),
            (532,),
            ProductCompleteness.PARTIAL,
            ProductStatus.PARTIAL_FAILURE,
            (_failure(532),),
            "disjoint",
        ),
        (
            (355,),
            (),
            ProductCompleteness.PARTIAL,
            ProductStatus.PARTIAL_FAILURE,
            (),
            "equal requested",
        ),
        (
            (355,),
            (532,),
            ProductCompleteness.COMPLETE,
            ProductStatus.SUCCESS,
            (_failure(532),),
            "contradict",
        ),
        (
            (355,),
            (532,),
            ProductCompleteness.PARTIAL,
            ProductStatus.SUCCESS,
            (_failure(532),),
            "contradict",
        ),
        (
            (355,),
            (532,),
            ProductCompleteness.PARTIAL,
            ProductStatus.PARTIAL_FAILURE,
            (_failure(355),),
            "diagnostics",
        ),
    ],
)
def test_completeness_contract_rejects_contradictory_states(
    processed: tuple[int, ...],
    failed: tuple[int, ...],
    completeness: ProductCompleteness,
    status: ProductStatus,
    diagnostics: tuple[WavelengthFailureDiagnostic, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Level2ProductContract(
            requested_wavelengths=(355, 532),
            processed_wavelengths=processed,
            failed_wavelengths=failed,
            completeness=completeness,
            product_status=status,
            failure_diagnostics=diagnostics,
        )


def test_failed_contract_is_representable_but_cannot_validate_for_publication() -> None:
    contract = Level2ProductContract(
        requested_wavelengths=(355, 532),
        processed_wavelengths=(),
        failed_wavelengths=(355, 532),
        completeness=ProductCompleteness.FAILED,
        product_status=ProductStatus.FAILURE,
        failure_diagnostics=(_failure(355), _failure(532)),
    )

    with pytest.raises(ValueError, match="cannot be published"):
        contract.validate_results([])
