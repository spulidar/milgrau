"""Explicit multispectral completeness and per-wavelength failure contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any, Iterable

from milgrau.level2.contracts import WavelengthRetrievalResult


class ProductCompleteness(StrEnum):
    """Scientific completeness of one requested multispectral Level 2 product."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class ProductStatus(StrEnum):
    """Scientific/product outcome, distinct from the generic execution status."""

    SUCCESS = "success"
    PARTIAL_FAILURE = "partial_failure"
    FAILURE = "failure"


class WavelengthAttemptStatus(StrEnum):
    """Outcome of one isolated wavelength attempt."""

    SUCCESS = "success"
    RECOVERABLE_FAILURE = "recoverable_failure"
    FATAL_FAILURE = "fatal_failure"


class WavelengthFailureStage(IntEnum):
    """Stable stages persisted for failed requested wavelengths."""

    UNKNOWN = 0
    SELECTION_AND_BLOCKING = 1
    GLUING = 2
    MOLECULAR_MODEL = 3
    RAYLEIGH_KFS = 4
    RESULT_ASSEMBLY = 5
    RETRIEVAL_VALIDATION = 6
    INTERNAL = 7


class WavelengthFailureCode(IntEnum):
    """Stable compact failure classes; Python exception names are not schema."""

    CHANNEL_SELECTION_FAILED = 1
    NO_VALID_SIGNAL = 2
    NO_VALID_RETRIEVAL_BLOCK = 3
    MOLECULAR_MODEL_FAILED = 4
    RAYLEIGH_FAILED = 5
    ELASTIC_INVERSION_FAILED = 6
    CONTRACT_VALIDATION_FAILED = 7
    INTERNAL_ERROR = 8


_RETRIEVAL_STAGE_MAP: dict[str, tuple[WavelengthFailureStage, WavelengthFailureCode]] = {
    "selection_and_blocking": (
        WavelengthFailureStage.SELECTION_AND_BLOCKING,
        WavelengthFailureCode.CHANNEL_SELECTION_FAILED,
    ),
    "gluing": (
        WavelengthFailureStage.GLUING,
        WavelengthFailureCode.NO_VALID_SIGNAL,
    ),
    "molecular_model": (
        WavelengthFailureStage.MOLECULAR_MODEL,
        WavelengthFailureCode.MOLECULAR_MODEL_FAILED,
    ),
    "rayleigh_kfs": (
        WavelengthFailureStage.RAYLEIGH_KFS,
        WavelengthFailureCode.ELASTIC_INVERSION_FAILED,
    ),
    "result_assembly": (
        WavelengthFailureStage.RESULT_ASSEMBLY,
        WavelengthFailureCode.CONTRACT_VALIDATION_FAILED,
    ),
}


def canonical_wavelengths(values: Iterable[int]) -> tuple[int, ...]:
    """Return positive wavelengths in deterministic ascending order without duplicates."""
    normalized: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            raise TypeError("Wavelengths must be positive integers, not booleans.")
        wavelength = int(value)
        if wavelength <= 0:
            raise ValueError("Wavelengths must be positive integers.")
        normalized.add(wavelength)
    if not normalized:
        raise ValueError("At least one requested wavelength is required.")
    return tuple(sorted(normalized))


def _short_text(value: object, *, limit: int = 240) -> str:
    text = " ".join(str(value).split()).strip()
    if not text:
        text = "No diagnostic message available"
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


@dataclass(frozen=True, slots=True)
class WavelengthFailureDiagnostic:
    """Compact stable diagnosis for one failed requested wavelength."""

    wavelength_nm: int
    stage: WavelengthFailureStage
    code: WavelengthFailureCode
    message: str
    cause_summary: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.wavelength_nm, bool) or int(self.wavelength_nm) <= 0:
            raise ValueError("wavelength_nm must be a positive integer.")
        if not isinstance(self.stage, WavelengthFailureStage):
            raise TypeError("stage must be WavelengthFailureStage.")
        if not isinstance(self.code, WavelengthFailureCode):
            raise TypeError("code must be WavelengthFailureCode.")
        object.__setattr__(self, "wavelength_nm", int(self.wavelength_nm))
        object.__setattr__(self, "message", _short_text(self.message))
        object.__setattr__(
            self,
            "cause_summary",
            "" if not str(self.cause_summary).strip() else _short_text(self.cause_summary),
        )


def diagnostic_from_exception(
    wavelength_nm: int,
    exc: Exception,
    *,
    retrieval_stage: str | None = None,
) -> WavelengthFailureDiagnostic:
    """Map one localized exception to stable stage/code fields."""
    stage, code = _RETRIEVAL_STAGE_MAP.get(
        str(retrieval_stage or ""),
        (WavelengthFailureStage.INTERNAL, WavelengthFailureCode.INTERNAL_ERROR),
    )
    return WavelengthFailureDiagnostic(
        wavelength_nm=int(wavelength_nm),
        stage=stage,
        code=code,
        message=str(exc),
        cause_summary=f"{type(exc).__module__}.{type(exc).__qualname__}",
    )


@dataclass(frozen=True, slots=True)
class WavelengthAttempt:
    """One isolated attempt: exactly one scientific result or one diagnosis."""

    wavelength_nm: int
    status: WavelengthAttemptStatus
    result: WavelengthRetrievalResult | None = None
    diagnostic: WavelengthFailureDiagnostic | None = None

    def __post_init__(self) -> None:
        wavelength = int(self.wavelength_nm)
        if isinstance(self.wavelength_nm, bool) or wavelength <= 0:
            raise ValueError("wavelength_nm must be a positive integer.")
        if not isinstance(self.status, WavelengthAttemptStatus):
            raise TypeError("status must be WavelengthAttemptStatus.")
        object.__setattr__(self, "wavelength_nm", wavelength)
        if self.status is WavelengthAttemptStatus.SUCCESS:
            if self.result is None or self.diagnostic is not None:
                raise ValueError("A successful wavelength attempt requires only a scientific result.")
            if int(self.result.wavelength_nm) != wavelength:
                raise ValueError("Attempt/result wavelength mismatch.")
        else:
            if self.result is not None or self.diagnostic is None:
                raise ValueError("A failed wavelength attempt requires only a diagnostic.")
            if self.diagnostic.wavelength_nm != wavelength:
                raise ValueError("Attempt/diagnostic wavelength mismatch.")

    @classmethod
    def success(cls, result: WavelengthRetrievalResult) -> WavelengthAttempt:
        return cls(
            wavelength_nm=int(result.wavelength_nm),
            status=WavelengthAttemptStatus.SUCCESS,
            result=result,
        )

    @classmethod
    def recoverable_failure(
        cls,
        diagnostic: WavelengthFailureDiagnostic,
    ) -> WavelengthAttempt:
        return cls(
            wavelength_nm=diagnostic.wavelength_nm,
            status=WavelengthAttemptStatus.RECOVERABLE_FAILURE,
            diagnostic=diagnostic,
        )

    @classmethod
    def fatal_failure(cls, diagnostic: WavelengthFailureDiagnostic) -> WavelengthAttempt:
        return cls(
            wavelength_nm=diagnostic.wavelength_nm,
            status=WavelengthAttemptStatus.FATAL_FAILURE,
            diagnostic=diagnostic,
        )


@dataclass(frozen=True, slots=True)
class Level2ProductContract:
    """Validated relationship between requested, processed and failed wavelengths."""

    requested_wavelengths: tuple[int, ...]
    processed_wavelengths: tuple[int, ...]
    failed_wavelengths: tuple[int, ...]
    completeness: ProductCompleteness
    product_status: ProductStatus
    failure_diagnostics: tuple[WavelengthFailureDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        requested = canonical_wavelengths(self.requested_wavelengths)
        processed = tuple(sorted(int(value) for value in self.processed_wavelengths))
        failed = tuple(sorted(int(value) for value in self.failed_wavelengths))
        if processed != tuple(dict.fromkeys(processed)) or failed != tuple(dict.fromkeys(failed)):
            raise ValueError("Processed and failed wavelength lists must not contain duplicates.")
        if any(value <= 0 for value in (*processed, *failed)):
            raise ValueError("Processed and failed wavelengths must be positive.")
        if set(processed) & set(failed):
            raise ValueError("processed_wavelengths and failed_wavelengths must be disjoint.")
        if set(processed) | set(failed) != set(requested):
            raise ValueError("Processed plus failed wavelengths must equal requested wavelengths.")
        if not isinstance(self.completeness, ProductCompleteness):
            raise TypeError("completeness must be ProductCompleteness.")
        if not isinstance(self.product_status, ProductStatus):
            raise TypeError("product_status must be ProductStatus.")

        diagnostics = tuple(sorted(self.failure_diagnostics, key=lambda item: item.wavelength_nm))
        if tuple(item.wavelength_nm for item in diagnostics) != failed:
            raise ValueError("Failure diagnostics must correspond exactly to failed_wavelengths.")
        expected = (
            (ProductCompleteness.COMPLETE, ProductStatus.SUCCESS)
            if processed and not failed
            else (
                (ProductCompleteness.PARTIAL, ProductStatus.PARTIAL_FAILURE)
                if processed and failed
                else (ProductCompleteness.FAILED, ProductStatus.FAILURE)
            )
        )
        if (self.completeness, self.product_status) != expected:
            raise ValueError(
                "Completeness/product_status contradict processed and failed wavelengths."
            )
        object.__setattr__(self, "requested_wavelengths", requested)
        object.__setattr__(self, "processed_wavelengths", processed)
        object.__setattr__(self, "failed_wavelengths", failed)
        object.__setattr__(self, "failure_diagnostics", diagnostics)

    @classmethod
    def from_attempts(
        cls,
        requested_wavelengths: Iterable[int],
        attempts: Iterable[WavelengthAttempt],
    ) -> Level2ProductContract:
        requested = canonical_wavelengths(requested_wavelengths)
        attempt_list = tuple(attempts)
        by_wavelength = {attempt.wavelength_nm: attempt for attempt in attempt_list}
        if len(by_wavelength) != len(attempt_list):
            raise ValueError("Each requested wavelength must be attempted exactly once.")
        if set(by_wavelength) != set(requested):
            raise ValueError("Wavelength attempts must correspond exactly to requested wavelengths.")
        if any(attempt.status is WavelengthAttemptStatus.FATAL_FAILURE for attempt in attempt_list):
            raise RuntimeError("A fatal/global wavelength attempt cannot be converted into a product.")
        processed = tuple(
            wavelength
            for wavelength in requested
            if by_wavelength[wavelength].status is WavelengthAttemptStatus.SUCCESS
        )
        failed = tuple(wavelength for wavelength in requested if wavelength not in processed)
        diagnostics = tuple(
            by_wavelength[wavelength].diagnostic
            for wavelength in failed
            if by_wavelength[wavelength].diagnostic is not None
        )
        completeness = (
            ProductCompleteness.COMPLETE
            if not failed
            else ProductCompleteness.PARTIAL if processed else ProductCompleteness.FAILED
        )
        status = (
            ProductStatus.SUCCESS
            if completeness is ProductCompleteness.COMPLETE
            else ProductStatus.PARTIAL_FAILURE
            if completeness is ProductCompleteness.PARTIAL
            else ProductStatus.FAILURE
        )
        return cls(
            requested_wavelengths=requested,
            processed_wavelengths=processed,
            failed_wavelengths=failed,
            completeness=completeness,
            product_status=status,
            failure_diagnostics=diagnostics,  # type: ignore[arg-type]
        )

    def validate_results(self, results: Iterable[WavelengthRetrievalResult]) -> None:
        result_wavelengths = tuple(sorted(int(result.wavelength_nm) for result in results))
        if result_wavelengths != self.processed_wavelengths:
            raise ValueError("Scientific result wavelengths must equal processed_wavelengths.")
        if self.completeness is ProductCompleteness.FAILED:
            raise ValueError("A failed product with no processed wavelength cannot be published.")
        for result in results:
            if not bool(result.optical.retrieval_success_flag.astype(bool).any()):
                raise ValueError(
                    f"Processed wavelength {result.wavelength_nm} has no valid optical retrieval block."
                )

    def provenance_result(self) -> dict[str, object]:
        """Return deterministic result metadata stored outside the expectation fingerprint."""
        return {
            "requested_wavelengths": list(self.requested_wavelengths),
            "processed_wavelengths": list(self.processed_wavelengths),
            "failed_wavelengths": list(self.failed_wavelengths),
            "product_completeness": self.completeness.value,
            "product_status": self.product_status.value,
            "failures": [
                {
                    "wavelength_nm": diagnostic.wavelength_nm,
                    "stage": int(diagnostic.stage),
                    "code": int(diagnostic.code),
                    "message": diagnostic.message,
                    "cause_summary": diagnostic.cause_summary,
                }
                for diagnostic in self.failure_diagnostics
            ],
        }

    def execution_metadata(self) -> dict[str, str]:
        """Return scalar-only metadata accepted by the generic operation contract."""
        join = lambda values: ",".join(str(value) for value in values)
        return {
            "product_completeness": self.completeness.value,
            "product_status": self.product_status.value,
            "requested_wavelengths": join(self.requested_wavelengths),
            "processed_wavelengths": join(self.processed_wavelengths),
            "failed_wavelengths": join(self.failed_wavelengths),
        }


def enum_flag_metadata(enum_type: type[IntEnum]) -> tuple[str, str]:
    """Return NetCDF flag values/meanings in stable numeric order."""
    members = tuple(sorted(enum_type, key=int))
    return ", ".join(str(int(member)) for member in members), " ".join(
        member.name.lower() for member in members
    )


def dataset_product_summary(dataset: Any) -> dict[str, object]:
    """Read compact completeness information without selecting absent wavelengths."""
    def integer_list(variable_name: str, fallback: tuple[int, ...] = ()) -> list[int]:
        if variable_name not in dataset:
            return list(fallback)
        return [int(value) for value in dataset[variable_name].values.tolist()]

    scientific = tuple(
        int(value) for value in dataset["wavelength"].values.tolist()
    ) if "wavelength" in dataset.coords else ()
    requested = integer_list("requested_wavelengths", scientific)
    processed = integer_list("processed_wavelengths", scientific)
    failed = integer_list("failed_wavelengths")
    stages = integer_list("failed_wavelength_stage")
    codes = integer_list("failed_wavelength_code")
    messages = (
        [str(value) for value in dataset["failed_wavelength_message"].values.tolist()]
        if "failed_wavelength_message" in dataset
        else []
    )
    diagnostics = []
    for wavelength, stage, code, message in zip(failed, stages, codes, messages):
        try:
            stage_name = WavelengthFailureStage(stage).name.lower()
        except ValueError:
            stage_name = "unknown"
        try:
            code_name = WavelengthFailureCode(code).name.lower()
        except ValueError:
            code_name = "internal_error"
        diagnostics.append(
            {
                "wavelength_nm": wavelength,
                "stage": stage_name,
                "code": code_name,
                "message": message,
            }
        )
    return {
        "product_completeness": str(dataset.attrs.get("product_completeness", "unknown")),
        "product_status": str(dataset.attrs.get("product_status", "unknown")),
        "requested_wavelengths": requested,
        "processed_wavelengths": processed,
        "failed_wavelengths": failed,
        "failures": diagnostics,
    }


def format_dataset_product_summary(dataset: Any) -> tuple[str, ...]:
    """Return short lines suitable for QA artifacts and the Explorer."""
    summary = dataset_product_summary(dataset)
    requested = ", ".join(str(value) for value in summary["requested_wavelengths"]) or "none"
    processed = ", ".join(str(value) for value in summary["processed_wavelengths"]) or "none"
    failed = ", ".join(str(value) for value in summary["failed_wavelengths"]) or "none"
    lines = [
        f"product_completeness: {summary['product_completeness']}",
        f"product_status: {summary['product_status']}",
        f"requested_wavelengths_nm: {requested}",
        f"processed_wavelengths_nm: {processed}",
        f"failed_wavelengths_nm: {failed}",
    ]
    lines.extend(
        f"failure_{item['wavelength_nm']}_nm: {item['stage']} / {item['code']} / {item['message']}"
        for item in summary["failures"]
    )
    return tuple(lines)
