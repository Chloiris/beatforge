from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .base import (
    AdapterDiagnostics,
    AlignmentAdapter,
    AlignmentAdapterError,
    AlignmentContext,
)
from .evaluator import AlignmentEvaluator
from .hubert_engine import (
    HubertAlignmentReport,
    HubertCandidateBundle,
    build_hubert_artifacts,
    load_hubert_candidates,
    load_hubert_report,
    publish_hubert_artifacts,
)
from .hybrid import HybridAlignmentAdapter
from .schema import (
    AlignmentErrorInfo,
    AlignmentMethod,
    AlignmentMethodId,
    AlignmentReport,
    AlignmentResult,
)

_METHOD_IDS: tuple[AlignmentMethodId, ...] = ("qwen", "mfa", "ctc", "singing", "hybrid")
_COMPONENT_IDS: tuple[AlignmentMethodId, ...] = ("qwen", "mfa", "ctc", "singing")


def _default_adapters() -> dict[AlignmentMethodId, AlignmentAdapter]:
    # Heavy optional dependencies remain behind subprocess adapters and are never
    # imported into the API process at module import time.
    from .hubert_ctc import HubertCTCAlignmentAdapter
    from .mfa_adapter import MFAAlignmentAdapter
    from .qwen_adapter import QwenAlignmentAdapter
    from .singing_adapter import SingingAlignmentAdapter

    adapters: list[AlignmentAdapter] = [
        QwenAlignmentAdapter(),
        MFAAlignmentAdapter(),
        HubertCTCAlignmentAdapter(),
        SingingAlignmentAdapter(),
        HybridAlignmentAdapter(),
    ]
    return {adapter.method: adapter for adapter in adapters}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


class AlignmentRunner:
    """Single-worker orchestration for memory-heavy local alignment experiments."""

    def __init__(
        self,
        storage_dir: Path,
        project_root: Path,
        *,
        adapters: dict[AlignmentMethodId, AlignmentAdapter] | None = None,
        evaluator: AlignmentEvaluator | None = None,
    ) -> None:
        self.storage_dir = storage_dir.resolve()
        self.project_root = project_root.resolve()
        self.root = self.storage_dir / "alignment"
        self.root.mkdir(parents=True, exist_ok=True)
        self.adapters = adapters or _default_adapters()
        self.evaluator = evaluator or AlignmentEvaluator()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alignment-lab")
        self._futures: dict[tuple[str, AlignmentMethodId], Future[None]] = {}
        self._lock = threading.RLock()
        self.recover_interrupted()

    @staticmethod
    def input_fingerprint(context: AlignmentContext) -> str:
        stat = context.vocals_path.stat()
        digest = hashlib.sha256()
        digest.update(context.track_id.encode())
        digest.update(context.lyrics.encode("utf-8"))
        digest.update(context.lyrics_format.encode())
        digest.update(str(context.sample_rate).encode())
        digest.update(str(context.sample_count).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        return digest.hexdigest()

    def _method_dir(self, track_id: str, method: AlignmentMethodId) -> Path:
        return self.root / track_id / method

    def _result_path(self, track_id: str, method: AlignmentMethodId, run_id: str) -> Path:
        return self._method_dir(track_id, method) / f"{run_id}.result.json"

    def _report_path(self, track_id: str, method: AlignmentMethodId, run_id: str) -> Path:
        return self._method_dir(track_id, method) / f"{run_id}.report.json"

    def _latest_path(self, track_id: str, method: AlignmentMethodId) -> Path:
        return self._method_dir(track_id, method) / "latest.json"

    def _write_result(self, result: AlignmentResult, *, publish_latest: bool = True) -> None:
        _atomic_json(
            self._result_path(result.track_id, result.method, result.run_id),
            result.model_dump(mode="json", by_alias=False),
        )
        if publish_latest:
            _atomic_json(
                self._latest_path(result.track_id, result.method),
                {"runId": result.run_id},
            )

    def _write_report(self, report: AlignmentReport) -> None:
        _atomic_json(
            self._report_path(report.track_id, report.method, report.run_id),
            report.model_dump(mode="json", by_alias=False),
        )

    def _latest_run_id(self, track_id: str, method: AlignmentMethodId) -> str | None:
        path = self._latest_path(track_id, method)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        run_id = payload.get("runId")
        return str(run_id) if run_id else None

    def get_result(self, track_id: str, method: AlignmentMethodId) -> AlignmentResult | None:
        run_id = self._latest_run_id(track_id, method)
        if not run_id:
            return None
        path = self._result_path(track_id, method, run_id)
        try:
            return AlignmentResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def get_report(self, track_id: str, method: AlignmentMethodId) -> AlignmentReport | None:
        run_id = self._latest_run_id(track_id, method)
        if not run_id:
            return None
        path = self._report_path(track_id, method, run_id)
        try:
            return AlignmentReport.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def get_hubert_candidates(self, track_id: str) -> HubertCandidateBundle | None:
        result = self.get_result(track_id, "ctc")
        if result is None or result.status != "completed" or result.hierarchy is None:
            return None
        return load_hubert_candidates(self.storage_dir, track_id, result.run_id)

    def get_hubert_report(self, track_id: str) -> HubertAlignmentReport | None:
        result = self.get_result(track_id, "ctc")
        if result is None or result.status != "completed" or result.hierarchy is None:
            return None
        return load_hubert_report(self.storage_dir, track_id, result.run_id)

    def methods(self, context: AlignmentContext | None = None) -> list[AlignmentMethod]:
        descriptors: list[AlignmentMethod] = []
        available_components = 0
        component_diagnostics: dict[str, AdapterDiagnostics] = {}
        for method in _COMPONENT_IDS:
            adapter = self.adapters[method]
            diagnostics = adapter.diagnostics(context)
            component_diagnostics[method] = diagnostics
            if diagnostics.available:
                available_components += 1
            descriptors.append(
                AlignmentMethod(
                    id=method,
                    name=adapter.name,
                    available=diagnostics.available,
                    reason=diagnostics.reason,
                    model=diagnostics.model,
                    automatic_downloads_enabled=diagnostics.automatic_downloads_enabled,
                    details=diagnostics.details,
                )
            )
        hybrid = self.adapters["hybrid"]
        hybrid_available = available_components >= 2
        descriptors.append(
            AlignmentMethod(
                id="hybrid",
                name=hybrid.name,
                available=hybrid_available,
                reason=(
                    None
                    if hybrid_available
                    else "Hybrid 至少需要两个可运行的真实 alignment 方法。"
                ),
                model="observed-span consensus",
                automatic_downloads_enabled=False,
                details={
                    "availableComponents": [
                        method
                        for method, diagnostics in component_diagnostics.items()
                        if diagnostics.available
                    ],
                    "minimumSuccessfulMethods": 2,
                },
            )
        )
        return descriptors

    def submit(self, context: AlignmentContext, method: AlignmentMethodId) -> AlignmentResult:
        key = (context.track_id, method)
        with self._lock:
            future = self._futures.get(key)
            current = self.get_result(context.track_id, method)
            if future is not None and not future.done() and current is not None:
                return current
            now = datetime.now(UTC)
            queued = AlignmentResult(
                run_id=str(uuid.uuid4()),
                track_id=context.track_id,
                method=method,
                status="queued",
                sample_rate=context.sample_rate,
                sample_count=context.sample_count,
                tokens=[],
                warnings=[],
                error=None,
                metadata={"inputFingerprint": self.input_fingerprint(context)},
                created_at=now,
                updated_at=now,
            )
            self._write_result(queued)
            future = self._executor.submit(self._worker, context, queued)
            self._futures[key] = future
            future.add_done_callback(lambda _future, run_key=key: self._forget(run_key))
            return queued

    def _forget(self, key: tuple[str, AlignmentMethodId]) -> None:
        with self._lock:
            self._futures.pop(key, None)

    def _processing(self, queued: AlignmentResult) -> AlignmentResult:
        return queued.model_copy(update={"status": "processing", "updated_at": datetime.now(UTC)})

    def _terminal_error(
        self,
        current: AlignmentResult,
        error: AlignmentAdapterError,
    ) -> AlignmentResult:
        status = error.status if error.status in {"failed", "unavailable"} else "failed"
        return current.model_copy(
            update={
                "status": status,
                "tokens": [],
                "hierarchy": None,
                "error": AlignmentErrorInfo(
                    code=error.code,
                    message=error.message,
                    details=error.details,
                ),
                "updated_at": datetime.now(UTC),
            }
        )

    def _worker(self, context: AlignmentContext, queued: AlignmentResult) -> None:
        processing = self._processing(queued)
        self._write_result(processing)
        try:
            if queued.method == "hybrid":
                terminal = self._execute_hybrid(context, processing)
            else:
                terminal = self._execute_adapter(context, processing, self.adapters[queued.method])
        except AlignmentAdapterError as error:
            terminal = self._terminal_error(processing, error)
        except Exception as error:  # pragma: no cover - defensive process boundary
            terminal = self._terminal_error(
                processing,
                AlignmentAdapterError(
                    "ALIGNMENT_INTERNAL_ERROR",
                    "The local alignment runner failed unexpectedly.",
                    details={"exceptionType": type(error).__name__, "message": str(error)},
                ),
            )
        self._write_result(terminal)
        self._update_comparison_report(context)

    def _execute_adapter(
        self,
        context: AlignmentContext,
        processing: AlignmentResult,
        adapter: AlignmentAdapter,
    ) -> AlignmentResult:
        diagnostics = adapter.diagnostics(context)
        if not diagnostics.available:
            failure_code = diagnostics.details.get("failureCode")
            raise AlignmentAdapterError(
                str(failure_code or f"{processing.method.upper()}_UNAVAILABLE"),
                diagnostics.reason or f"{adapter.name} is unavailable.",
                status="unavailable",
                details=diagnostics.details,
            )
        output = adapter.run(context)
        if not output.tokens:
            raise AlignmentAdapterError(
                f"{processing.method.upper()}_EMPTY",
                f"{adapter.name} returned no real model timestamps.",
            )
        metadata = {
            **processing.metadata,
            **output.metadata,
            "adapter": adapter.__class__.__name__,
        }
        completed = processing.model_copy(
            update={
                "status": "completed",
                "tokens": list(output.tokens),
                "hierarchy": output.hierarchy,
                "warnings": list(output.warnings),
                "error": None,
                "metadata": metadata,
                "updated_at": datetime.now(UTC),
            }
        )
        try:
            report = self.evaluator.evaluate(context, completed)
            self._write_report(report)
        except Exception as error:  # alignment remains valid if proxy evaluation fails
            completed = completed.model_copy(
                update={
                    "warnings": [
                        *completed.warnings,
                        f"Proxy evaluation failed: {type(error).__name__}: {error}",
                    ],
                    "metadata": {
                        **completed.metadata,
                        "evaluationError": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    },
                    "updated_at": datetime.now(UTC),
                }
            )
        if completed.method == "ctc" and completed.hierarchy is not None:
            completed = self._publish_hubert_outputs(context, completed)
        return completed

    def _publish_hubert_outputs(
        self,
        context: AlignmentContext,
        completed: AlignmentResult,
    ) -> AlignmentResult:
        """Publish v0.6.1 products without invalidating real CTC timestamps."""

        try:
            qwen_result = self._compatible_result(context, "qwen")
            qwen_report = (
                self.get_report(context.track_id, "qwen")
                if qwen_result is not None
                else None
            )
            artifacts = build_hubert_artifacts(
                context,
                completed,
                qwen_result=qwen_result,
                qwen_report=qwen_report,
            )
            persisted_count = publish_hubert_artifacts(context, artifacts)
        except Exception as error:
            return completed.model_copy(
                update={
                    "warnings": [
                        *completed.warnings,
                        (
                            "HuBERT candidate/report publishing failed: "
                            f"{type(error).__name__}: {error}"
                        ),
                    ],
                    "metadata": {
                        **completed.metadata,
                        "hubertPostprocessError": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    },
                    "updated_at": datetime.now(UTC),
                }
            )
        return completed.model_copy(
            update={
                "metadata": {
                    **completed.metadata,
                    "hubertCandidateEventCount": len(artifacts.candidates.events),
                    "hubertPersistedCandidateCount": persisted_count,
                    "hubertReport": "reports/hubert-alignment-report.json",
                    "hubertQwenComparisonAvailable": artifacts.report.qwen_coverage is not None,
                },
                "updated_at": datetime.now(UTC),
            }
        )

    def _compatible_result(
        self,
        context: AlignmentContext,
        method: AlignmentMethodId,
    ) -> AlignmentResult | None:
        result = self.get_result(context.track_id, method)
        if (
            result is not None
            and result.status == "completed"
            and result.tokens
            and result.metadata.get("inputFingerprint") == self.input_fingerprint(context)
            and (method != "ctc" or result.hierarchy is not None)
        ):
            return result
        return None

    def _execute_component_inline(
        self,
        context: AlignmentContext,
        method: AlignmentMethodId,
    ) -> AlignmentResult:
        now = datetime.now(UTC)
        processing = AlignmentResult(
            run_id=str(uuid.uuid4()),
            track_id=context.track_id,
            method=method,
            status="processing",
            sample_rate=context.sample_rate,
            sample_count=context.sample_count,
            metadata={"inputFingerprint": self.input_fingerprint(context), "requestedBy": "hybrid"},
            created_at=now,
            updated_at=now,
        )
        self._write_result(processing)
        try:
            result = self._execute_adapter(context, processing, self.adapters[method])
        except AlignmentAdapterError as error:
            result = self._terminal_error(processing, error)
        except Exception as error:  # pragma: no cover - defensive process boundary
            result = self._terminal_error(
                processing,
                AlignmentAdapterError(
                    "ALIGNMENT_INTERNAL_ERROR",
                    "The component alignment runner failed unexpectedly.",
                    details={"exceptionType": type(error).__name__, "message": str(error)},
                ),
            )
        self._write_result(result)
        return result

    def _execute_hybrid(
        self,
        context: AlignmentContext,
        processing: AlignmentResult,
    ) -> AlignmentResult:
        results: list[AlignmentResult] = []
        failures: dict[str, dict[str, Any]] = {}
        for method in _COMPONENT_IDS:
            result = self._compatible_result(context, method)
            if result is None:
                result = self._execute_component_inline(context, method)
            if result.status == "completed" and result.tokens:
                results.append(result)
            else:
                failures[method] = {
                    "status": result.status,
                    "code": result.error.code if result.error else "UNKNOWN",
                    "message": (
                        result.error.message if result.error else "Component returned no tokens."
                    ),
                }
        adapter = self.adapters["hybrid"]
        if not isinstance(adapter, HybridAlignmentAdapter):
            raise AlignmentAdapterError(
                "HYBRID_ADAPTER_INVALID",
                "Configured hybrid adapter does not support component fusion.",
            )
        output = adapter.fuse(context, results, failures)
        completed = processing.model_copy(
            update={
                "status": "completed",
                "tokens": list(output.tokens),
                "hierarchy": None,
                "warnings": list(output.warnings),
                "error": None,
                "metadata": {
                    **processing.metadata,
                    **output.metadata,
                    "adapter": adapter.__class__.__name__,
                },
                "updated_at": datetime.now(UTC),
            }
        )
        try:
            report = self.evaluator.evaluate(context, completed)
            self._write_report(report)
        except Exception as error:  # fused timestamps remain valid if proxy evaluation fails
            completed = completed.model_copy(
                update={
                    "warnings": [
                        *completed.warnings,
                        f"Proxy evaluation failed: {type(error).__name__}: {error}",
                    ],
                    "metadata": {
                        **completed.metadata,
                        "evaluationError": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    },
                    "updated_at": datetime.now(UTC),
                }
            )
        return completed

    def _update_comparison_report(self, context: AlignmentContext) -> None:
        methods: list[dict[str, Any]] = []
        for descriptor in self.methods(context):
            result = self.get_result(context.track_id, descriptor.id)
            if result is None:
                continue
            report = self.get_report(context.track_id, descriptor.id)
            methods.append(
                {
                    "id": descriptor.id,
                    "name": descriptor.name,
                    "runId": result.run_id,
                    "status": result.status,
                    "tokenCount": len(result.tokens),
                    "hierarchyCounts": (
                        {
                            "phonemes": len(result.hierarchy.phonemes),
                            "moras": len(result.hierarchy.moras),
                            "characters": len(result.hierarchy.characters),
                        }
                        if result.hierarchy is not None
                        else None
                    ),
                    "coverage": report.coverage if report else None,
                    "acoustic": report.acoustic if report else None,
                    "rhythm": report.rhythm if report else None,
                    "stability": report.stability if report else None,
                    "score": report.score if report else None,
                    "error": result.error.model_dump(mode="json", by_alias=True)
                    if result.error
                    else None,
                }
            )
        payload = {
            "schemaVersion": "1.0",
            "song": context.song or context.track_id,
            "artist": context.artist,
            "trackId": context.track_id,
            "sampleRate": context.sample_rate,
            "sampleCount": context.sample_count,
            "generatedAt": datetime.now(UTC).isoformat(),
            "groundTruth": "proxy_only",
            "methods": methods,
        }
        _atomic_json(self.project_root / "reports" / "alignment-comparison.json", payload)

    def recover_interrupted(self) -> None:
        for latest_path in self.root.glob("*/*/latest.json"):
            try:
                method = latest_path.parent.name
                track_id = latest_path.parent.parent.name
                if method not in _METHOD_IDS:
                    continue
                result = self.get_result(track_id, method)  # type: ignore[arg-type]
                if result is None or result.status not in {"queued", "processing"}:
                    continue
                interrupted = self._terminal_error(
                    result,
                    AlignmentAdapterError(
                        "ALIGNMENT_RUN_INTERRUPTED",
                        "The previous local alignment run ended when the API stopped.",
                    ),
                )
                self._write_result(interrupted)
            except (OSError, ValueError):
                continue
