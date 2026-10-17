"""Minimal native SCC client, using the documented browser form + read-only API.

Historical protocol reference: https://repositories.imaa.cnr.it/public/scc_access/
Unlike scc_access, TLS verification is mandatory and ambiguous POST requests
are NEVER automatically retried or assumed to have succeeded.
"""

from __future__ import annotations

import os
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urljoin, urlsplit

from milgrau.scc.packages import validate_upload_package


class SCCRemoteError(RuntimeError):
    """Remote state is unavailable, rejected or ambiguous: do not blindly retry."""


class SCCAlreadyExists(SCCRemoteError):
    """Remote measurement already exists; deletion/replacement is unsupported."""


@dataclass(frozen=True)
class SCCCredentials:
    website_username: str
    website_password: str = field(repr=False)
    basic_username: str | None = None
    basic_password: str | None = field(default=None, repr=False)

    @classmethod
    def from_environment(cls) -> "SCCCredentials":
        """No credentials are stored in YAML, manifests, logs or NetCDF."""
        username = os.environ.get("MILGRAU_SCC_USERNAME")
        password = os.environ.get("MILGRAU_SCC_PASSWORD")
        if not username or not password:
            raise SCCRemoteError(
                "Set MILGRAU_SCC_USERNAME and MILGRAU_SCC_PASSWORD in the "
                "local environment before using remote SCC operations."
            )
        basic_username = os.environ.get("MILGRAU_SCC_BASIC_USERNAME")
        basic_password = os.environ.get("MILGRAU_SCC_BASIC_PASSWORD")
        if bool(basic_username) != bool(basic_password):
            raise SCCRemoteError(
                "Set both MILGRAU_SCC_BASIC_USERNAME and MILGRAU_SCC_BASIC_PASSWORD, "
                "or neither when the server does not require HTTP Basic authentication."
            )
        return cls(username, password, basic_username, basic_password)


def validated_origin(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https" or not parsed.hostname
        or parsed.username or parsed.password or parsed.query
        or parsed.fragment or parsed.path not in {"", "/"}
    ):
        raise ValueError("SCC server must be a credential-free HTTPS origin.")
    return value.rstrip("/") + "/"


class SCCClient:
    """Synchronous authenticated SCC transport; session may be mocked in tests."""

    def __init__(
        self, base_url: str, credentials: SCCCredentials,
        *, timeout: float = 60.0, retries: int = 2, session: Any = None,
    ) -> None:
        self.base_url = validated_origin(base_url)
        if not 0 < timeout <= 600:
            raise ValueError("SCC timeout must be in (0, 600] seconds.")
        if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 5:
            raise ValueError("SCC retries must be between 0 and 5.")
        self.timeout = float(timeout)
        self.retries = retries
        self.credentials = credentials
        if session is None:
            try:
                import requests
            except ImportError as exc:
                raise SCCRemoteError(
                    "Install the optional SCC client: python -m pip install -e '.[scc]'"
                ) from exc
            session = requests.Session()
        self.session = session
        self.session.verify = True
        if credentials.basic_username and credentials.basic_password:
            self.session.auth = (credentials.basic_username, credentials.basic_password)
        self._logged_in = False

    def _url(self, relative: str) -> str:
        return urljoin(self.base_url, relative)

    def _require_origin(self, url: str) -> None:
        if urlsplit(url).netloc.lower() != urlsplit(self.base_url).netloc.lower() or (
            urlsplit(url).scheme != "https"
        ):
            raise SCCRemoteError("SCC returned a response redirected outside its HTTPS origin.")

    def _read(self, relative: str, **kwargs: Any) -> Any:
        """Bounded retries only for safe GET requests, never for multipart POST."""
        url = self._url(relative)
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(
                    url, timeout=self.timeout, allow_redirects=False, **kwargs,
                )
                self._require_origin(response.url)
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        raise SCCRemoteError("SCC GET redirect has no Location header.")
                    redirected = urljoin(url, location)
                    self._require_origin(redirected)
                    response = self.session.get(
                        redirected, timeout=self.timeout, allow_redirects=False,
                    )
                    self._require_origin(response.url)
                    if response.status_code in {301, 302, 303, 307, 308}:
                        raise SCCRemoteError("SCC issued multiple GET redirects.")
            except Exception as exc:
                if attempt == self.retries:
                    raise SCCRemoteError(f"Could not retrieve SCC {relative}: {type(exc).__name__}") from exc
                time.sleep(min(0.25 * (2**attempt), 2.0))
                continue
            if response.status_code in {502, 503, 504} and attempt < self.retries:
                time.sleep(min(0.25 * (2**attempt), 2.0))
                continue
            return response
        raise AssertionError("Unreachable retry loop")

    def _complete_post(self, response: Any, request_url: str) -> Any:
        """Follow only an allowed same-origin 302/303 redirect via a safe GET."""
        self._require_origin(response.url)
        if response.status_code in {301, 302, 303}:
            location = response.headers.get("Location")
            if not location:
                raise SCCRemoteError("SCC POST redirect has no Location.")
            destination = urljoin(request_url, location)
            self._require_origin(destination)
            return self._read(destination.removeprefix(self.base_url))
        if response.status_code in {307, 308}:
            raise SCCRemoteError("SCC requested repeating a POST; refusing unsafe redirect.")
        return response

    def _csrf(self, response: Any) -> str:
        token = response.cookies.get("csrftoken")
        if not token and hasattr(self.session, "cookies"):
            token = self.session.cookies.get("csrftoken")
        if not token:
            raise SCCRemoteError("SCC did not provide a CSRF cookie.")
        return str(token)

    def login(self) -> None:
        url = self._url("accounts/login/")
        page = self._read("accounts/login/")
        if page.status_code == 401:
            challenge = str(page.headers.get("WWW-Authenticate", "")).lower()
            if "basic" in challenge:
                raise SCCRemoteError(
                    "SCC login page returned HTTP 401 with an HTTP Basic challenge, "
                    "before website login. Configure MILGRAU_SCC_BASIC_USERNAME "
                    "and MILGRAU_SCC_BASIC_PASSWORD separately from your "
                    "MILGRAU_SCC_USERNAME and MILGRAU_SCC_PASSWORD."
                )
            raise SCCRemoteError(
                "SCC login page returned HTTP 401 before website login. "
                "The SCC server may require separate HTTP Basic credentials, "
                "or the HTTPS base URL or access permissions may be incorrect. "
                "Check scc.yaml server.base_url and the separate "
                "MILGRAU_SCC_BASIC_USERNAME/MILGRAU_SCC_BASIC_PASSWORD variables."
            )
        if page.status_code != 200:
            raise SCCRemoteError(f"SCC login page returned HTTP {page.status_code}.")
        token = self._csrf(page)
        try:
            response = self.session.post(
                url,
                data={
                    "username": self.credentials.website_username,
                    "password": self.credentials.website_password,
                },
                headers={"X-CSRFToken": token, "Referer": url},
                timeout=self.timeout,
                allow_redirects=False,
            )
        except Exception as exc:
            raise SCCRemoteError("SCC authentication failed due to a transport error.") from exc
        response = self._complete_post(response, url)
        if response.status_code != 200 or urlsplit(response.url).path.rstrip("/") == "/accounts/login":
            raise SCCRemoteError(
                "SCC login was rejected or stayed on the login page. "
                "Check website and optional HTTP Basic credentials."
            )
        self._logged_in = True

    def measurement(self, identity: str) -> Mapping[str, Any] | None:
        """Read-only API lookup; 404 means absent, every other error is fatal."""
        if not self._logged_in:
            self.login()
        from milgrau.level0.scc_staging import pending_scc_package

        # Validate IDs via the same safe path rules; no path construction escapes.
        # A synthetic config avoids any filesystem operation.
        pending_scc_package(
            {"scc": {"enabled": True, "staging_dir": ".cache/scc"}}, identity,
        )
        response = self._read(f"api/v1/measurements/{identity}/")
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise SCCRemoteError(
                f"SCC measurement query for {identity} returned HTTP {response.status_code}; "
                "remote state is unknown."
            )
        try:
            result = response.json()
        except (TypeError, ValueError) as exc:
            raise SCCRemoteError("SCC measurement API did not return valid JSON.") from exc
        if not isinstance(result, dict) or str(result.get("id", "")) != identity:
            raise SCCRemoteError(
                "SCC measurement API returned a record without the requested Measurement_ID."
            )
        return result

    def delete_measurement(
        self, identity: str, *, expected_fingerprint: str,
    ) -> None:
        """Delete exactly one SCC record via its documented Django admin form.

        Called only by the durable guarded replacement orchestration after a
        verified archive and a fresh API fingerprint match. Never auto-retry
        DELETE POST or silently delete related/calibration measurements.
        """
        record = self.measurement(identity)
        if record is None:
            raise SCCRemoteError(
                f"SCC record {identity} disappeared before deletion; state uncertain."
            )
        if record.get("is_queued") or record.get("is_being_processed"):
            raise SCCRemoteError("Cannot delete queued or running SCC measurement.")
        from milgrau.scc.backup import remote_fingerprint

        if remote_fingerprint(record) != expected_fingerprint:
            raise SCCRemoteError(
                "Remote SCC metadata changed since pre-replacement backup. "
                "Stop before deletion and reconcile."
            )
        relative = f"admin/database/measurements/{identity}/delete/"
        url = self._url(relative)
        form = self._read(relative)
        if form.status_code != 200:
            raise SCCRemoteError(
                f"SCC deletion confirmation page returned HTTP {form.status_code}."
            )
        token = self._csrf(form)
        try:
            response = self.session.post(
                url,
                data={
                    "post": "yes",
                    "select_delete_related_measurements": "not_delete_related",
                },
                headers={"X-CSRFToken": token, "Referer": url},
                timeout=self.timeout,
                allow_redirects=False,
            )
        except Exception as exc:
            raise SCCRemoteError(
                f"Deletion POST for {identity} has an UNKNOWN outcome "
                f"({type(exc).__name__}); check remote state and journal, "
                "do not automatically retry."
            ) from exc
        response = self._complete_post(response, url)
        if response.status_code != 200:
            raise SCCRemoteError(
                f"SCC delete confirmation returned HTTP {response.status_code}; "
                "the result is ambiguous and must be reconciled."
            )
        # Deletion is successful only if the authoritative API now returns 404.
        # Any error/unchanged record must stop the replacement.
        if self.measurement(identity) is not None:
            raise SCCRemoteError(
                f"SCC measurement {identity} still exists after delete POST; "
                "replacement aborted."
            )

    def upload_package(self, path: Any) -> dict[str, Any]:
        """Upload one validated package, never overwrite or blindly retry a POST.

        An HTTP/network interruption after POST is an ambiguous state: the
        caller must query the remote Measurement_ID before any future upload.
        """
        manifest = validate_upload_package(path)
        identity = manifest["measurement_id"]
        if self.measurement(identity) is not None:
            raise SCCAlreadyExists(
                f"SCC already contains measurement {identity}; remote replacement is disabled."
            )
        url = self._url("data_processing/measurements/quick/")
        page = self._read("data_processing/measurements/quick/")
        if page.status_code != 200 or urlsplit(page.url).path.rstrip("/") == "/accounts/login":
            raise SCCRemoteError("SCC Quick upload page is unavailable or login expired.")
        token = self._csrf(page)
        with ExitStack() as stack:
            raw = stack.enter_context(open(path / "raw.nc", "rb"))
            uploads = {"data": ("raw.nc", raw, "application/x-netcdf")}
            lr_name = f"lr_{identity}.nc"
            if lr_name in manifest["files"]:
                lr_handle = stack.enter_context(open(path / lr_name, "rb"))
                uploads["lidar_ratio_file"] = (lr_name, lr_handle, "application/x-netcdf")
            try:
                response = self.session.post(
                    url,
                    data={
                        "system": str(manifest["scc_configuration_id"]),
                        "delay": "0",
                        "manually_cloud_free": "False",
                    },
                    files=uploads,
                    headers={"X-CSRFToken": token, "Referer": url},
                    timeout=self.timeout,
                    allow_redirects=False,
                )
            except Exception as exc:
                raise SCCRemoteError(
                    f"SCC POST for {identity} has an UNKNOWN remote outcome. "
                    "Never resend blindly; inspect status first."
                ) from exc
        response = self._complete_post(response, url)
        if response.status_code != 200:
            raise SCCRemoteError(
                f"SCC returned HTTP {response.status_code} after uploading {identity}; "
                "check remote status before retrying."
            )
        # The official scc_access client documents SCC API indexing delays
        # after a successful Quick upload. Poll using safe GET ONLY: never
        # repeat the multipart POST. HTTP 200 on the Quick upload form itself
        # can also mean server-side form validation rejected the input.
        form_stayed_open = (
            urlsplit(response.url).path.rstrip("/")
            == urlsplit(url).path.rstrip("/")
        )
        confirmed = None
        for attempt in range(3):
            if attempt:
                time.sleep(2)
            confirmed = self.measurement(identity)
            if confirmed is not None:
                break
        if confirmed is None:
            if form_stayed_open:
                raise SCCRemoteError(
                    f"SCC Quick upload for {identity} returned to its original "
                    "form rather than the measurement detail page. This commonly "
                    "means validation rejected raw.nc or LR, but the remote "
                    "outcome is still unconfirmed. Inspect SCC portal/form "
                    "messages; do not retry the POST."
                )
            raise SCCRemoteError(
                f"Upload response for {identity} returned HTTP 200 but was "
                "not confirmed by the SCC API after 3 read-only checks. "
                "Remote state is uncertain; do not repeat the POST."
            )
        return {
            "measurement_id": identity,
            "remote_page": self._url(f"data_processing/measurements/{identity}/"),
            "status": "accepted_by_remote_api",
        }
