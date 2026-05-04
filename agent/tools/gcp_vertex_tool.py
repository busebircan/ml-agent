"""GCP Vertex AI training job tool.

Submits custom training jobs to Google Cloud Vertex AI.
Uses the google-cloud-aiplatform SDK when credentials are available;
falls back to generating the equivalent gcloud CLI command when not.

Operations:
  run     — submit a new custom training job
  status  — check job state
  logs    — tail job logs via gcloud
  cancel  — cancel a running job
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware sizing table (mirrors system prompt hardware guidance)
# ---------------------------------------------------------------------------

MACHINE_SPECS: dict[str, dict[str, Any]] = {
    # name → {machine_type, accelerator_type, accelerator_count, description}
    "t4": {
        "machine_type": "n1-standard-4",
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 1,
        "description": "T4 16GB — 1-3B params",
    },
    "t4x2": {
        "machine_type": "n1-standard-8",
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 2,
        "description": "T4 x2 32GB — 3-7B params",
    },
    "v100": {
        "machine_type": "n1-standard-8",
        "accelerator_type": "NVIDIA_TESLA_V100",
        "accelerator_count": 1,
        "description": "V100 16GB — 3-7B params",
    },
    "a100": {
        "machine_type": "a2-highgpu-1g",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 1,
        "description": "A100 80GB — 7-30B params",
    },
    "a100x4": {
        "machine_type": "a2-highgpu-4g",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 4,
        "description": "A100 x4 320GB — 30-70B params",
    },
    "a100x8": {
        "machine_type": "a2-highgpu-8g",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 8,
        "description": "A100 x8 640GB — 70B+ params",
    },
}

# Pre-built Vertex AI containers (avoids building custom Docker images)
CONTAINERS: dict[str, str] = {
    "pytorch-gpu-2-0":   "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0:latest",
    "pytorch-gpu-2-1":   "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest",
    "pytorch-cpu":       "us-docker.pkg.dev/vertex-ai/training/pytorch-xla-cpu.2-0:latest",
    "tf-gpu-2-13":       "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-13:latest",
    "sklearn-1-0":       "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
}
DEFAULT_CONTAINER = CONTAINERS["pytorch-gpu-2-1"]

HARDWARE_HELP = "\n".join(
    f"  {k:10} — {v['description']}" for k, v in MACHINE_SPECS.items()
)

CONTAINER_HELP = "\n".join(f"  {k}" for k in CONTAINERS)

# ---------------------------------------------------------------------------
# Tool spec
# ---------------------------------------------------------------------------

GCP_VERTEX_TOOL_SPEC: dict[str, Any] = {
    "name": "gcp_vertex",
    "description": (
        "Submit and manage training jobs on Google Cloud Vertex AI.\n"
        "\n"
        "Operations:\n"
        "  run    — submit a custom training job\n"
        "  status — check job state (PENDING/RUNNING/SUCCEEDED/FAILED)\n"
        "  logs   — tail job logs\n"
        "  cancel — cancel a running job\n"
        "\n"
        "Hardware options (hardware arg):\n"
        f"{HARDWARE_HELP}\n"
        "\n"
        "Container options (container arg):\n"
        f"{CONTAINER_HELP}\n"
        "\n"
        "Required GCP setup:\n"
        "  - Set GCP_PROJECT, GCP_REGION in .env (default region: europe-west4)\n"
        "  - Authenticate: gcloud auth application-default login\n"
        "  - Enable APIs: Vertex AI, Cloud Storage\n"
        "\n"
        "If credentials are not configured, the tool returns the equivalent\n"
        "gcloud command you can run manually.\n"
        "\n"
        "Pre-flight before calling run:\n"
        "  - Training script must pass lint_python\n"
        "  - Set timeout_hours based on model size (minimum 2h for any training)\n"
        "  - Always set a GCS output_dir to persist the trained model\n"
    ),
    "parameters": {
        "type": "object",
        "required": ["operation"],
        "additionalProperties": False,
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["run", "status", "logs", "cancel"],
                "description": "Operation to perform.",
            },
            # run params
            "job_name": {
                "type": "string",
                "description": "Unique job name (run). Auto-generated if omitted.",
            },
            "script_path": {
                "type": "string",
                "description": "Local path to the Python training script (run).",
            },
            "script_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "CLI args passed to the script, e.g. ['--epochs', '10'] (run).",
            },
            "hardware": {
                "type": "string",
                "description": f"Hardware preset. One of: {', '.join(MACHINE_SPECS)}. Default: t4.",
            },
            "container": {
                "type": "string",
                "description": f"Pre-built container key. One of: {', '.join(CONTAINERS)}. Default: pytorch-gpu-2-1.",
            },
            "output_dir": {
                "type": "string",
                "description": "GCS path for model output, e.g. gs://my-bucket/runs/exp1 (run). Required.",
            },
            "requirements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra pip packages to install at job start, e.g. ['lightgbm', 'timm'] (run).",
            },
            "env_vars": {
                "type": "object",
                "description": "Extra environment variables to inject into the job (run).",
            },
            "timeout_hours": {
                "type": "number",
                "description": "Job timeout in hours. Default 4. Never use less than 2 for real training.",
            },
            # status/logs/cancel params
            "job_id": {
                "type": "string",
                "description": "Vertex AI job resource name or numeric ID (status/logs/cancel).",
            },
            "log_lines": {
                "type": "integer",
                "description": "Number of log lines to return (logs). Default 100.",
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_project() -> str | None:
    return os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")


def _get_region() -> str:
    return os.environ.get("GCP_REGION", "europe-west4")


def _gcloud(args: list[str]) -> tuple[str, int]:
    result = subprocess.run(
        ["gcloud"] + args,
        capture_output=True, text=True,
    )
    return (result.stdout + result.stderr).strip(), result.returncode


def _sdk_available() -> bool:
    try:
        import google.cloud.aiplatform  # noqa: F401
        return True
    except ImportError:
        return False


def _credentials_available() -> bool:
    """Check if ADC (application default credentials) are configured."""
    _, code = _gcloud(["auth", "application-default", "print-access-token"])
    return code == 0


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def _run_job(args: dict[str, Any]) -> str:
    import uuid
    import datetime

    project = _get_project()
    region = _get_region()
    job_name = args.get("job_name") or f"ml-agent-{datetime.date.today()}-{uuid.uuid4().hex[:6]}"
    script_path = args.get("script_path", "train.py")
    script_args = args.get("script_args") or []
    hardware_key = args.get("hardware", "t4")
    container_key = args.get("container", "pytorch-gpu-2-1")
    output_dir = args.get("output_dir", "")
    requirements = args.get("requirements") or []
    env_vars: dict[str, str] = args.get("env_vars") or {}
    timeout_hours = float(args.get("timeout_hours") or 4)

    if hardware_key not in MACHINE_SPECS:
        return (
            f"Unknown hardware '{hardware_key}'. "
            f"Valid options: {', '.join(MACHINE_SPECS)}\n{HARDWARE_HELP}"
        )

    hw = MACHINE_SPECS[hardware_key]
    container_uri = CONTAINERS.get(container_key, container_key)  # allow raw URI too

    if not output_dir:
        return (
            "output_dir is required (GCS path, e.g. gs://my-bucket/runs/exp1).\n"
            "The trained model must be saved there — job storage is ephemeral."
        )

    if not project:
        return (
            "GCP_PROJECT not set. Add it to your .env file:\n"
            "  GCP_PROJECT=your-project-id\n"
            "Then re-run the job."
        )

    # Build worker pool spec
    worker_spec: dict[str, Any] = {
        "machine_spec": {
            "machine_type": hw["machine_type"],
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": container_uri,
            "command": ["bash", "-c"],
            "args": [_build_entrypoint(script_path, script_args, requirements, output_dir)],
            "env": [{"name": k, "value": v} for k, v in env_vars.items()]
                   + [{"name": "AIP_MODEL_DIR", "value": output_dir}],
        },
    }
    if hw.get("accelerator_type"):
        worker_spec["machine_spec"]["accelerator_type"] = hw["accelerator_type"]
        worker_spec["machine_spec"]["accelerator_count"] = hw["accelerator_count"]

    job_spec = {
        "displayName": job_name,
        "jobSpec": {
            "workerPoolSpecs": [worker_spec],
            "baseOutputDirectory": {"outputUriPrefix": output_dir},
        },
    }

    timeout_seconds = int(timeout_hours * 3600)

    # Try SDK submission first
    if _sdk_available() and _credentials_available():
        try:
            return _submit_via_sdk(job_name, job_spec, project, region, timeout_seconds)
        except Exception as exc:
            logger.warning("SDK submission failed, falling back to gcloud: %s", exc)

    # Fall back to gcloud command
    return _generate_gcloud_command(job_name, job_spec, project, region, timeout_seconds, hw, container_uri)


def _build_entrypoint(
    script_path: str,
    script_args: list[str],
    requirements: list[str],
    output_dir: str,
) -> str:
    """Build the bash entrypoint string for the container."""
    lines = []
    if requirements:
        pkgs = " ".join(requirements)
        lines.append(f"pip install -q {pkgs}")
    lines.append(f"python {script_path} {' '.join(script_args)}")
    return " && ".join(lines) if lines else f"python {script_path}"


def _submit_via_sdk(
    job_name: str,
    job_spec: dict[str, Any],
    project: str,
    region: str,
    timeout_seconds: int,
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=job_spec["jobSpec"]["workerPoolSpecs"],
        base_output_dir=job_spec["jobSpec"]["baseOutputDirectory"]["outputUriPrefix"],
    )
    job.submit(timeout=timeout_seconds)
    resource_name = job.resource_name
    console_url = (
        f"https://console.cloud.google.com/vertex-ai/training/custom-jobs"
        f"?project={project}"
    )
    return (
        f"✅ Job submitted via SDK\n"
        f"  Job name:     {job_name}\n"
        f"  Resource:     {resource_name}\n"
        f"  Region:       {region}\n"
        f"  Console:      {console_url}\n\n"
        f"Check status: gcp_vertex({{\"operation\": \"status\", \"job_id\": \"{resource_name}\"}})\n"
        f"View logs:    gcp_vertex({{\"operation\": \"logs\",   \"job_id\": \"{resource_name}\"}})"
    )


def _generate_gcloud_command(
    job_name: str,
    job_spec: dict[str, Any],
    project: str,
    region: str,
    timeout_seconds: int,
    hw: dict[str, Any],
    container_uri: str,
) -> str:
    """Return the gcloud CLI command when SDK/credentials are unavailable."""
    spec_json = json.dumps(job_spec, indent=2)
    return (
        f"⚠️  GCP credentials not configured — returning gcloud command.\n\n"
        f"Run this to submit the job:\n\n"
        f"gcloud ai custom-jobs create \\\n"
        f"  --project={project} \\\n"
        f"  --region={region} \\\n"
        f"  --display-name={job_name} \\\n"
        f"  --config=- <<'EOF'\n{spec_json}\nEOF\n\n"
        f"Then authenticate with:\n"
        f"  gcloud auth application-default login\n\n"
        f"Hardware: {hw['description']}\n"
        f"Container: {container_uri}"
    )


def _status(args: dict[str, Any]) -> str:
    job_id = args.get("job_id", "")
    if not job_id:
        return "job_id is required for status."
    project = _get_project()
    region = _get_region()
    out, code = _gcloud([
        "ai", "custom-jobs", "describe", job_id,
        f"--project={project}", f"--region={region}",
        "--format=json",
    ])
    if code != 0:
        return f"gcloud error:\n{out}"
    try:
        data = json.loads(out)
        state = data.get("state", "UNKNOWN")
        name = data.get("displayName", job_id)
        create_time = data.get("createTime", "")
        end_time = data.get("endTime", "")
        error = data.get("error", {})
        lines = [
            f"Job:    {name}",
            f"State:  {state}",
            f"Start:  {create_time}",
        ]
        if end_time:
            lines.append(f"End:    {end_time}")
        if error:
            lines.append(f"Error:  {error.get('message', error)}")
        return "\n".join(lines)
    except json.JSONDecodeError:
        return out


def _logs(args: dict[str, Any]) -> str:
    job_id = args.get("job_id", "")
    if not job_id:
        return "job_id is required for logs."
    project = _get_project()
    region = _get_region()
    n = args.get("log_lines", 100)
    # Use gcloud logging to fetch logs
    filter_str = f'resource.type="aiplatform.googleapis.com/CustomJob" resource.labels.job_id="{job_id}"'
    out, code = _gcloud([
        "logging", "read", filter_str,
        f"--project={project}",
        f"--limit={n}",
        "--format=value(textPayload)",
        "--order=asc",
    ])
    if code != 0:
        return f"gcloud error:\n{out}"
    return out or "(no logs yet — job may still be initialising)"


def _cancel(args: dict[str, Any]) -> str:
    job_id = args.get("job_id", "")
    if not job_id:
        return "job_id is required for cancel."
    project = _get_project()
    region = _get_region()
    out, code = _gcloud([
        "ai", "custom-jobs", "cancel", job_id,
        f"--project={project}", f"--region={region}",
    ])
    if code != 0:
        return f"gcloud error:\n{out}"
    return f"Job {job_id} cancelled."


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

async def gcp_vertex_handler(args: dict[str, Any], **_kw) -> tuple[str, bool]:
    operation = args.get("operation", "")
    try:
        if operation == "run":
            result = _run_job(args)
        elif operation == "status":
            result = _status(args)
        elif operation == "logs":
            result = _logs(args)
        elif operation == "cancel":
            result = _cancel(args)
        else:
            result = f"Unknown operation '{operation}'. Valid: run, status, logs, cancel."

        success = not result.startswith("⚠️") and "error" not in result.lower()[:50]
        return result, success
    except Exception as exc:
        logger.exception("gcp_vertex tool error")
        return f"gcp_vertex error: {exc}", False
