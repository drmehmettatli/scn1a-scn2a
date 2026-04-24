"""
Stage 1 - Step 2: Run AlphaFold3
=================================
Prepare AlphaFold3 JSON input files for SCN1A and SCN2A with sodium ion
ligands, then submit to AlphaFold3 server or local installation.

Usage:
    python 02_run_alphafold3.py \
        --sequences_dir ../data/raw/sequences \
        --output_dir ../data/processed/alphafold
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

GENES = ["SCN1A", "SCN2A"]
UNIPROT_IDS = {"SCN1A": "P35498", "SCN2A": "Q99250"}

# AlphaFold3 local binary path (adjust for your installation)
AF3_BINARY = os.environ.get("AF3_BINARY", "run_alphafold.sh")
AF3_MODEL_DIR = os.environ.get("AF3_MODEL_DIR", "/opt/alphafold3/models")
AF3_DB_DIR = os.environ.get("AF3_DB_DIR", "/opt/alphafold3/databases")


def read_fasta(fasta_path: str) -> tuple[str, str]:
    """Read a FASTA file and return (header, sequence).

    Args:
        fasta_path: Path to FASTA file.

    Returns:
        Tuple of (header_line_without_gt, sequence).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid FASTA.
    """
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    with open(fasta_path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    header_lines = [l for l in lines if l.startswith(">")]
    if not header_lines:
        raise ValueError(f"No FASTA header found in {fasta_path}")
    header = header_lines[0].lstrip(">").strip()
    sequence = "".join(l for l in lines if not l.startswith(">") and l.strip())
    if not sequence:
        raise ValueError(f"Empty sequence in {fasta_path}")
    return header, sequence


def build_af3_input(gene_name: str, sequence: str, n_sodium_ions: int = 4) -> dict:
    """Build an AlphaFold3 JSON input dictionary for a sodium channel protein.

    Includes the protein chain and sodium ion ligands.

    Args:
        gene_name: Gene name for the job identifier.
        sequence: Protein amino acid sequence.
        n_sodium_ions: Number of sodium ions to include (default: 4, one per domain).

    Returns:
        AlphaFold3 input dictionary ready for JSON serialization.
    """
    sequences = [
        {
            "protein": {
                "id": "A",
                "sequence": sequence,
            }
        }
    ]
    # Add sodium ions as individual ligand entries
    for i in range(n_sodium_ions):
        ion_id = chr(ord("B") + i)  # B, C, D, E
        sequences.append(
            {
                "ligand": {
                    "id": ion_id,
                    "ccdCodes": ["NA"],
                }
            }
        )

    af3_input = {
        "name": f"{gene_name.lower()}_with_sodium",
        "modelSeeds": [42, 1337],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 2,
    }
    return af3_input


def save_af3_input(af3_input: dict, output_dir: str, gene_name: str) -> str:
    """Save AlphaFold3 input JSON to disk.

    Args:
        af3_input: Dictionary produced by build_af3_input().
        output_dir: Base output directory.
        gene_name: Gene name used for subdirectory and filename.

    Returns:
        Path to the saved JSON file.
    """
    gene_dir = os.path.join(output_dir, gene_name.lower())
    os.makedirs(gene_dir, exist_ok=True)
    json_path = os.path.join(gene_dir, f"{gene_name.lower()}_af3_input.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(af3_input, fh, indent=2)
    logger.info("Saved AlphaFold3 input JSON: %s", json_path)
    return json_path


def run_af3_local(json_path: str, output_dir: str, gene_name: str) -> bool:
    """Submit an AlphaFold3 job to a local installation via shell command.

    Requires AF3_BINARY, AF3_MODEL_DIR, and AF3_DB_DIR environment variables
    to point to a valid AlphaFold3 installation.

    Args:
        json_path: Path to the AlphaFold3 JSON input file.
        output_dir: Directory where outputs will be written.
        gene_name: Gene name for logging.

    Returns:
        True if the command succeeds, False otherwise.
    """
    import subprocess

    gene_out_dir = os.path.join(output_dir, gene_name.lower())
    cmd = [
        AF3_BINARY,
        f"--json_path={json_path}",
        f"--output_dir={gene_out_dir}",
        f"--model_dir={AF3_MODEL_DIR}",
        f"--db_dir={AF3_DB_DIR}",
    ]
    logger.info("Running AlphaFold3 locally: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("AlphaFold3 stdout:\n%s", result.stdout[-2000:])
        return True
    except FileNotFoundError:
        logger.warning(
            "AlphaFold3 binary not found at '%s'. "
            "Set AF3_BINARY environment variable to the correct path.",
            AF3_BINARY,
        )
        return False
    except subprocess.CalledProcessError as exc:
        logger.error("AlphaFold3 failed with return code %d:\n%s", exc.returncode, exc.stderr)
        return False


def submit_af3_server(af3_input: dict, gene_name: str) -> dict | None:
    """Submit an AlphaFold3 job to the public AlphaFold3 server.

    Note: The AlphaFold3 server (https://alphafoldserver.com) requires
    authentication. This function demonstrates the API pattern; actual
    usage requires a valid session token set in AF3_SESSION_TOKEN env var.

    Args:
        af3_input: AlphaFold3 input dictionary.
        gene_name: Gene name for logging.

    Returns:
        Job response dict or None if submission fails.
    """
    session_token = os.environ.get("AF3_SESSION_TOKEN", "")
    if not session_token:
        logger.warning(
            "AF3_SESSION_TOKEN not set. Skipping server submission for %s. "
            "Set the environment variable with your AlphaFold3 server token.",
            gene_name,
        )
        return None

    headers = {
        "Authorization": f"Bearer {session_token}",
        "Content-Type": "application/json",
    }
    url = "https://alphafoldserver.com/api/predictions"
    try:
        response = requests.post(url, json=af3_input, headers=headers, timeout=60)
        response.raise_for_status()
        job_info = response.json()
        logger.info("Submitted %s to AlphaFold3 server. Job ID: %s", gene_name, job_info.get("id"))
        return job_info
    except requests.RequestException as exc:
        logger.error("Server submission failed for %s: %s", gene_name, exc)
        return None


def poll_af3_server(job_id: str, output_dir: str, gene_name: str, max_wait_s: int = 7200) -> bool:
    """Poll the AlphaFold3 server until job completion and download results.

    Args:
        job_id: Server-assigned job identifier.
        output_dir: Directory where downloaded files will be saved.
        gene_name: Gene name for path construction.
        max_wait_s: Maximum seconds to wait before timing out.

    Returns:
        True if results downloaded successfully, False otherwise.
    """
    session_token = os.environ.get("AF3_SESSION_TOKEN", "")
    headers = {"Authorization": f"Bearer {session_token}"}
    status_url = f"https://alphafoldserver.com/api/predictions/{job_id}"
    gene_dir = os.path.join(output_dir, gene_name.lower())
    os.makedirs(gene_dir, exist_ok=True)

    elapsed = 0
    poll_interval = 60
    while elapsed < max_wait_s:
        try:
            resp = requests.get(status_url, headers=headers, timeout=30)
            resp.raise_for_status()
            status_data = resp.json()
            status = status_data.get("status", "unknown")
            logger.info("[%s] Job %s status: %s (elapsed: %ds)", gene_name, job_id, status, elapsed)
            if status == "completed":
                download_url = status_data.get("downloadUrl")
                if download_url:
                    dl_resp = requests.get(download_url, headers=headers, timeout=120)
                    dl_resp.raise_for_status()
                    zip_path = os.path.join(gene_dir, f"{gene_name.lower()}_af3_results.zip")
                    with open(zip_path, "wb") as fh:
                        fh.write(dl_resp.content)
                    logger.info("Downloaded results to %s", zip_path)
                    _extract_zip(zip_path, gene_dir)
                    return True
                return True
            if status in ("failed", "error"):
                logger.error("Job %s failed: %s", job_id, status_data.get("error"))
                return False
        except requests.RequestException as exc:
            logger.warning("Polling error: %s", exc)
        time.sleep(poll_interval)
        elapsed += poll_interval

    logger.error("Timed out waiting for job %s after %ds", job_id, max_wait_s)
    return False


def _extract_zip(zip_path: str, extract_dir: str) -> None:
    """Extract a ZIP archive to a directory.

    Args:
        zip_path: Path to the ZIP file.
        extract_dir: Destination directory.
    """
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    logger.info("Extracted %s to %s", zip_path, extract_dir)


def find_fasta(sequences_dir: str, gene_name: str, uniprot_id: str) -> str | None:
    """Find the FASTA file for a gene in the sequences directory.

    Args:
        sequences_dir: Directory containing FASTA files.
        gene_name: Gene name (e.g., 'SCN1A').
        uniprot_id: UniProt ID (e.g., 'P35498').

    Returns:
        Path to the FASTA file or None if not found.
    """
    candidates = [
        os.path.join(sequences_dir, f"{gene_name}_{uniprot_id}.fasta"),
        os.path.join(sequences_dir, f"{gene_name}.fasta"),
        os.path.join(sequences_dir, f"{uniprot_id}.fasta"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    # Fallback: glob
    for fname in os.listdir(sequences_dir):
        if gene_name.lower() in fname.lower() and fname.endswith(".fasta"):
            return os.path.join(sequences_dir, fname)
    return None


def main(sequences_dir: str, output_dir: str, local: bool, server: bool) -> None:
    """Main workflow: build JSON inputs and optionally run AlphaFold3.

    Args:
        sequences_dir: Directory containing FASTA files.
        output_dir: Directory for AlphaFold3 outputs.
        local: Whether to attempt local AlphaFold3 run.
        server: Whether to attempt server submission.
    """
    os.makedirs(output_dir, exist_ok=True)

    for gene_name in GENES:
        uniprot_id = UNIPROT_IDS[gene_name]
        fasta_path = find_fasta(sequences_dir, gene_name, uniprot_id)
        if fasta_path is None:
            logger.error(
                "Could not find FASTA for %s in %s. "
                "Run 01_prepare_sequences.py first.",
                gene_name, sequences_dir,
            )
            continue

        _, sequence = read_fasta(fasta_path)
        logger.info("Building AlphaFold3 input for %s (length=%d)", gene_name, len(sequence))

        af3_input = build_af3_input(gene_name, sequence, n_sodium_ions=4)
        json_path = save_af3_input(af3_input, output_dir, gene_name)

        if local:
            success = run_af3_local(json_path, output_dir, gene_name)
            if success:
                logger.info("Local AlphaFold3 run completed for %s", gene_name)
        elif server:
            job_info = submit_af3_server(af3_input, gene_name)
            if job_info and job_info.get("id"):
                poll_af3_server(job_info["id"], output_dir, gene_name)
        else:
            logger.info(
                "Input JSON saved for %s. Use --local or --server to run AlphaFold3.",
                gene_name,
            )

    logger.info(
        "Done. AlphaFold3 input JSONs are in %s. "
        "To run locally, install AlphaFold3 and use --local flag, or use --server.",
        output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare AlphaFold3 inputs for SCN1A and SCN2A."
    )
    parser.add_argument(
        "--sequences_dir",
        default=os.path.join("..", "data", "raw", "sequences"),
        help="Directory with FASTA files",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "data", "processed", "alphafold"),
        help="Output directory for AlphaFold3 results",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run AlphaFold3 locally (requires AF3_BINARY env var)",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Submit to AlphaFold3 server (requires AF3_SESSION_TOKEN env var)",
    )
    args = parser.parse_args()
    main(args.sequences_dir, args.output_dir, args.local, args.server)
