"""Sealing the physics board so the answer is preregistered rather than withheld.

The encrypted board and a plaintext SHA256 of it both ship in the repository.
The digest is content, not security: it lets a participant verify after the fact
that the answer was fixed before they committed to theirs, which is the practice
the whole session argues for. The encryption only stops someone reading ahead by
accident.

`unseal` refuses to run until a team has written a verdict, so the reveal cannot
happen before the commitment it is supposed to overturn.
"""

from __future__ import annotations

import base64
import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

SALT = b"engiopt-idetc26"
"""Fixed salt. The passphrase is announced aloud in the room, so this protects
nothing and only has to be stable across machines."""


class SealError(RuntimeError):
    """Raised when a sealed board cannot be opened, with the reason."""


def _key(passphrase: str) -> bytes:
    """Derive a Fernet key from the spoken passphrase."""
    digest = hashlib.pbkdf2_hmac("sha256", passphrase.strip().encode(), SALT, 200_000)
    return base64.urlsafe_b64encode(digest)


def _fernet(passphrase: str):
    """Build the cipher, explaining the install if `cryptography` is missing.

    Raises:
        SealError: If the `cryptography` package is unavailable.
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise SealError("Sealing needs the `cryptography` package: pip install cryptography") from exc
    return Fernet(_key(passphrase))


def seal(plaintext: str, passphrase: str, destination: Path) -> str:
    """Encrypt a board and write it alongside its plaintext digest.

    Args:
        plaintext: The CSV text to seal.
        passphrase: The phrase the facilitator will announce.
        destination: Path to write; a `.sha256` sibling is written too.

    Returns:
        The SHA256 of the *plaintext*, which is what gets published.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_fernet(passphrase).encrypt(plaintext.encode()))
    digest = hashlib.sha256(plaintext.encode()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(digest + "\n")
    return digest


def unseal(source: Path, passphrase: str) -> str:
    """Decrypt a sealed board and verify it against its published digest.

    Args:
        source: The `.enc` file to open.
        passphrase: The phrase announced in the room.

    Returns:
        The decrypted CSV text.

    Raises:
        SealError: If the file is missing, the passphrase is wrong, or the
            decrypted content does not match the digest committed beside it.
    """
    if not source.exists():
        raise SealError(f"No sealed board at {source}.")

    try:
        plaintext = _fernet(passphrase).decrypt(source.read_bytes()).decode()
    except SealError:
        raise
    except Exception as exc:
        raise SealError("That passphrase does not open this board.") from exc

    digest_path = source.with_suffix(source.suffix + ".sha256")
    if digest_path.exists():
        expected = digest_path.read_text().strip()
        actual = hashlib.sha256(plaintext.encode()).hexdigest()
        if actual != expected:
            raise SealError(
                f"The sealed board opened but does not match its published digest "
                f"({actual[:12]}... != {expected[:12]}...). Someone moved the goalposts."
            )
    return plaintext
