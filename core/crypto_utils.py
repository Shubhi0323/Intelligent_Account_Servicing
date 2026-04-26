import os
import logging
from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

from core.config import ENCRYPTION_KEY

# Initialize Fernet suite. If no key is provided, we won't crash immediately but encryption will fail.
_fernet = None
if ENCRYPTION_KEY:
    try:
        _fernet = Fernet(ENCRYPTION_KEY.encode('utf-8'))
    except Exception as e:
        logger.error(f"Failed to initialize encryption key: {e}")

def encrypt_data(text: str) -> str:
    """Encrypt a plaintext string using Fernet symmetric encryption."""
    if not text:
        return text
    if not _fernet:
        logger.warning("Encryption key not set. Saving data as plaintext!")
        return text
    
    # Check if already encrypted (starts with 'gAAAAA')
    if text.startswith('gAAAAA'):
        return text
        
    try:
        encrypted_bytes = _fernet.encrypt(text.encode('utf-8'))
        return encrypted_bytes.decode('utf-8')
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        return text

def decrypt_data(ciphertext: str) -> str:
    """Decrypt a Fernet token back to plaintext string."""
    if not ciphertext:
        return ciphertext
    if not _fernet:
        return ciphertext

    # If it doesn't look like a fernet token, return as is (legacy support)
    if not ciphertext.startswith('gAAAAA'):
        return ciphertext

    try:
        decrypted_bytes = _fernet.decrypt(ciphertext.encode('utf-8'))
        return decrypted_bytes.decode('utf-8')
    except InvalidToken:
        logger.error("Decryption failed: Invalid token. Returning raw string.")
        return ciphertext
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        return ciphertext

def mask_email(email: str) -> str:
    """Mask email: john.doe@example.com -> j***@example.com"""
    if not email or "@" not in email:
        return email
    local, domain = email.split("@", 1)
    if len(local) > 1:
        masked_local = f"{local[0]}***"
    else:
        masked_local = "***"
    return f"{masked_local}@{domain}"

def mask_phone(phone: str) -> str:
    """Mask phone: 9876543210 -> ******3210"""
    if not phone or len(phone) < 4:
        return phone
    return f"{'*' * (len(phone) - 4)}{phone[-4:]}"

def mask_address(address: str) -> str:
    """Mask address: 42, Sector 18, Noida -> 42, ***, Noida"""
    if not address:
        return address
    parts = [p.strip() for p in address.split(",")]
    if len(parts) > 2:
        parts[1] = "***"
    elif len(parts) == 2:
        parts[0] = "***"
    else:
        return "***"
    return ", ".join(parts)

def mask_dob(dob: str) -> str:
    """Mask DOB: 05-07-1990 -> **-**-1990"""
    if not dob:
        return dob
    parts = dob.replace('/', '-').split('-')
    if len(parts) == 3:
        return f"**-**-{parts[-1]}"
    return "***"
