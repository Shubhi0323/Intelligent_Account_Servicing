"""
Database layer – SQLite via SQLAlchemy Core.
Tables: users, requests
"""
import uuid
import logging
from datetime import datetime

from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    String, Float, Text, DateTime, insert,
    select, update,
)
from core.config import DB_PATH
from core.crypto_utils import encrypt_data, decrypt_data

logger = logging.getLogger(__name__)

# ─── Engine & Schema ──────────────────────────────────────────────────────────
engine   = create_engine(f"sqlite:///{DB_PATH}", echo=False)
metadata = MetaData()

# ─── Users Table (Full Profile) ──────────────────────────────────────────────
users_table = Table(
    "users", metadata,
    Column("user_id",      String(36),  primary_key=True),
    Column("name",         String(128), nullable=False),
    Column("address",      Text,        nullable=True),
    Column("dob",          String(32),  nullable=True),
    Column("email",        String(128), nullable=True),
    Column("phone_number", String(32),  nullable=True),
    Column("role",         String(16),  nullable=False),       # USER or ADMIN
    Column("created_at",   DateTime,    default=datetime.utcnow),
    Column("updated_at",   DateTime,    nullable=True),
)

# ─── Requests Table ──────────────────────────────────────────────────────────
requests_table = Table(
    "requests", metadata,
    Column("request_id",       String(36),  primary_key=True),
    Column("customer_id",      String(64),  nullable=False), # actually user_id now
    Column("change_type",      String(64),  nullable=False),
    Column("old_value",        Text,        nullable=False),
    Column("new_value",        Text,        nullable=False),
    Column("extracted_value",  Text,        nullable=True),
    Column("confidence_score", Float,       nullable=True),
    Column("ai_status",        String(64),  default="AI_VERIFIED_PENDING_HUMAN"),
    Column("decision",         String(32),  default="PENDING"),
    Column("checker_remarks",  Text,        nullable=True),
    Column("ai_summary",       Text,        nullable=True),
    Column("score_breakdown",  Text,        nullable=True),
    Column("ai_explanation",   Text,        nullable=True),
    Column("created_by",       String(36),  nullable=True),
    Column("decision_by",      String(36),  nullable=True),
    Column("timestamp",        DateTime,    default=datetime.utcnow),
    Column("decision_ts",      DateTime,    nullable=True),
)

# ─── Field mapping: change_type → users column ───────────────────────────
_CHANGE_TYPE_TO_COLUMN = {
    "Legal Name Change":      "name",
    "Address Change":         "address",
    "Date of Birth Change":   "dob",
    # Contact / Email Change is handled dynamically in apply_change()
}

# ─── Init ─────────────────────────────────────────────────────────────────────

def init_db() -> None:
    metadata.create_all(engine)
    logger.info("Database initialized.")
    with engine.connect() as conn:
        if not conn.execute(select(users_table)).fetchall():
            _seed_default_users()


def _seed_default_users() -> None:
    """Insert default accounts with full profiles."""
    defaults = [
        {"user_id": "USR-001", "name": "Ravi Sharma", "role": "USER",
         "address": "12, MG Road, Mumbai - 400001", "dob": "05-07-1990",
         "email": "ravi.sharma@email.com", "phone_number": "9876543210"},
        {"user_id": "USR-002", "name": "Priya Verma", "role": "USER",
         "address": "88, Park Street, Kolkata - 700016", "dob": "12-03-1995",
         "email": "priya.verma@email.com", "phone_number": "9123456789"},
        {"user_id": "ADM-001", "name": "Amit Singh", "role": "ADMIN",
         "address": "5, Connaught Place, New Delhi - 110001", "dob": "22-11-1988",
         "email": "amit.admin@bank.com", "phone_number": "9998887776"},
        {"user_id": "ADM-002", "name": "Deepika Patel", "role": "ADMIN",
         "address": "Admin Office, Ahmedabad - 380001", "dob": "15-05-1992",
         "email": "deepika.admin@bank.com", "phone_number": "9887766554"},
    ]
    with engine.begin() as conn:
        for u in defaults:
            conn.execute(insert(users_table).values(
                user_id=u["user_id"], name=u["name"], role=u["role"],
                address=encrypt_data(u["address"]), 
                dob=encrypt_data(u["dob"]),
                email=encrypt_data(u["email"]), 
                phone_number=encrypt_data(u["phone_number"]),
                created_at=datetime.utcnow()
            ))
    logger.info("Seeded %d default users.", len(defaults))


# ─── User Profile Management ──────────────────────────────────────────────────

def create_user(name: str, role: str, address: str = "", dob: str = "",
                email: str = "", phone_number: str = "") -> str:
    """Create a new user with full profile."""
    prefix = "ADM" if role == "ADMIN" else "USR"
    user_id = f"{prefix}-{uuid.uuid4().hex[:6].upper()}"
    with engine.begin() as conn:
        conn.execute(insert(users_table).values(
            user_id=user_id, name=name.strip(), role=role.upper(),
            address=encrypt_data(address.strip()), 
            dob=encrypt_data(dob.strip()),
            email=encrypt_data(email.strip()), 
            phone_number=encrypt_data(phone_number.strip()),
            created_at=datetime.utcnow()
        ))
    logger.info("Created user %s (%s, %s)", user_id, name, role)
    return user_id


def get_users(role: str | None = None) -> list[dict]:
    with engine.connect() as conn:
        stmt = select(users_table).order_by(users_table.c.name)
        if role:
            stmt = stmt.where(users_table.c.role == role.upper())
        rows = conn.execute(stmt).fetchall()
    users = []
    for row in rows:
        d = dict(row._mapping)
        d['address'] = decrypt_data(d['address'])
        d['dob'] = decrypt_data(d['dob'])
        d['email'] = decrypt_data(d['email'])
        d['phone_number'] = decrypt_data(d['phone_number'])
        users.append(d)
    return users


def get_user_profile(user_id: str) -> dict | None:
    """Fetch a single user's full profile."""
    with engine.connect() as conn:
        row = conn.execute(
            select(users_table).where(users_table.c.user_id == user_id)
        ).fetchone()
    if not row:
        return None
    d = dict(row._mapping)
    d['address'] = decrypt_data(d['address'])
    d['dob'] = decrypt_data(d['dob'])
    d['email'] = decrypt_data(d['email'])
    d['phone_number'] = decrypt_data(d['phone_number'])
    return d


def delete_user(user_id: str) -> None:
    """Delete a user from the system."""
    with engine.begin() as conn:
        conn.execute(users_table.delete().where(users_table.c.user_id == user_id))
    logger.info("Deleted user %s", user_id)


def apply_change(user_id: str, change_type: str, new_value: str) -> bool:
    """
    Apply an approved change to the user's profile.
    """
    col_name = _CHANGE_TYPE_TO_COLUMN.get(change_type)
    if not col_name:
        # If Contact change, maybe check if new_value has @ to update email or phone
        if change_type == "Contact / Email Change":
            if "@" in new_value:
                col_name = "email"
            else:
                col_name = "phone_number"
        else:
            logger.warning("Unknown change_type '%s', cannot apply.", change_type)
            return False

    prof = get_user_profile(user_id)
    if not prof:
        logger.warning("User '%s' not found, cannot apply change.", user_id)
        return False

    # Only encrypt sensitive fields, not 'name'
    _SENSITIVE_COLUMNS = {"address", "dob", "email", "phone_number"}
    store_value = encrypt_data(new_value.strip()) if col_name in _SENSITIVE_COLUMNS else new_value.strip()

    with engine.begin() as conn:
        conn.execute(
            update(users_table)
            .where(users_table.c.user_id == user_id)
            .values(**{col_name: store_value, "updated_at": datetime.utcnow()})
        )
    logger.info("Applied %s for %s: %s → '%s'",
                change_type, user_id, col_name, new_value[:50])
    return True


# ─── Request CRUD ─────────────────────────────────────────────────────────────

def save_request(
    customer_id: str, change_type: str,
    old_value: str, new_value: str,
    extracted_value: str, confidence_score: float,
    ai_summary: str, score_breakdown: str,
    ai_explanation: str = "",
    created_by: str = "",
) -> str:
    request_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(insert(requests_table).values(
            request_id=request_id, customer_id=customer_id,
            change_type=change_type, 
            old_value=encrypt_data(old_value),
            new_value=encrypt_data(new_value), 
            extracted_value=encrypt_data(extracted_value),
            confidence_score=confidence_score,
            ai_status="AI_VERIFIED_PENDING_HUMAN", decision="PENDING",
            ai_summary=ai_summary, score_breakdown=score_breakdown,
            ai_explanation=ai_explanation,
            created_by=created_by, timestamp=datetime.utcnow()))
    logger.info("Saved request %s (by %s)", request_id, created_by)
    return request_id


def get_pending_requests() -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(requests_table)
            .where(requests_table.c.decision == "PENDING")
            .order_by(requests_table.c.timestamp.desc())
        ).fetchall()
    reqs = []
    for row in rows:
        d = dict(row._mapping)
        d['old_value'] = decrypt_data(d['old_value'])
        d['new_value'] = decrypt_data(d['new_value'])
        d['extracted_value'] = decrypt_data(d['extracted_value'])
        reqs.append(d)
    return reqs


def get_requests_by_user(user_id: str) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(requests_table)
            .where(requests_table.c.created_by == user_id)
            .order_by(requests_table.c.timestamp.desc())
        ).fetchall()
    reqs = []
    for row in rows:
        d = dict(row._mapping)
        d['old_value'] = decrypt_data(d['old_value'])
        d['new_value'] = decrypt_data(d['new_value'])
        d['extracted_value'] = decrypt_data(d['extracted_value'])
        reqs.append(d)
    return reqs


def get_all_requests() -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(requests_table).order_by(requests_table.c.timestamp.desc())
        ).fetchall()
    reqs = []
    for row in rows:
        d = dict(row._mapping)
        d['old_value'] = decrypt_data(d['old_value'])
        d['new_value'] = decrypt_data(d['new_value'])
        d['extracted_value'] = decrypt_data(d['extracted_value'])
        reqs.append(d)
    return reqs


def update_decision(request_id: str, decision: str,
                    remarks: str = "", decision_by: str = "") -> None:
    with engine.begin() as conn:
        conn.execute(
            update(requests_table)
            .where(requests_table.c.request_id == request_id)
            .values(
                decision=decision, checker_remarks=remarks,
                decision_by=decision_by,
                ai_status="HUMAN_DECISION_RECORDED",
                decision_ts=datetime.utcnow()))
    logger.info("Request %s → %s (by %s)", request_id, decision, decision_by)
