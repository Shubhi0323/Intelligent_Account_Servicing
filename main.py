"""
IASW – Intelligent Account Servicing Workflow
Main Streamlit Application with RBAC
"""
import json, logging, sys, os, re, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from core.database import (init_db, get_pending_requests,
                           get_all_requests, get_requests_by_user,
                           update_decision, create_user, get_users,
                           apply_change, delete_user)
from core.address_validator import validate_address
from core.graph import run_pipeline
from core.config import CHANGE_TYPES, USE_MOCK_OCR, USE_OCR_API
from core.crypto_utils import mask_email, mask_phone, mask_address, mask_dob

logger = logging.getLogger(__name__)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="IASW", page_icon="🏦", layout="wide",
                   initial_sidebar_state="expanded")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #0d1117; }
  .iasw-header {
    background: linear-gradient(135deg, #1a2332 0%, #0f2027 50%, #203a43 100%);
    border: 1px solid #2d4059; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  .iasw-header h1 {
    font-size: 2rem; font-weight: 800; margin:0;
    background: linear-gradient(90deg, #63b3ed, #68d391);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .iasw-header p { color: #94a3b8; margin: 0.3rem 0 0; font-size: 0.95rem; }
  .card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem; transition: border-color 0.2s;
  }
  .card:hover { border-color: #63b3ed; }
  .card h3 { color: #e2e8f0; font-size: 1rem; font-weight: 700; margin: 0 0 0.4rem; }
  .card p  { color: #8b949e; font-size: 0.85rem; margin: 0; }
  .badge {
    display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.5px;
  }
  .badge-pass   { background: #1a3d2b; color: #68d391; border: 1px solid #276749; }
  .badge-flag   { background: #3d2e1a; color: #f6ad55; border: 1px solid #744210; }
  .badge-fail   { background: #3d1a1a; color: #fc8181; border: 1px solid #742a2a; }
  .badge-pending { background: #1a2d3d; color: #63b3ed; border: 1px solid #2a4365; }
  .section-label {
    color: #63b3ed; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.4rem;
  }
  .stTabs [data-baseweb="tab-list"] {
    background: #161b22; border-radius: 10px; padding: 4px; gap: 4px;
    border: 1px solid #30363d;
  }
  .stTabs [data-baseweb="tab"] { color: #8b949e !important; font-weight: 600; border-radius: 8px; }
  .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #1e3a5f, #1a3d2b) !important; color: #e2e8f0 !important; }
  [data-testid="metric-container"] {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 0.8rem 1rem;
  }
  [data-testid="metric-container"] label { color: #8b949e !important; }
  [data-testid="metric-container"] [data-testid="metric-value"] { color: #e2e8f0 !important; }
  div[data-testid="stButton"] > button { border-radius: 8px; font-weight: 600; }
  hr { border-color: #30363d !important; }
  .role-user  { background: #1a2d3d; color: #63b3ed; border: 1px solid #2a4365; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
  .role-admin { background: #3d1a2d; color: #f687b3; border: 1px solid #742a5a; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

init_db()

# ─── Sidebar: Role & User Selection ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 Session")
    role = st.selectbox("Select Role", ["USER", "ADMIN"], key="role_select")
    users_list = get_users(role=role)
    if not users_list:
        st.warning("No users found for this role.")
        st.stop()
    user_map = {f"{u['name']} ({u['user_id']})": u for u in users_list}
    selected_label = st.selectbox("Select User", list(user_map.keys()), key="user_select")
    current_user = user_map[selected_label]
    role_cls = "role-admin" if role == "ADMIN" else "role-user"
    st.markdown(f'Logged in as: **{current_user["name"]}** <span class="{role_cls}">{role}</span>',
                unsafe_allow_html=True)
    st.markdown("---")
    ocr_mode = "OCR.space API" if (USE_OCR_API and not USE_MOCK_OCR) else "Mock OCR"
    st.caption(f"OCR Mode: {ocr_mode}")

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="iasw-header">
  <h1>🏦 Intelligent Account Servicing Workflow</h1>
  <p>AI-powered document verification with HITL approval &nbsp;|&nbsp;
     {current_user['name']} &nbsp;
     <span class="{role_cls}">{role}</span></p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# USER VIEW
# ═══════════════════════════════════════════════════════════════════════════════
if role == "USER":
    tab_intake, tab_my = st.tabs(["📋 Submit Request", "📂 My Requests"])

    # ── Tab 1: Customer Intake ────────────────────────────────────────────────
    with tab_intake:
        st.markdown("### Submit a Change Request")
        col_form, col_guide = st.columns([3, 1], gap="large")
        with col_form:
            with st.form("intake_form", clear_on_submit=False):
                st.markdown('<p class="section-label">Customer Details</p>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                customer_id = c1.text_input("Customer ID", value=current_user["user_id"], disabled=True)
                change_type = c2.selectbox("Change Type", CHANGE_TYPES)

                st.markdown('<p class="section-label">Change Details</p>', unsafe_allow_html=True)
                c3, c4 = st.columns(2)
                if change_type == "Legal Name Change":
                    real_old_value = current_user.get("name", "")
                    old_value = c3.text_input("Current Name", value=real_old_value, disabled=True)
                    new_value = c4.text_input("Requested New Name", placeholder="e.g. Priya Ravi Sharma")
                elif change_type == "Address Change":
                    real_old_value = current_user.get("address", "")
                    old_value = c3.text_area("Current Address", value=mask_address(real_old_value), height=80, disabled=True)
                    new_value = c4.text_area("New Address", placeholder="e.g. 42, Sector 18, Noida - 201301", height=80)
                elif change_type == "Date of Birth Change":
                    real_old_value = current_user.get("dob", "")
                    old_value = c3.text_input("Current DOB (DD-MM-YYYY)", value=mask_dob(real_old_value), disabled=True)
                    new_value = c4.text_input("Correct DOB (DD-MM-YYYY)", placeholder="e.g. 05-07-1989")
                else:
                    curr_contact = current_user.get("email", "") or current_user.get("phone_number", "")
                    real_old_value = curr_contact
                    if "@" in curr_contact:
                        masked_contact = mask_email(curr_contact)
                    else:
                        masked_contact = mask_phone(curr_contact)
                    old_value = c3.text_input("Current Email / Phone", value=masked_contact, disabled=True)
                    new_value = c4.text_input("New Email / Phone", placeholder="e.g. ravi.sharma@example.com")

                st.markdown('<p class="section-label">Supporting Document</p>', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("Upload image (JPG, PNG, BMP, TIFF)",
                    type=["png","jpg","jpeg","bmp","tiff","webp"])
                submitted = st.form_submit_button("🚀 Submit for AI Verification",
                                                  use_container_width=True, type="primary")

            if submitted:
                errors = []
                if not customer_id.strip(): errors.append("Customer ID required.")
                if not real_old_value.strip(): errors.append("Current value required.")
                if not new_value.strip():   errors.append("New value required.")
                if uploaded_file is None:   errors.append("Upload a document image.")
                if errors:
                    for e in errors: st.error(e)
                else:
                    with st.spinner("🤖 AI LangGraph pipeline processing…"):
                        # Address Pre-validation (gate before graph)
                        is_valid_address = True
                        if change_type == "Address Change":
                            addr_res = validate_address(new_value.strip())
                            if not addr_res["found"]:
                                is_valid_address = False
                                st.error(f"❌ Address Validation Failed: {addr_res.get('error', 'Address not found in OpenStreetMap.')} Please provide a real-world address.")
                        
                        if is_valid_address:
                            fb = uploaded_file.read()
                            result = run_pipeline({
                                "customer_id":   customer_id.strip(),
                                "customer_name": current_user.get("name", ""),
                                "change_type":   change_type,
                                "old_value":     real_old_value.strip(),
                                "new_value":     new_value.strip(),
                                "file_bytes":    fb,
                                "created_by":    current_user["user_id"],
                            })
                    
                    if is_valid_address:
                        status = result["status"]
                    bcls = {"PASS":"badge-pass","FLAG":"badge-flag","FAIL":"badge-fail"}.get(status,"badge-pending")
                    risk = result.get("risk_level", "LOW")
                    risk_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")
                    st.success(f"Request submitted! **ID:** `{result['request_id']}`")
                    st.markdown(f'<div class="card"><h3>AI Result <span class="badge {bcls}">{status}</span> &nbsp; Fraud Risk: {risk_icon} {risk}</h3></div>', unsafe_allow_html=True)
                    m1,m2,m3,m4,m5 = st.columns(5)
                    m1.metric("Confidence", f"{result['percentage']}%")
                    m2.metric("Data Match", f"{round(result['data_match_score']*100,1)}%")
                    m3.metric("Authenticity", f"{round(result['authenticity_score']*100,1)}%")
                    m4.metric("Semantic", f"{round(result['semantic_score']*100,1)}%")
                    m5.metric("Fraud Score", f"{round(result.get('fraud_score',0)*100,1)}%")
                    st.progress(result["confidence"], text=f"Overall: {result['percentage']}%")
                    with st.expander("📊 Score Breakdown", expanded=True):
                        bd = result["breakdown"]
                        st.dataframe(pd.DataFrame({
                            "Component": ["Data Match","Authenticity","Semantic","OCR Quality","Biz Rules","Fraud Risk"],
                            "Score (%)": [round(bd["data_match"]*100,1), round(bd["authenticity"]*100,1),
                                          round(bd["semantic_similarity"]*100,1), round(bd["ocr_quality"]*100,1),
                                          round(bd["business_rules"]*100,1), bd.get("fraud_risk", "LOW")],
                            "Weight": ["35%","25%","20%","10%","10%","Penalty"],
                        }), use_container_width=True, hide_index=True)
                    with st.expander("🛡️ Fraud Detection", expanded=(risk != "LOW")):
                        fraud_flags = result.get("fraud_flags", [])
                        if fraud_flags:
                            st.warning(f"**Fraud flags detected:** {', '.join(fraud_flags)}")
                        else:
                            st.success("No fraud flags detected.")
                        for d in result.get("fraud_details", []):
                            st.markdown(f"- {d}")
                    with st.expander("🔍 Findings"):
                        for f in result["validation_findings"]: st.markdown(f"- {f}")
                    with st.expander("📝 AI Summary", expanded=True):
                        st.code(result["summary"], language=None)
                    st.info("⏳ Request is now **pending human review** by an Admin.")

        with col_guide:
            st.markdown("#### 📘 Accepted Documents")
            for ct, doc in {"Legal Name Change":"Marriage Certificate / Gazette",
                            "Address Change":"Utility Bill / Lease / Govt ID",
                            "Date of Birth Change":"Birth Certificate / PAN / Passport",
                            "Contact / Email Change":"Signed Consent Form"}.items():
                st.markdown(f'<div class="card"><h3>{ct}</h3><p>{doc}</p></div>', unsafe_allow_html=True)

    # ── Tab 2: My Requests ────────────────────────────────────────────────────
    with tab_my:
        st.markdown("### 📂 My Requests")
        if st.button("🔄 Refresh", key="refresh_my"):
            st.rerun()
        my_reqs = get_requests_by_user(current_user["user_id"])
        if not my_reqs:
            st.info("You haven't submitted any requests yet.")
        else:
            rows = []
            for r in my_reqs:
                sc = r["confidence_score"] or 0
                
                # Mask sensitive fields
                if r["change_type"] == "Address Change":
                    disp_old = mask_address(r['old_value'])
                    disp_new = mask_address(r['new_value'])
                elif r["change_type"] == "Date of Birth Change":
                    disp_old = mask_dob(r['old_value'])
                    disp_new = mask_dob(r['new_value'])
                elif r["change_type"] == "Contact / Email Change":
                    disp_old = mask_email(r['old_value']) if "@" in str(r['old_value']) else mask_phone(r['old_value'])
                    disp_new = mask_email(r['new_value']) if "@" in str(r['new_value']) else mask_phone(r['new_value'])
                else:
                    disp_old = r['old_value']
                    disp_new = r['new_value']

                rows.append({"Request ID": r["request_id"][:8]+"…", "Change Type": r["change_type"],
                    "Old → New": f"{str(disp_old)[:20]} → {str(disp_new)[:20]}",
                    "Confidence": f"{round(sc*100,1)}%",
                    "AI Score": "PASS" if sc>=0.75 else "FLAG" if sc>=0.50 else "FAIL",
                    "AI Summary": (r.get("ai_summary") or "N/A")[:80] + "…",
                    "Decision": r["decision"], "Submitted": str(r["timestamp"])[:16]})
            df = pd.DataFrame(rows)
            def _clr(val):
                if val == "APPROVED": return "background-color:#1a3d2b;color:#68d391;font-weight:bold"
                if val == "REJECTED": return "background-color:#3d1a1a;color:#fc8181;font-weight:bold"
                return "color:#63b3ed"
            st.dataframe(df.style.applymap(_clr, subset=["Decision"]),
                         use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN VIEW
# ═══════════════════════════════════════════════════════════════════════════════
else:
    tab_checker, tab_history, tab_users, tab_profiles = st.tabs([
        "🔍 Checker Dashboard", "📂 All Requests", "👥 Create User", "🏦 User Profiles"])

    # ── Tab 1: Checker Dashboard ──────────────────────────────────────────────
    with tab_checker:
        st.markdown("### 🔍 Checker Dashboard — Human-in-the-Loop")
        st.caption("Only you can approve or reject. The AI recommendation is advisory.")
        if st.button("🔄 Refresh Queue", key="refresh_checker"):
            st.rerun()
        pending = get_pending_requests()
        if not pending:
            st.success("✅ No pending requests!")
        else:
            st.markdown(f"**{len(pending)} request(s) awaiting review**")
            for req in pending:
                try: bd = json.loads(req["score_breakdown"]) if req["score_breakdown"] else {}
                except: bd = {}
                sc = req["confidence_score"] or 0.0
                pct = round(sc*100,1)
                status = "PASS" if sc>=0.75 else "FLAG" if sc>=0.50 else "FAIL"
                bcls = {"PASS":"badge-pass","FLAG":"badge-flag","FAIL":"badge-fail"}.get(status)
                created_label = req.get("created_by", "Unknown")

                with st.expander(f"📄 {req['change_type']} | {req['customer_id']} | {pct}% [{status}] | by {created_label}",
                                 expanded=(status=="FLAG")):
                    c1, c2 = st.columns([2,1])
                    with c1:
                        st.markdown(f"**Request ID:** `{req['request_id']}`")
                        st.markdown(f"**Submitted by:** `{created_label}` at {req['timestamp']}")
                        st.markdown(f"**Change Type:** {req['change_type']}")
                        st.markdown("---")
                        st.markdown(f"**Current Value:** `{req['old_value']}`")
                        st.markdown(f"**Requested New:** `{req['new_value']}`")
                        st.markdown(f"**AI Extracted:** `{req['extracted_value']}`")
                        st.markdown("---")
                        st.markdown(f'<span class="badge {bcls}">{status}</span>', unsafe_allow_html=True)
                        st.progress(sc, text=f"Confidence: {pct}%")
                        if bd:
                            for label, key in [("Data Match","data_match"),("Authenticity","authenticity"),
                                               ("Semantic","semantic_similarity"),("OCR","ocr_quality"),("Biz Rules","business_rules")]:
                                st.markdown(f"  - **{label}:** {round(bd.get(key,0)*100,1)}%")
                    with c2:
                        st.markdown("**AI Summary:**")
                        st.info(req.get("ai_summary") or "N/A")
                    st.markdown("---")
                    st.markdown("#### Decision")
                    remarks = st.text_area("Remarks", key=f"rem_{req['request_id']}", height=60)
                    ca, cr = st.columns(2)
                    with ca:
                        if st.button("✅ APPROVE", key=f"app_{req['request_id']}",
                                     type="primary", use_container_width=True):
                            update_decision(req["request_id"], "APPROVED", remarks,
                                            decision_by=current_user["user_id"])
                            # Actually update the customer record!
                            apply_change(req["customer_id"], req["change_type"], req["new_value"])
                            st.success(f"`{req['request_id']}` **APPROVED**")
                            st.balloons(); st.rerun()
                    with cr:
                        if st.button("❌ REJECT", key=f"rej_{req['request_id']}",
                                     use_container_width=True):
                            update_decision(req["request_id"], "REJECTED", remarks,
                                            decision_by=current_user["user_id"])
                            st.warning(f"`{req['request_id']}` **REJECTED**"); st.rerun()

    # ── Tab 2: All Requests (Audit) ──────────────────────────────────────────
    with tab_history:
        st.markdown("### 📂 Full Audit History")
        if st.button("🔄 Refresh", key="refresh_hist"):
            st.rerun()
        all_reqs = get_all_requests()
        if not all_reqs:
            st.info("No requests yet.")
        else:
            total = len(all_reqs)
            approved = sum(1 for r in all_reqs if r["decision"]=="APPROVED")
            rejected = sum(1 for r in all_reqs if r["decision"]=="REJECTED")
            pend = sum(1 for r in all_reqs if r["decision"]=="PENDING")
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("Total", total); mc2.metric("Approved", approved)
            mc3.metric("Rejected", rejected); mc4.metric("Pending", pend)
            st.markdown("---")
            rows = []
            for r in all_reqs:
                sc = r["confidence_score"] or 0
                rows.append({"ID": r["request_id"][:8]+"…", "Customer": r["customer_id"],
                    "Type": r["change_type"], "Created By": r.get("created_by",""),
                    "Confidence": f"{round(sc*100,1)}%", "Decision": r["decision"],
                    "Decided By": r.get("decision_by",""), "Time": str(r["timestamp"])[:16]})
            df = pd.DataFrame(rows)
            def _clr(val):
                if val=="APPROVED": return "background-color:#1a3d2b;color:#68d391;font-weight:bold"
                if val=="REJECTED": return "background-color:#3d1a1a;color:#fc8181;font-weight:bold"
                return "color:#63b3ed"
            st.dataframe(df.style.applymap(_clr, subset=["Decision"]),
                         use_container_width=True, hide_index=True)

    # ── Tab 3: Create User ────────────────────────────────────────────────────
    with tab_users:
        st.markdown("### 👥 Create New User")
        st.caption("Add a new Maker (USER) or Checker (ADMIN) to the system.")

        with st.form("create_user_form"):
            cu1, cu2 = st.columns(2)
            new_name = cu1.text_input("Full Name", placeholder="e.g. Rahul Gupta")
            new_role = cu2.selectbox("Role", ["USER", "ADMIN"])
            
            cu3, cu4 = st.columns(2)
            new_dob = cu3.text_input("Date of Birth", placeholder="DD-MM-YYYY")
            new_phone = cu4.text_input("Phone Number", placeholder="9876543210")
            
            new_email = st.text_input("Email Address", placeholder="rahul@example.com")
            new_address = st.text_area("Home Address", placeholder="e.g. 12, Main Street...")

            create_btn = st.form_submit_button("➕ Create User", use_container_width=True, type="primary")

        if create_btn:
            is_valid = True
            
            # Name validation
            if not new_name.strip():
                st.error("Name is required.")
                is_valid = False
            elif not re.match(r"^[A-Za-z\s\.\'\-]+$", new_name.strip()):
                st.error("Invalid characters in Name. Only letters, spaces, and basic punctuation allowed.")
                is_valid = False
                
            # DOB validation
            if new_dob.strip():
                if not re.match(r"^\d{2}-\d{2}-\d{4}$", new_dob.strip()):
                    st.error("DOB must be in DD-MM-YYYY format.")
                    is_valid = False
                else:
                    try:
                        dob_date = datetime.strptime(new_dob.strip(), "%d-%m-%Y")
                        age = (datetime.now() - dob_date).days / 365.25
                        if age > 110:
                            st.error("Age cannot be greater than 110 years.")
                            is_valid = False
                        elif age < 0:
                            st.error("DOB cannot be in the future.")
                            is_valid = False
                    except ValueError:
                        st.error("Invalid date.")
                        is_valid = False
                
            # Phone validation
            if new_phone.strip() and not re.match(r"^\d{10}$", new_phone.strip()):
                st.error("Phone number must be exactly 10 digits.")
                is_valid = False
                
            # Email validation
            if new_email.strip() and not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", new_email.strip()):
                st.error("Invalid email address format.")
                is_valid = False
                
            # SQL Injection prevention & basic sanitization
            sql_inj_pattern = re.compile(r"(--|;|/\*|\*/|@@|@|char|nchar|varchar|nvarchar|alter|begin|cast|create|cursor|declare|delete|drop|end|exec|execute|fetch|insert|kill|select|sys|sysobjects|syscolumns|table|update)", re.IGNORECASE)
            
            if new_address.strip() and sql_inj_pattern.search(new_address.strip()):
                st.error("Invalid characters or potential SQL injection detected in Address.")
                is_valid = False

            if is_valid:
                if new_address.strip():
                    with st.spinner("Validating Address..."):
                        addr_res = validate_address(new_address.strip())
                        if not addr_res["found"]:
                            is_valid = False
                            st.error(f"❌ Address Validation Failed: {addr_res.get('error', 'Address not found in OpenStreetMap.')} Cannot create user.")
                
                if is_valid:
                    uid = create_user(new_name.strip(), new_role, new_address.strip(), new_dob.strip(), new_email.strip(), new_phone.strip())
                    st.success(f"User created! **ID:** `{uid}` | **Name:** {new_name} | **Role:** {new_role}")
                    time.sleep(2)
                    st.rerun()

    # ── Tab 4: User Profiles (Source of Truth) ────────────────────────────────
    with tab_profiles:
        st.markdown("### 🏦 User Profiles")
        st.caption("Live database of all users in the system. Approving a request directly updates these records.")
        if st.button("🔄 Refresh Profiles", key="refresh_cust"):
            st.rerun()
        
        all_users = get_users()
        if all_users:
            c_df = pd.DataFrame(all_users)
            # Reorder columns for better visibility
            cols = ["user_id", "name", "role", "address", "dob", "email", "phone_number"]
            c_df = c_df[cols]
            st.dataframe(c_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("#### ❌ Delete User")
            user_opts = [f"{u['name']} ({u['user_id']})" for u in all_users]
            user_to_delete = st.selectbox("Select user to remove", user_opts)
            if st.button("Delete User", type="primary"):
                uid = user_to_delete.split('(')[1].strip(')')
                delete_user(uid)
                st.success(f"User {uid} deleted!")
                st.rerun()
        else:
            st.info("No users found in database.")
