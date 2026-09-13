from pathlib import Path
from datetime import datetime
import hashlib
import uuid
import textwrap
import base64
import io

import streamlit.components.v1 as components

import cv2
import numpy as np
import qrcode
import streamlit as st
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

from supabase import create_client, Client


# ============================================================
# SUPABASE CLIENT
# ============================================================

@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


supabase = init_supabase()


# ============================================================
# PATHS
# ============================================================
# Only DATA_DIR is still needed — it's used as temporary local
# storage for face photos captured during registration, before
# they're converted into an embedding and stored permanently in
# Supabase. Everything else (users, students, embeddings,
# attendance sessions) now lives in Supabase instead of on disk.

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# BRANDING
# ============================================================

UNIVERSITY_NAME = "Usman Dan Fodiyo University Sokoto"
DEPARTMENT_NAME = "Department of Computer Science"


def get_logo_base64():
    """
    Returns the university logo as a base64 string so it can be
    embedded directly inside custom HTML cards. Returns None if
    the logo file hasn't been placed in the assets folder yet.
    """

    if not LOGO_PATH.exists():
        return None

    with open(LOGO_PATH, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def logo_img_tag(size=64):
    """
    Returns an <img> tag for the logo, or an empty string if the
    logo file is missing (so the layout doesn't break).
    """

    logo_base64 = get_logo_base64()

    if not logo_base64:
        return ""

    return (
        f'<img src="data:image/png;base64,{logo_base64}" '
        f'style="width:{size}px;height:{size}px;object-fit:contain;">'
    )


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=f"{DEPARTMENT_NAME} | Attendance",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

CSS = """
<style>

.stApp {
    background: #f5f7fb;
    color: #0f172a;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}


/* ============================================================
   TEXT COLOR SAFETY NET
   Plain Streamlit widgets (labels, captions, subheaders, metric
   text) don't have an explicit color set anywhere else in this
   stylesheet, so they normally inherit the browser/OS theme.
   That means they can render as invisible white-on-white if the
   viewer's system is in dark mode. This block forces them dark,
   scoped to the main content area only (not the sidebar, which
   intentionally keeps white text on its dark background).
============================================================ */

[data-testid="stMain"] p,
[data-testid="stMain"] span,
[data-testid="stMain"] label,
[data-testid="stMain"] small,
[data-testid="stMain"] strong,
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5,
[data-testid="stMain"] h6,
[data-testid="stCaptionContainer"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stWidgetLabel"] p {
    color: #0f172a !important;
}


/* ============================================================
   SIDEBAR
============================================================ */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #172554 100%
    );
}

section[data-testid="stSidebar"] * {
    color: white;
}

.sidebar-brand {
    text-align: center;
    padding: 15px 5px 30px 5px;
}

.sidebar-logo {
    font-size: 48px;
    font-weight: 900;
    color: #60a5fa;
    line-height: 1;
}

.sidebar-title {
    font-size: 23px;
    font-weight: 800;
    margin-top: 8px;
}

.sidebar-subtitle {
    font-size: 11px;
    opacity: 0.65;
    margin-top: 4px;
}

.user-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 20px;
}


/* ============================================================
   PAGE HEADER
============================================================ */

.page-title {
    font-size: 32px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 3px;
}

.page-subtitle {
    color: #64748b;
    font-size: 15px;
    margin-bottom: 25px;
}


/* ============================================================
   HERO
============================================================ */

.hero {
    background: linear-gradient(
        135deg,
        #1d4ed8,
        #312e81
    );
    color: white;
    padding: 35px;
    border-radius: 22px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(30,64,175,0.20);
}

.hero-title {
    font-size: 32px;
    font-weight: 800;
}

.hero-text {
    opacity: 0.88;
    margin-top: 7px;
    font-size: 15px;
}


/* ============================================================
   CARDS
============================================================ */

.dashboard-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 5px 20px rgba(15,23,42,0.05);
    margin-bottom: 18px;
}

.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 5px 20px rgba(15,23,42,0.05);
    min-height: 125px;
}

.metric-label {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
    margin-top: 7px;
}

.metric-value {
    color: #0f172a;
    font-size: 30px;
    font-weight: 800;
    margin-top: 4px;
}


/* ============================================================
   LOGIN
============================================================ */

.login-logo {
    text-align: center;
    font-size: 60px;
    font-weight: 900;
    color: #1d4ed8;
    margin-top: 20px;
}

.login-title {
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    color: #0f172a;
}

.login-subtitle {
    text-align: center;
    color: #64748b;
    margin-bottom: 30px;
}


/* ============================================================
   BUTTONS
============================================================ */

.stButton > button {
    background-color: #2563eb !important;
    color: #000000 !important;
    border: 1px solid #2563eb !important;
    border-radius: 8px;
    font-weight: 700;
    box-shadow: none !important;
    transition: none !important;
}

.stButton > button:hover,
.stButton > button:focus,
.stButton > button:active {
    background-color: #2563eb !important;
    color: #000000 !important;
    border-color: #2563eb !important;
    box-shadow: none !important;
    transform: none !important;
}

.stButton > button[kind="primary"] {
    background: #2563eb !important;
    color: #ffffff !important;
    border-color: #2563eb !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:focus,
.stButton > button[kind="primary"]:active {
    background: #2563eb !important;
    color: #ffffff !important;
    border-color: #2563eb !important;
}


/* ============================================================
   INPUTS
============================================================ */

.stTextInput input,
.stNumberInput input {
    border-radius: 10px;
}

div[data-baseweb="select"] {
    border-radius: 10px;
}


/* ============================================================
   STATUS
============================================================ */

.status-success {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    padding: 16px;
    border-radius: 12px;
    font-weight: 600;
}

.status-warning {
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
    padding: 16px;
    border-radius: 12px;
    font-weight: 600;
}

.status-error {
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    padding: 16px;
    border-radius: 12px;
    font-weight: 600;
}


/* ============================================================
   QR
============================================================ */

.qr-container {
    text-align: center;
    background: white;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 5px 20px rgba(15,23,42,0.05);
}


/* ============================================================
   INFO BOX
============================================================ */

.info-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
}

.info-title {
    color: #0f172a;
    font-size: 20px;
    font-weight: 750;
}

.info-text {
    color: #64748b;
    line-height: 1.6;
    margin-top: 8px;
}


/* ============================================================
   MOBILE
============================================================ */

@media (max-width: 768px) {

    .page-title {
        font-size: 25px;
    }

    .hero-title {
        font-size: 25px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
"""

st.html(CSS)


# ============================================================
# HTML HELPER
# ============================================================

def html(content):
    """
    Safely render HTML without Streamlit interpreting
    the indentation as a code block.
    """

    st.html(textwrap.dedent(content).strip())


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "logged_in": False,
    "username": None,
    "role": None,
    "page": "login",
    "active_session": None,
    "view_qr_session": None,
    "print_qr_session": None,
    "view_attendance_session": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# PASSWORD
# ============================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# ============================================================
# USER SYSTEM  (Supabase: "users" table)
# ============================================================

def authenticate(username, password):

    password_hash = hash_password(password)

    response = (
        supabase.table("users")
        .select("*")
        .eq("username", username)
        .eq("password_hash", password_hash)
        .execute()
    )

    if response.data:
        return response.data[0]

    return None


def get_staff_accounts():

    response = (
        supabase.table("users")
        .select("*")
        .eq("role", "staff")
        .order("name")
        .execute()
    )

    return response.data or []


def username_exists(username):

    response = (
        supabase.table("users")
        .select("id")
        .eq("username", username)
        .execute()
    )

    return bool(response.data)


def create_staff_account(name, username, password):

    supabase.table("users").insert(
        {
            "username": username,
            "password_hash": hash_password(password),
            "name": name,
            "role": "staff",
        }
    ).execute()


# ============================================================
# STUDENTS  (Supabase: "students" table)
# ============================================================

def load_students():

    response = (
        supabase.table("students")
        .select("*")
        .order("registered_at", desc=True)
        .execute()
    )

    return response.data or []


def upsert_student(name, admission_number):

    supabase.table("students").upsert(
        {
            "admission_number": admission_number,
            "name": name,
            "registered_at": datetime.now().isoformat(),
        },
        on_conflict="admission_number",
    ).execute()


# ============================================================
# FACE MODELS
# ============================================================

@st.cache_resource
def load_face_models():

    mtcnn = MTCNN(
        image_size=240,
        keep_all=True,
        min_face_size=60,
    )

    resnet = InceptionResnetV1(
        pretrained="vggface2"
    ).eval()

    return mtcnn, resnet


mtcnn, resnet = load_face_models()


# ============================================================
# FACE EMBEDDING
# ============================================================

def get_face_embedding(image):

    try:

        image = image.convert("RGB")

    except Exception:

        return None, "Unable to process image."

    faces, probabilities = mtcnn(
        image,
        return_prob=True,
    )

    if faces is None:

        return None, "No face detected."

    if len(faces) != 1:

        return (
            None,
            "Please provide an image containing exactly one face.",
        )

    probability = probabilities[0]

    if probability < 0.7:

        return (
            None,
            "Face detection confidence is too low.",
        )

    with torch.no_grad():

        embedding = resnet(
            faces
        )

    embedding = embedding.squeeze(0)

    return embedding, None


# ============================================================
# EMBEDDING STORAGE  (Supabase: "face_embeddings" table)
# ============================================================
# Embeddings are 512-number FaceNet vectors. Postgres can't
# store a PyTorch tensor directly, so we convert tensor <-> list
# of floats when saving/loading. The math and comparisons stay
# identical to before, only the storage layer changed.

def load_embeddings():

    response = supabase.table("face_embeddings").select("*").execute()

    embeddings = []

    for row in response.data or []:

        embedding_values = row.get("embedding")

        if not embedding_values:
            continue

        embedding_tensor = torch.tensor(
            embedding_values,
            dtype=torch.float32,
        )

        embeddings.append(
            (
                embedding_tensor,
                row.get("name"),
                row.get("admission_number"),
            )
        )

    return embeddings


def save_embedding(name, admission_number, embedding_tensor):

    supabase.table("face_embeddings").upsert(
        {
            "admission_number": admission_number,
            "name": name,
            "embedding": embedding_tensor.tolist(),
        },
        on_conflict="admission_number",
    ).execute()


# ============================================================
# REGISTER FACE
# ============================================================

def register_face(
    name,
    admission_number,
):

    folder_name = admission_number

    person_folder = (
        DATA_DIR
        / folder_name
        / folder_name
    )

    if not person_folder.exists():

        return (
            False,
            "Student image folder does not exist.",
        )

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    ]

    image_files = []

    for extension in extensions:

        image_files.extend(
            person_folder.glob(extension)
        )

    if len(image_files) < 5:

        return (
            False,
            "At least 5 images are required.",
        )

    valid_embeddings = []

    for image_path in image_files:

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception:

            continue

        embedding, error = get_face_embedding(
            image
        )

        if embedding is not None:

            valid_embeddings.append(
                embedding
            )

    if len(valid_embeddings) < 5:

        return (
            False,
            "Not enough valid face images. "
            "At least 5 clear face images are required.",
        )

    stacked_embeddings = torch.stack(
        valid_embeddings
    )

    average_embedding = torch.mean(
        stacked_embeddings,
        dim=0,
    )

    save_embedding(
        name,
        admission_number,
        average_embedding,
    )

    # The embedding is now safely stored in Supabase, so the
    # temporary local photos are no longer needed. Clean them up
    # so the app's local disk doesn't fill up with images that
    # would be wiped on the next redeploy anyway.
    for image_path in image_files:

        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            pass

    return (
        True,
        f"{name} was successfully registered.",
    )


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_face(
    image,
    threshold=0.8,
):

    embedding, error = get_face_embedding(
        image
    )

    if embedding is None:

        return {
            "success": False,
            "message": error,
        }

    registered_embeddings = load_embeddings()

    if not registered_embeddings:

        return {
            "success": False,
            "message": "No registered faces found.",
        }

    best_distance = float("inf")

    best_name = None

    best_admission = None

    for item in registered_embeddings:

        if len(item) < 3:

            continue

        saved_embedding, name, admission = item

        distance = torch.norm(
            embedding - saved_embedding
        ).item()

        if distance < best_distance:

            best_distance = distance
            best_name = name
            best_admission = admission

    if best_name is None:

        return {
            "success": False,
            "message": "No valid registered faces found.",
        }

    if best_distance <= threshold:

        return {
            "success": True,
            "name": best_name,
            "admission_number": best_admission,
            "distance": best_distance,
        }

    return {
        "success": False,
        "message": "Face not recognized.",
        "distance": best_distance,
    }


# ============================================================
# ATTENDANCE SESSION  (Supabase: "attendance_sessions" table)
# ============================================================

def save_attendance_session(session):

    payload = {
        "session_id": session.get("session_id"),
        "course": session.get("course"),
        "lecturer": session.get("lecturer"),
        "duration_minutes": session.get("duration_minutes"),
        "status": session.get("status", "active"),
        "created_by": session.get("created_by"),
        "attendance": session.get("attendance", []),
    }

    if session.get("created_at"):
        payload["created_at"] = session["created_at"]

    supabase.table("attendance_sessions").upsert(
        payload,
        on_conflict="session_id",
    ).execute()


def load_attendance_session(session_id):

    response = (
        supabase.table("attendance_sessions")
        .select("*")
        .eq("session_id", session_id)
        .execute()
    )

    if response.data:
        return response.data[0]

    return None


def get_all_sessions():

    response = (
        supabase.table("attendance_sessions")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    return response.data or []


def delete_attendance_session(session_id):

    supabase.table("attendance_sessions").delete().eq(
        "session_id",
        session_id,
    ).execute()


# ============================================================
# UI HELPERS
# ============================================================

def page_header(
    title,
    subtitle="",
):

    html(
        f"""
        <div class="page-title">
            {title}
        </div>

        <div class="page-subtitle">
            {subtitle}
        </div>
        """
    )


def metric_card(
    label,
    value,
):

    html(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value}
            </div>

        </div>
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    role = st.session_state.role
    username = st.session_state.username

    with st.sidebar:

        html(
            f"""
            <div class="sidebar-brand">

                {logo_img_tag(70)}

                <div class="sidebar-title">
                    {UNIVERSITY_NAME}
                </div>

                <div class="sidebar-subtitle">
                    {DEPARTMENT_NAME}
                </div>

            </div>
            """
        )

        html(
            f"""
            <div class="user-card">

                <strong>
                    {username}
                </strong>

                <br>

                <small>
                    {role.capitalize()} Account
                </small>

            </div>
            """
        )

        if role == "admin":

            if st.button(
                "Dashboard",
                use_container_width=True,
            ):

                st.session_state.page = (
                    "admin_dashboard"
                )

                st.rerun()

            if st.button(
                "Manage Staff",
                use_container_width=True,
            ):

                st.session_state.page = (
                    "manage_staff"
                )

                st.rerun()

            if st.button(
                "Students",
                use_container_width=True,
            ):

                st.session_state.page = (
                    "students"
                )

                st.rerun()

        elif role == "staff":

            if st.button(
                "Dashboard",
                use_container_width=True,
            ):

                st.session_state.page = (
                    "staff_dashboard"
                )

                st.rerun()

            if st.button(
                "New Attendance",
                use_container_width=True,
            ):

                st.session_state.page = (
                    "create_attendance"
                )

                st.rerun()

            if st.button(
                "Manage Attendance",
                use_container_width=True,
            ):

                st.session_state.page = (
                    "attendance_records"
                )

                st.rerun()

            if st.button(
                "Students",
                use_container_width=True,
            ):

                st.session_state.page = (
                    "students"
                )

                st.rerun()

        st.divider()

        if st.button(
            "Logout",
            use_container_width=True,
        ):

            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.page = "login"
            st.session_state.active_session = None

            st.rerun()


# ============================================================
# LOGIN
# ============================================================

def login_page():

    left, center, right = st.columns(
        [1, 1.2, 1]
    )

    with center:

        html(
            f"""
            <div class="login-logo">
                {logo_img_tag(90)}
            </div>

            <div class="login-title">
                {UNIVERSITY_NAME}
            </div>

            <div class="login-subtitle">
                {DEPARTMENT_NAME} — Smart Attendance Management System
            </div>
            """
        )

        with st.container(border=True):

            st.subheader(
                "Welcome back"
            )

            st.write(
                "Sign in to continue to your dashboard."
            )

            username = st.text_input(
                "Username",
                placeholder="Enter your username",
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
            )

            if st.button(
                "Sign in",
                use_container_width=True,
                type="primary",
            ):

                if not username or not password:

                    st.warning(
                        "Please enter your username and password."
                    )

                else:

                    user = authenticate(
                        username,
                        password,
                    )

                    if user:

                        st.session_state.logged_in = True
                        st.session_state.username = (
                            user["username"]
                        )
                        st.session_state.role = (
                            user["role"]
                        )

                        if user["role"] == "admin":

                            st.session_state.page = (
                                "admin_dashboard"
                            )

                        else:

                            st.session_state.page = (
                                "staff_dashboard"
                            )

                        st.rerun()

                    else:

                        st.error(
                            "Invalid username or password."
                        )

        st.write("")

        html(
            """
            <div class="info-card">

                <div class="info-title">
                    Student Portal
                </div>

                <div class="info-text">
                    Students do not need an account.
                    Register your face or mark attendance
                    directly through the student portal.
                </div>

            </div>
            """
        )

        if st.button(
            "Open Student Portal",
            use_container_width=True,
        ):

            st.session_state.page = (
                "student_home"
            )

            st.rerun()


# ============================================================
# ADMIN DASHBOARD
# ============================================================

def admin_dashboard():

    page_header(
        "Admin Dashboard",
        "Manage your attendance system from one place.",
    )

    staff_count = len(get_staff_accounts())

    students = load_students()

    sessions = get_all_sessions()

    html(
        f"""
        <div class="hero">

            <div class="hero-title">
                Welcome to {DEPARTMENT_NAME} ATMS
            </div>

            <div class="hero-text">
                Manage staff, students and attendance
                sessions efficiently.
            </div>

        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        metric_card(
            "Staff",
            staff_count,
        )

    with col2:

        metric_card(
            "Students",
            len(students),
        )

    with col3:

        metric_card(
            "Attendance Sessions",
            len(sessions),
        )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Manage Staff",
            use_container_width=True,
        ):

            st.session_state.page = (
                "manage_staff"
            )

            st.rerun()

    with col2:

        if st.button(
            "View Students",
            use_container_width=True,
        ):

            st.session_state.page = (
                "students"
            )

            st.rerun()


# ============================================================
# STAFF DASHBOARD
# ============================================================

def staff_dashboard():

    page_header(
        "Staff Dashboard",
        "Create and manage your attendance sessions.",
    )

    sessions = get_all_sessions()

    students = load_students()

    active_sessions = [
        session
        for session in sessions
        if session.get("status") == "active"
    ]

    total_attendance = sum(
        len(
            session.get(
                "attendance",
                [],
            )
        )
        for session in sessions
    )

    html(
        """
        <div class="hero">

            <div class="hero-title">
                Welcome back
            </div>

            <div class="hero-text">
                Create an attendance session and let
                students check in using QR + Face Verification.
            </div>

        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        metric_card(
            "Registered Students",
            len(students),
        )

    with col2:

        metric_card(
            "Active Sessions",
            len(active_sessions),
        )

    with col3:

        metric_card(
            "Total Attendance",
            total_attendance,
        )

    st.write("")

    if st.button(
        "Create New Attendance",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.page = (
            "create_attendance"
        )

        st.rerun()

    st.write("")

    if st.button(
        "Manage Attendance",
        use_container_width=True,
    ):

        st.session_state.page = (
            "attendance_records"
        )

        st.rerun()


# ============================================================
# MANAGE STAFF
# ============================================================

def manage_staff():

    page_header(
        "Manage Staff",
        "Add and manage staff accounts.",
    )

    with st.container(border=True):

        st.subheader(
            "Add New Staff"
        )

        col1, col2 = st.columns(2)

        with col1:

            name = st.text_input(
                "Full name"
            )

            username = st.text_input(
                "Username"
            )

        with col2:

            password = st.text_input(
                "Password",
                type="password",
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password",
            )

        if st.button(
            "Add Staff",
            type="primary",
        ):

            if (
                not name
                or not username
                or not password
            ):

                st.warning(
                    "Please complete all fields."
                )

            elif password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            elif username_exists(username):

                st.error(
                    "Username already exists."
                )

            else:

                create_staff_account(
                    name,
                    username,
                    password,
                )

                st.success(
                    "Staff account created successfully."
                )

                st.rerun()

    st.write("")

    st.subheader(
        "Staff Accounts"
    )

    staff = get_staff_accounts()

    if not staff:

        st.info(
            "No staff accounts found."
        )

    else:

        rows = []

        for user in staff:

            rows.append(
                {
                    "Name": user.get(
                        "name",
                        "",
                    ),
                    "Username": user.get(
                        "username",
                        "",
                    ),
                    "Role": "Staff",
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# STUDENTS PAGE
# ============================================================

def students_page():

    page_header(
        "Students",
        "View students registered in the system.",
    )

    students = load_students()

    if not students:

        st.info(
            "No students have registered yet."
        )

        return

    rows = []

    for student in students:

        rows.append(
            {
                "Name": student.get(
                    "name",
                    "",
                ),
                "Admission Number": student.get(
                    "admission_number",
                    "",
                ),
                "Registered": student.get(
                    "registered_at",
                    "",
                ),
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CREATE ATTENDANCE
# ============================================================

def create_attendance():

    page_header(
        "Create Attendance",
        "Create a session and generate a QR code for students.",
    )

    with st.container(border=True):

        course = st.text_input(
            "Course / Subject",
            placeholder="e.g. Machine Learning",
        )

        col1, col2 = st.columns(2)

        with col1:

            lecturer = st.text_input(
                "Lecturer",
                value=st.session_state.username,
            )

        with col2:

            duration = st.number_input(
                "Duration (minutes)",
                min_value=1,
                max_value=300,
                value=30,
            )

        if st.button(
            "Generate Attendance QR",
            type="primary",
            use_container_width=True,
        ):

            if not course:

                st.warning(
                    "Please enter the course name."
                )

            else:

                session_id = uuid.uuid4().hex[:12]

                session = {
                    "session_id": session_id,
                    "course": course,
                    "lecturer": lecturer,
                    "duration_minutes": duration,
                    "created_at": datetime.now().isoformat(),
                    "status": "active",
                    "created_by": st.session_state.username,
                    "attendance": [],
                }

                save_attendance_session(
                    session
                )

                st.session_state.active_session = (
                    session_id
                )

                st.success(
                    "Attendance session created successfully."
                )

    if st.session_state.active_session:

        session = load_attendance_session(
            st.session_state.active_session
        )

        if session:

            st.divider()

            show_qr_code(
                session
            )


# ============================================================
# QR CODE
# ============================================================

def qr_png_bytes(session):

    payload = f"ATMS|{session['session_id']}"

    qr = qrcode.make(payload)
    buffer = io.BytesIO()
    qr.save(buffer, format="PNG")

    return buffer.getvalue()


def print_qr_code(session):

    image_bytes = qr_png_bytes(session)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    course = str(session.get("course", "Attendance"))
    lecturer = str(session.get("lecturer", ""))
    created_at = str(session.get("created_at", ""))

    printable_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>ATMS Attendance QR</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: white;
                color: #000;
                text-align: center;
                margin: 0;
                padding: 24px;
            }}
            .sheet {{
                max-width: 520px;
                margin: 0 auto;
                border: 2px solid #000;
                padding: 24px;
            }}
            h1 {{ margin: 0 0 8px; }}
            h2 {{ margin: 0 0 8px; color: #2563eb; }}
            p {{ margin: 6px 0; }}
            img {{ width: 320px; max-width: 90%; margin: 20px auto; }}
            .session {{ font-size: 12px; word-break: break-all; }}
            .print-btn {{
                background: #2563eb;
                color: white;
                border: 0;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                cursor: pointer;
                margin-top: 12px;
            }}
            @media print {{
                .print-btn {{ display: none; }}
                body {{ padding: 0; }}
                .sheet {{ border: 2px solid #000; }}
            }}
        </style>
    </head>
    <body>
        <div class="sheet">
            <h1>{UNIVERSITY_NAME}</h1>
            <h2>{DEPARTMENT_NAME}</h2>
            <h2>{course}</h2>
            <p><strong>Lecturer:</strong> {lecturer}</p>
            <p>Scan this QR code to mark attendance.</p>
            <img src="data:image/png;base64,{image_base64}" alt="Attendance QR Code">
            <p class="session"><strong>Session:</strong> {session.get('session_id', '')}</p>
            <p><strong>Created:</strong> {created_at}</p>
            <button class="print-btn" onclick="window.print()">Print QR Code</button>
        </div>
    </body>
    </html>
    """

    components.html(
        printable_html,
        height=600,
        scrolling=False,
    )


def print_attendance_list(session):
    """
    Renders a printable attendance sheet showing the course,
    lecturer, date, and the list of students who marked
    attendance for this session.
    """

    course = str(session.get("course", "Attendance"))
    lecturer = str(session.get("lecturer", ""))
    created_at = str(session.get("created_at", ""))
    attendance = session.get("attendance", [])

    logo_html = logo_img_tag(70)

    rows_html = ""

    for index, record in enumerate(attendance, start=1):

        rows_html += f"""
        <tr>
            <td>{index}</td>
            <td>{record.get('name', '')}</td>
            <td>{record.get('admission_number', '')}</td>
            <td>{record.get('time', '')}</td>
        </tr>
        """

    if not rows_html:

        rows_html = (
            '<tr><td colspan="4" style="text-align:center;">'
            "No attendance recorded for this session."
            "</td></tr>"
        )

    printable_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Attendance Sheet</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: white;
                color: #000;
                margin: 0;
                padding: 24px;
            }}
            .sheet {{
                max-width: 720px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                margin-bottom: 20px;
            }}
            .header h1 {{
                margin: 6px 0 2px;
                font-size: 20px;
            }}
            .header h2 {{
                margin: 2px 0;
                font-size: 15px;
                color: #2563eb;
            }}
            .meta {{
                margin-bottom: 18px;
                font-size: 14px;
            }}
            .meta p {{
                margin: 4px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 12px;
            }}
            th, td {{
                border: 1px solid #333;
                padding: 8px;
                font-size: 13px;
                text-align: left;
            }}
            th {{
                background: #f1f5f9;
            }}
            .print-btn {{
                background: #2563eb;
                color: white;
                border: 0;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                cursor: pointer;
                margin-top: 20px;
            }}
            @media print {{
                .print-btn {{ display: none; }}
                body {{ padding: 0; }}
            }}
        </style>
    </head>
    <body>
        <div class="sheet">

            <div class="header">
                {logo_html}
                <h1>{UNIVERSITY_NAME}</h1>
                <h2>{DEPARTMENT_NAME}</h2>
            </div>

            <div class="meta">
                <p><strong>Course:</strong> {course}</p>
                <p><strong>Lecturer:</strong> {lecturer}</p>
                <p><strong>Date:</strong> {created_at}</p>
                <p><strong>Total Present:</strong> {len(attendance)}</p>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Name</th>
                        <th>Admission Number</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>

            <button class="print-btn" onclick="window.print()">Print Attendance Sheet</button>

        </div>
    </body>
    </html>
    """

    components.html(
        printable_html,
        height=700,
        scrolling=True,
    )


def show_qr_code(session):

    qr_array = np.array(
        qrcode.make(
            f"ATMS|{session['session_id']}"
        ).convert("RGB")
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        html(
            f"""
            <div class="qr-container">
                <h2>{session.get('course', 'Attendance')}</h2>
                <p><strong>Lecturer:</strong> {session.get('lecturer', '')}</p>
                <p>Students should scan this QR code to mark attendance.</p>
            </div>
            """
        )

        st.image(qr_array, width=300)

        st.caption(
            f"Session ID: {session['session_id']}"
        )

        st.write("")

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button(
                "Print QR Code",
                key=f"print_current_{session['session_id']}",
                use_container_width=True,
            ):
                print_qr_code(session)

        with col_b:
            if st.button(
                "Close Attendance",
                key=f"close_current_{session['session_id']}",
                use_container_width=True,
            ):
                session["status"] = "closed"
                save_attendance_session(session)
                st.session_state.active_session = None
                st.success("Attendance session closed.")
                st.rerun()


# ============================================================
# ATTENDANCE RECORDS
# ============================================================

def attendance_records():

    page_header(
        "Manage Attendance",
        "View, print, close or delete attendance sessions.",
    )

    sessions = get_all_sessions()

    # Staff should only manage sessions they created.
    if st.session_state.role == "staff":
        sessions = [
            session
            for session in sessions
            if session.get("created_by") == st.session_state.username
        ]

    if not sessions:
        st.info("You have not created any attendance sessions yet.")
        return

    for session in sessions:

        session_id = session.get("session_id", "")
        course = session.get("course", "Unknown Course")
        lecturer = session.get("lecturer", "")
        status = session.get("status", "closed")
        attendance = session.get("attendance", [])
        created_at = session.get("created_at", "")

        with st.container(border=True):

            col1, col2, col3 = st.columns([2.5, 1, 1])

            with col1:
                st.subheader(course)
                st.caption(f"Lecturer: {lecturer}")
                st.caption(f"Created: {created_at}")
                st.caption(f"Session ID: {session_id}")

            with col2:
                st.metric("Present", len(attendance))

            with col3:
                if status == "active":
                    st.success("ACTIVE")
                else:
                    st.info("CLOSED")

            st.write("")

            action1, action2, action3, action4 = st.columns(4)

            with action1:
                view_key = f"view_qr_{session_id}"
                if st.button("View QR", key=view_key, use_container_width=True):
                    st.session_state.view_qr_session = (
                        None
                        if st.session_state.get("view_qr_session") == session_id
                        else session_id
                    )
                    st.rerun()

            with action2:
                if st.button("Print QR", key=f"print_{session_id}", use_container_width=True):
                    st.session_state.print_qr_session = session_id
                    st.rerun()

            with action3:
                if st.button("View Attendance", key=f"records_{session_id}", use_container_width=True):
                    st.session_state.view_attendance_session = (
                        None
                        if st.session_state.get("view_attendance_session") == session_id
                        else session_id
                    )
                    st.rerun()

            with action4:
                if st.button("Delete", key=f"delete_{session_id}", use_container_width=True):
                    try:
                        delete_attendance_session(session_id)
                        if st.session_state.get("active_session") == session_id:
                            st.session_state.active_session = None
                        if st.session_state.get("view_qr_session") == session_id:
                            st.session_state.view_qr_session = None
                        if st.session_state.get("print_qr_session") == session_id:
                            st.session_state.print_qr_session = None
                        if st.session_state.get("view_attendance_session") == session_id:
                            st.session_state.view_attendance_session = None
                        st.success(f"{course} attendance session deleted.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Unable to delete session: {exc}")

            if st.session_state.get("view_qr_session") == session_id:
                st.divider()
                show_qr_code_readonly(session)

            if st.session_state.get("print_qr_session") == session_id:
                st.divider()
                st.write("Print preview")
                print_qr_code(session)
                if st.button("Hide Print Preview", key=f"hide_print_{session_id}", use_container_width=True):
                    st.session_state.print_qr_session = None
                    st.rerun()

            if st.session_state.get("view_attendance_session") == session_id:
                st.divider()
                st.subheader("Attendance Records")
                if attendance:
                    rows = []
                    for record in attendance:
                        rows.append({
                            "Name": record.get("name", ""),
                            "Admission Number": record.get("admission_number", ""),
                            "Time": record.get("time", ""),
                            "Face Distance": record.get("face_distance", ""),
                        })
                    st.dataframe(rows, use_container_width=True, hide_index=True)

                    if st.button(
                        "Print Attendance List",
                        key=f"print_attendance_{session_id}",
                        use_container_width=True,
                    ):
                        print_attendance_list(session)
                else:
                    st.info("No attendance has been marked for this session.")

            if status == "active":
                st.write("")
                if st.button("Close Attendance", key=f"manage_close_{session_id}", use_container_width=True):
                    session["status"] = "closed"
                    save_attendance_session(session)
                    st.success("Attendance session closed.")
                    st.rerun()


def show_qr_code_readonly(session):

    qr_array = np.array(
        qrcode.make(
            f"ATMS|{session['session_id']}"
        ).convert("RGB")
    )

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.image(qr_array, width=300)
        st.caption(f"Session ID: {session['session_id']}")


# ============================================================
# STUDENT HOME
# ============================================================

def student_home():

    page_header(
        "Student Portal",
        "Register your face or mark attendance.",
    )

    html(
        """
        <div class="hero">

            <div class="hero-title">
                Smart Attendance
            </div>

            <div class="hero-text">
                Fast attendance using QR code and
                secure face verification.
            </div>

        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        html(
            """
            <div class="info-card">

                <div class="info-title">
                    Student Registration
                </div>

                <div class="info-text">
                    New student? Register your details
                    and capture at least five clear
                    face images.
                </div>

            </div>
            """
        )

        if st.button(
            "Start Registration",
            use_container_width=True,
            type="primary",
        ):

            st.session_state.page = (
                "student_register"
            )

            st.rerun()

    with col2:

        html(
            """
            <div class="info-card">

                <div class="info-title">
                    Mark Attendance
                </div>

                <div class="info-text">
                    Scan your lecturer's QR code and
                    complete face verification.
                </div>

            </div>
            """
        )

        if st.button(
            "Mark Attendance",
            use_container_width=True,
        ):

            st.session_state.page = (
                "student_attendance"
            )

            st.rerun()

    st.divider()

    if st.button(
        "Back to Login",
        use_container_width=True,
    ):

        st.session_state.page = "login"

        st.rerun()


# ============================================================
# STUDENT REGISTRATION
# ============================================================

def student_register():

    page_header(
        "Student Registration",
        "Register your face using clear images.",
    )

    html(
        """
        <div class="status-warning">

            Please capture at least 5 clear images.
            Make sure only one face is visible in each image.

        </div>
        """
    )

    st.write("")

    name = st.text_input(
        "Full Name",
        placeholder="Enter your full name",
    )

    admission_number = st.text_input(
        "Admission Number",
        placeholder="e.g. mahmudyahaya235403019",
    )

    if not name or not admission_number:

        st.info(
            "Enter your name and admission number "
            "to begin registration."
        )

        if st.button(
            "Back",
            use_container_width=True,
        ):

            st.session_state.page = (
                "student_home"
            )

            st.rerun()

        return

    folder_name = admission_number.strip()

    person_folder = (
        DATA_DIR
        / folder_name
        / folder_name
    )

    person_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    extensions = [
        "*.jpeg",
        "*.jpg",
        "*.png",
    ]

    existing_images = []

    for extension in extensions:

        existing_images.extend(
            person_folder.glob(extension)
        )

    image_number = (
        len(existing_images) + 1
    )

    st.subheader(
        f"Capture Image {image_number}"
    )

    picture = st.camera_input(
        "Take a clear face photo",
        key=(
            f"registration_camera_"
            f"{image_number}"
        ),
    )

    if picture:

        image_path = (
            person_folder
            / f"frame{image_number}.jpeg"
        )

        with open(
            image_path,
            "wb",
        ) as file:

            file.write(
                picture.getbuffer()
            )

        st.success(
            f"Image {image_number} saved."
        )

        st.rerun()

    current_images = []

    for extension in extensions:

        current_images.extend(
            person_folder.glob(extension)
        )

    current_count = len(
        current_images
    )

    progress = min(
        current_count / 5,
        1.0,
    )

    st.progress(
        progress
    )

    st.write(
        f"Captured: **{current_count}/5 minimum images**"
    )

    if current_count >= 5:

        st.success(
            "You have enough images for registration."
        )

        if st.button(
            "Register My Face",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Processing face images..."
            ):

                # The student must exist before we can save a
                # face embedding for them, since face_embeddings
                # references students(admission_number).
                upsert_student(
                    name.strip(),
                    admission_number.strip(),
                )

                success, message = register_face(
                    name.strip(),
                    admission_number.strip(),
                )

            if success:

                st.success(
                    message
                )

                st.balloons()

                st.session_state.page = (
                    "student_home"
                )

                st.rerun()

            else:

                st.error(
                    message
                )

    st.write("")

    if st.button(
        "Back to Student Portal",
        use_container_width=True,
    ):

        st.session_state.page = (
            "student_home"
        )

        st.rerun()


# ============================================================
# QR DECODER
# ============================================================

def decode_qr(image_file):

    try:

        image_bytes = np.asarray(
            bytearray(
                image_file.getvalue()
            ),
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR,
        )

        if frame is None:

            return None

        detector = cv2.QRCodeDetector()

        data, points, _ = detector.detectAndDecode(
            frame
        )

        if data:

            return data

    except Exception:

        return None

    return None


# ============================================================
# STUDENT ATTENDANCE
# ============================================================

def student_attendance():

    page_header(
        "Mark Attendance",
        "Scan the lecturer's QR code and verify your face.",
    )

    html(
        """
        <div class="info-card">

            <div class="info-title">
                Step 1 — Scan QR Code
            </div>

            <div class="info-text">
                Point your camera at the attendance QR
                code displayed by your lecturer.
            </div>

        </div>
        """
    )

    qr_picture = st.camera_input(
        "Scan Attendance QR Code",
        key="qr_camera",
    )

    if not qr_picture:

        if st.button(
            "Back",
            use_container_width=True,
        ):

            st.session_state.page = (
                "student_home"
            )

            st.rerun()

        return

    qr_data = decode_qr(
        qr_picture
    )

    if not qr_data:

        st.error(
            "QR code could not be detected. "
            "Please scan the QR code clearly."
        )

        return

    if not qr_data.startswith(
        "ATMS|"
    ):

        st.error(
            "This is not a valid ATMS attendance QR code."
        )

        return

    session_id = qr_data.split(
        "|",
        1,
    )[1]

    session = load_attendance_session(
        session_id
    )

    if not session:

        st.error(
            "Attendance session does not exist."
        )

        return

    if session.get("status") != "active":

        st.error(
            "This attendance session is closed."
        )

        return

    st.success(
        f"Attendance session found: "
        f"{session.get('course', '')}"
    )

    st.write(
        f"**Lecturer:** "
        f"{session.get('lecturer', '')}"
    )

    st.divider()

    html(
        """
        <div class="info-card">

            <div class="info-title">
                Step 2 — Face Verification
            </div>

            <div class="info-text">
                Take a clear photo of your face.
                The system will compare it with
                your registered face.
            </div>

        </div>
        """
    )

    face_picture = st.camera_input(
        "Take a clear face photo",
        key="face_camera",
    )

    if not face_picture:

        return

    image = Image.open(
        face_picture
    ).convert("RGB")

    with st.spinner(
        "Verifying your face..."
    ):

        result = recognize_face(
            image,
            threshold=0.8,
        )

    if not result["success"]:

        st.error(
            result.get(
                "message",
                "Face verification failed.",
            )
        )

        if "distance" in result:

            st.caption(
                "Face distance: "
                f"{result['distance']:.4f}"
            )

        return

    name = result["name"]

    admission_number = result[
        "admission_number"
    ]

    distance = result["distance"]

    # Prevent duplicate attendance
    existing_attendance = session.get(
        "attendance",
        [],
    )

    already_marked = any(
        record.get(
            "admission_number"
        )
        == admission_number
        for record in existing_attendance
    )

    if already_marked:

        html(
            """
            <div class="status-warning">

                You have already marked attendance
                for this session.

            </div>
            """
        )

        return

    attendance_record = {

        "name": name,

        "admission_number":
            admission_number,

        "time":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "face_distance":
            round(
                distance,
                4,
            ),
    }

    session.setdefault(
        "attendance",
        []
    ).append(
        attendance_record
    )

    save_attendance_session(
        session
    )

    html(
        f"""
        <div class="status-success">

            Attendance Confirmed

            <br><br>

            <strong>Name:</strong>
            {name}

            <br>

            <strong>Admission Number:</strong>
            {admission_number}

            <br>

            <strong>Time:</strong>
            {attendance_record['time']}

        </div>
        """
    )

    st.balloons()

    st.write("")

    if st.button(
        "Return to Student Portal",
        use_container_width=True,
        type="primary",
    ):

        st.session_state.page = (
            "student_home"
        )

        st.rerun()


# ============================================================
# MAIN ROUTER
# ============================================================

def main():

    # --------------------------------------------------------
    # PUBLIC AREA
    # --------------------------------------------------------

    if not st.session_state.logged_in:

        page = st.session_state.page

        if page == "student_home":

            student_home()

        elif page == "student_register":

            student_register()

        elif page == "student_attendance":

            student_attendance()

        else:

            login_page()

        return

    # --------------------------------------------------------
    # LOGGED-IN AREA
    # --------------------------------------------------------

    show_sidebar()

    role = st.session_state.role
    page = st.session_state.page

    # --------------------------------------------------------
    # ADMIN
    # --------------------------------------------------------

    if role == "admin":

        if page == "admin_dashboard":

            admin_dashboard()

        elif page == "manage_staff":

            manage_staff()

        elif page == "students":

            students_page()

        else:

            admin_dashboard()

    # --------------------------------------------------------
    # STAFF
    # --------------------------------------------------------

    elif role == "staff":

        if page == "staff_dashboard":

            staff_dashboard()

        elif page == "create_attendance":

            create_attendance()

        elif page == "attendance_records":

            attendance_records()

        elif page == "students":

            students_page()

        else:

            staff_dashboard()

    # --------------------------------------------------------
    # UNKNOWN ROLE
    # --------------------------------------------------------

    else:

        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.page = "login"

        st.rerun()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()