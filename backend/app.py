import streamlit as st
import pandas as pd
import joblib
import io
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction")
st.write("Predict whether a student will **Pass** or **Fail**.")

# =====================================
# Load Models
# =====================================

MODEL_DIR = Path("../model")

if not MODEL_DIR.exists():
    MODEL_DIR = Path("model")

model = joblib.load(MODEL_DIR / "best_model.pkl")

MODEL_ACCURACY = 0.956019

# ==========================
# SESSION STATE / RESET LOGIC
# ==========================
if "show_result" not in st.session_state:
    st.session_state.show_result = False

def reset_result():
    """Called on every input change to hide a stale prediction."""
    st.session_state.show_result = False

# ==========================
# INPUT FORM (grouped into card-style sections)
# ==========================
with st.container(border=True):
    st.markdown("#### 🏫 School & Demographics")
    c1, c2, c3 = st.columns(3)
    with c1:
        school = st.selectbox("School", ["GP", "MS"], key="school", on_change=reset_result)
        address = st.selectbox("Address", ["Urban", "Rural"], key="address", on_change=reset_result)
    with c2:
        age = st.number_input("Age", min_value=15, max_value=22, value=17, key="age", on_change=reset_result)
        guardian = st.selectbox("Guardian", ["Mother", "Father", "Other"], key="guardian", on_change=reset_result)
    with c3:
        reason = st.selectbox("Reason for Choosing School", ["course", "home", "reputation", "other"], key="reason", on_change=reset_result)

with st.container(border=True):
    st.markdown("#### 👪 Family Background")
    c1, c2, c3 = st.columns(3)
    with c1:
        Mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"], key="Mjob", on_change=reset_result)
        Medu = st.number_input("Mother's Education (0-4)", min_value=0, max_value=4, value=2, key="Medu", on_change=reset_result)
    with c2:
        Fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"], key="Fjob", on_change=reset_result)
        Fedu = st.number_input("Father's Education (0-4)", min_value=0, max_value=4, value=2, key="Fedu", on_change=reset_result)
    with c3:
        parent_edu = (Medu + Fedu) / 2
        st.text_input("Parent Education Score", value=f"{parent_edu:.2f}", disabled=True)

with st.container(border=True):
    st.markdown("#### 📚 Academics")
    c1, c2, c3 = st.columns(3)
    with c1:
        studytime = st.number_input("Study Time (1-4)", min_value=1, max_value=4, value=2, key="studytime", on_change=reset_result)
        studytime_label = st.selectbox("Weekly Study Hours", ["<2h", "2-5h", "5-10h", ">10h"], key="studytime_label", on_change=reset_result)
    with c2:
        failures = st.number_input("Past Class Failures", min_value=0, max_value=4, value=0, key="failures", on_change=reset_result)
        absences = st.number_input("Absences", min_value=0, value=4, key="absences", on_change=reset_result)
    with c3:
        G1 = st.number_input("First Period Grade (G1)", min_value=0, max_value=20, value=10, key="G1", on_change=reset_result)
        G2 = st.number_input("Second Period Grade (G2)", min_value=0, max_value=20, value=10, key="G2", on_change=reset_result)
    c1, c2 = st.columns(2)
    with c1:
        paid = st.selectbox("Extra Paid Classes", ["No", "Yes"], key="paid", on_change=reset_result)
    with c2:
        higher = st.selectbox("Wants Higher Education", ["No", "Yes"], key="higher", on_change=reset_result)

with st.container(border=True):
    st.markdown("#### 🌐 Lifestyle & Social")
    c1, c2, c3 = st.columns(3)
    with c1:
        internet = st.selectbox("Internet Access", ["No", "Yes"], key="internet", on_change=reset_result)
        romantic = st.selectbox("Romantic Relationship", ["No", "Yes"], key="romantic", on_change=reset_result)
    with c2:
        Dalc = st.number_input("Workday Alcohol Consumption (1-5)", min_value=1, max_value=5, value=1, key="Dalc", on_change=reset_result)
        Walc = st.number_input("Weekend Alcohol Consumption (1-5)", min_value=1, max_value=5, value=1, key="Walc", on_change=reset_result)
    with c3:
        social_score = st.number_input("Social Score", min_value=2, max_value=10, value=6, key="social_score", on_change=reset_result)
        alc_consumption = (Dalc + Walc) / 2
        st.text_input("Alcohol Consumption Score", value=f"{alc_consumption:.2f}", disabled=True)

if st.button("🔮 Predict", use_container_width=True):
    st.session_state.show_result = True

# ==========================
# MAPPINGS
# ==========================
school_map = {"GP": 0, "MS": 1}
address_map = {"Urban": 0, "Rural": 1}
guardian_map = {"Mother": 0, "Father": 1, "Other": 2}
yn_map = {"No": 0, "Yes": 1}

school_enc = school_map[school]
address_enc = address_map[address]
guardian_enc = guardian_map[guardian]
paid_enc = yn_map[paid]
higher_enc = yn_map[higher]
internet_enc = yn_map[internet]
romantic_enc = yn_map[romantic]

# ==========================
# INPUT DATAFRAME (for model)
# ==========================
input_df = pd.DataFrame({
    "school": [school_enc],
    "age": [age],
    "address": [address_enc],
    "Mjob": [Mjob],
    "Fjob": [Fjob],
    "reason": [reason],
    "guardian": [guardian_enc],
    "studytime": [studytime],
    "failures": [failures],
    "paid": [paid_enc],
    "higher": [higher_enc],
    "internet": [internet_enc],
    "romantic": [romantic_enc],
    "absences": [absences],
    "G1": [G1],
    "G2": [G2],
    "studytime_label": [studytime_label],
    "parent_edu": [parent_edu],
    "social_score": [social_score],
    "alc_consumption": [alc_consumption]
})

# ==========================
# SUGGESTIONS LOGIC
# ==========================
def generate_suggestions(row, prediction):
    tips = []
    if row["failures"] > 0:
        tips.append("Past failures detected — recommend remedial classes and closer mentor check-ins.")
    if row["studytime"] <= 1:
        tips.append("Low weekly study time — suggest structured study schedule (aim 5h+/week).")
    if row["absences"] > 10:
        tips.append("High absence count — attendance counseling recommended.")
    if row["alc_consumption"] > 2.5:
        tips.append("Elevated alcohol consumption score — recommend counselor referral.")
    if row["higher"] == 0:
        tips.append("Student has not expressed interest in higher education — guidance session advised.")
    if row["G2"] < 10:
        tips.append("Recent grade (G2) below passing threshold — extra tutoring advised.")
    if row["internet"] == 0:
        tips.append("No internet access — provide offline study materials / library access.")
    if not tips:
        tips.append("Student profile shows healthy indicators — continue current support and monitoring.")
    if prediction == 0:
        tips.insert(0, "Overall risk of failing is high — prioritize an intervention plan immediately.")
    return tips


# ==========================
# PDF GENERATION
# ==========================
def build_pdf(inputs: dict, prediction_label: str, prob_pass: float, prob_fail: float,
              suggestions: list, accuracy):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                             topMargin=2 * cm, bottomMargin=2 * cm,
                             leftMargin=1.8 * cm, rightMargin=1.8 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=20, spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    normal = styles["Normal"]

    elements = []
    elements.append(Paragraph("Student Performance Prediction Report", title_style))
    elements.append(Paragraph(datetime.now().strftime("Generated on %B %d, %Y at %H:%M"), normal))
    elements.append(HRFlowable(width="100%", color=colors.grey, spaceBefore=8, spaceAfter=10))

    # Prediction summary
    elements.append(Paragraph("Prediction Result", h2))
    result_color = colors.green if prediction_label == "PASS" else colors.red
    result_style = ParagraphStyle("Result", parent=styles["Heading2"], textColor=result_color)
    elements.append(Paragraph(f"Predicted Outcome: {prediction_label}", result_style))
    prob_table = Table(
        [["Pass Probability", f"{prob_pass*100:.2f}%"],
         ["Fail Probability", f"{prob_fail*100:.2f}%"]],
        colWidths=[8 * cm, 6 * cm]
    )
    prob_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(prob_table)

    if accuracy is not None:
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(f"Model Accuracy (validation set): {accuracy*100:.2f}%", normal))

    # Student inputs grouped by category
    elements.append(Paragraph("Student Inputs", h2))
    grouped = inputs  # dict of dicts: category -> {field: value}
    for category, fields in grouped.items():
        elements.append(Paragraph(category, ParagraphStyle("Cat", parent=styles["Heading3"], spaceBefore=8)))
        data = [[k, str(v)] for k, v in fields.items()]
        t = Table(data, colWidths=[8 * cm, 7 * cm])
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f7f7f7")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 6))

    # Suggestions
    elements.append(Paragraph("Suggestions / Recommendations", h2))
    for s in suggestions:
        elements.append(Paragraph(f"• {s}", normal))
        elements.append(Spacer(1, 3))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ==========================
# PREDICTION
# ==========================
if st.session_state.show_result:

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    prediction_label = "PASS" if prediction == 1 else "FAIL"

    st.divider()
    st.subheader("Prediction Result")

    res_col, prob_col = st.columns([1, 1.5])
    with res_col:
        with st.container(border=True):
            if prediction == 1:
                st.success("✅ Student is predicted to PASS")
            else:
                st.error("❌ Student is predicted to FAIL")
            if MODEL_ACCURACY is not None:
                st.caption(f"Model accuracy: {MODEL_ACCURACY*100:.2f}%")

    with prob_col:
        with st.container(border=True):
            st.markdown("#### Prediction Probability")
            st.write(f"**Pass:** {probability[1]*100:.2f}%")
            st.progress(float(probability[1]))
            st.write(f"**Fail:** {probability[0]*100:.2f}%")
            st.progress(float(probability[0]))

    suggestions = generate_suggestions(input_df.iloc[0], prediction)

    with st.container(border=True):
        st.markdown("#### 💡 Suggestions")
        for s in suggestions:
            st.write(f"- {s}")

    # Grouped inputs for PDF
    grouped_inputs = {
        "School & Demographics": {
            "School": school, "Age": age, "Address": address,
            "Reason for School": reason, "Guardian": guardian,
        },
        "Family Background": {
            "Mother's Job": Mjob, "Father's Job": Fjob,
            "Mother's Education": Medu, "Father's Education": Fedu,
            "Parent Education Score": f"{parent_edu:.2f}",
        },
        "Academics": {
            "Study Time": studytime, "Weekly Study Hours": studytime_label,
            "Past Failures": failures, "Absences": absences,
            "G1": G1, "G2": G2, "Extra Paid Classes": paid,
            "Wants Higher Education": higher,
        },
        "Lifestyle & Social": {
            "Internet Access": internet, "Romantic Relationship": romantic,
            "Workday Alcohol": Dalc, "Weekend Alcohol": Walc,
            "Alcohol Consumption Score": f"{alc_consumption:.2f}",
            "Social Score": social_score,
        },
    }

    pdf_buffer = build_pdf(
        inputs=grouped_inputs,
        prediction_label=prediction_label,
        prob_pass=float(probability[1]),
        prob_fail=float(probability[0]),
        suggestions=suggestions,
        accuracy=MODEL_ACCURACY,
    )

    st.divider()
    st.download_button(
        label="📄 Download Report as PDF",
        data=pdf_buffer,
        file_name=f"student_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )