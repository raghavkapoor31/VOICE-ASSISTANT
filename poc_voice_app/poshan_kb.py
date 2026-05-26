"""Poshan Tracker knowledge base — used for RAG context.
Sources: MoWCD Poshan 2.0, NHM WIFS, NHM Vitamin A, NHM Anaemia, IMNCI,
         SNP Norms Sept-2025, Poshan Tracker Q&A Expert 2025, ICMR-NIN 2024.
"""

DOCS = [
    # ── BENEFICIARY TYPES ─────────────────────────────────────────────────────
    {
        "id": "beneficiary_types",
        "title": "Beneficiary Categories in Poshan Tracker",
        "content": """
Poshan Tracker registers four beneficiary types:
1. Children (0–6 years): Track weight, height, age in months, vaccination, attendance, mid-day meals.
2. Pregnant Women: Track LMP date, ANC visits, iron-folic tablets, weight gain, institutional delivery.
3. Lactating Mothers (0–6 months postpartum): Track breastfeeding, supplementary nutrition, health checkups.
4. Adolescent Girls (14–18 years): Track weight, height, iron tablets, anemia status, school attendance.

Each beneficiary requires: Name, Aadhaar number, date of birth, village, gram panchayat, block, district.

Poshan Tracker platform facts (2025):
- Available in 22 scheduled Indian languages + English.
- Offline functionality: up to 3–5 days without internet; auto-syncs when connectivity restored.
- Handles ~45 crore (450 million) transactions per day.
- Used by over 14 lakh Anganwadi Workers across India.
        """,
    },

    # ── CHILD GROWTH MONITORING ───────────────────────────────────────────────
    {
        "id": "child_growth",
        "title": "Child Growth Monitoring — Fields and WHO Norms",
        "content": """
Monthly growth monitoring for children 0–6 years:
- Weight (kg): Recorded monthly. Reference: WHO Child Growth Standards.
- Height/Length (cm): Recorded quarterly for children under 2 (lying down = length).
- Age (months): Calculated from date of birth.
- MUAC (Mid-Upper Arm Circumference): Green > 12.5 cm (normal), Yellow 11.5–12.5 cm (MAM), Red < 11.5 cm (SAM).

Malnutrition classification:
- Normal: Weight-for-age Z-score > -2
- Underweight / MAM: Z-score -2 to -3, MUAC 11.5–12.5 cm → Provide RUTF
- SAM (Severe): Z-score < -3, MUAC < 11.5 cm or bilateral pitting edema → Refer to NRC immediately
- Wasting: Low weight-for-height
- Stunting: Low height-for-age

WHO Weight-for-Age normal ranges (approximate):
- 0 months: 2.5–4.5 kg    | 3 months: 4.5–7.0 kg
- 6 months: 5.5–9.0 kg    | 9 months: 6.5–10.5 kg
- 12 months: 7.0–11.5 kg  | 14 months: 7.5–11.8 kg
- 18 months: 7.5–13.0 kg  | 24 months: 9.0–15.0 kg
- 36 months: 10.5–17.0 kg | 48 months: 12.0–19.5 kg
- 60 months: 13.5–22.0 kg

NRC (Nutrition Rehabilitation Centre): refer all SAM children for 14-day inpatient management.
        """,
    },

    # ── DATA ENTRY FIELDS ─────────────────────────────────────────────────────
    {
        "id": "data_entry_fields",
        "title": "Poshan Tracker App — Daily Data Entry Fields",
        "content": """
Daily data entry by Anganwadi Worker (AWW):

AWC Status:
- Anganwadi Centre open/closed
- Date and time of opening

Child Attendance:
- Child name, age, attendance (present/absent)
- Meal provided: Hot Cooked Meal (HCM) or Take-Home Ration (THR)

Growth Monitoring Entry:
- Child name (required)
- Age in months (auto-calculated from DOB)
- Weight in kg (required, monthly)
- Height in cm (required, quarterly)
- MUAC reading

Supplementary Nutrition Programme (SNP):
- THR distributed: quantity in kg
- HCM: number of children fed
- Type of food (rice, dal, egg, banana, etc.)

Vaccination Tracking:
- Vaccine name (BCG, OPV, DPT, Measles, etc.)
- Date administered
- Due date for next dose

IFA/Vitamin A distribution:
- Number of tablets/doses distributed per visit
- Record beneficiary-wise in Poshan Tracker
        """,
    },

    # ── REGISTRATION FIELDS ───────────────────────────────────────────────────
    {
        "id": "registration_fields",
        "title": "Beneficiary Registration — Required Fields",
        "content": """
Mandatory fields for new beneficiary registration:
- Full name of beneficiary
- Aadhaar number (12 digits) — for deduplication
- Date of birth
- Gender
- Mother's name
- Father's name / Husband's name
- Mobile number
- Village name
- Gram Panchayat
- Block
- District
- State
- Caste category (General / OBC / SC / ST)
- Religion
- BPL status (Below Poverty Line: Yes/No)
- Bank account number (for DBT — Direct Benefit Transfer)

For child registration:
- Child's Aadhaar or birth certificate number
- Birth order (1st, 2nd, 3rd child)
- Birth weight (kg)
- Type of delivery (institutional/home)

Deduplication: Aadhaar is the primary key. Duplicate entries are flagged automatically by the platform.
        """,
    },

    # ── IFA / WIFS SUPPLEMENTATION ────────────────────────────────────────────
    {
        "id": "ifa_wifs",
        "title": "Iron & Folic Acid (IFA) and WIFS Supplementation Guidelines",
        "content": """
IFA Supplementation by beneficiary group (NHM WIFS Guidelines):

Children 6–59 months:
- Syrup: 1 mg elemental iron per kg body weight per day.
- Give with meals to reduce GI side effects.

Children 5–10 years (school-age):
- 1 IFA tablet (45 mg iron + 400 mcg folic acid) weekly.
- Given every Monday (or first school day of the week).

Adolescent Girls / Boys 10–19 years (WIFS programme):
- 1 IFA tablet (60 mg iron + 500 mcg folic acid) weekly.
- 52 tablets per year (one per week throughout the year).
- Give after meals — specifically recommended on Mondays.
- If a dose is missed, give next day; do not double-dose.
- Albendazole 400 mg deworming given biannually (August and February).

Pregnant Women:
- 1 tablet daily: 100 mg elemental iron + 500 mcg folic acid.
- Start from first trimester as early as possible.
- Total: minimum 180 tablets during pregnancy.
- Give at bedtime or with meals to manage nausea.

Lactating Mothers:
- 1 IFA tablet daily for 180 days postpartum (6 months).

Recording in Poshan Tracker:
- Record number of tablets distributed per beneficiary per visit.
- Record compliance (taken / not taken / reason for non-compliance).

Side effects counselling:
- Darkening of stool is normal and harmless.
- Take with Vitamin C-rich food (amla, lemon) to improve absorption.
- Avoid tea/coffee within 1 hour of taking IFA tablet.
        """,
    },

    # ── ANAEMIA ───────────────────────────────────────────────────────────────
    {
        "id": "anaemia",
        "title": "Anaemia — Haemoglobin Thresholds and Management",
        "content": """
Anaemia definition — Haemoglobin (Hb) thresholds (WHO / NHM Anaemia Technical Handbook):

Children 6 months – 6 years:
- Normal: Hb ≥ 11.0 g/dL
- Mild anaemia: Hb 10.0–10.9 g/dL
- Moderate anaemia: Hb 7.0–9.9 g/dL
- Severe anaemia: Hb < 7.0 g/dL → urgent referral

Children 6–12 years:
- Normal: Hb ≥ 11.5 g/dL
- Anaemia: Hb < 11.5 g/dL

Adolescent Girls (non-pregnant, 12–14 years):
- Normal: Hb ≥ 12.0 g/dL
- Anaemia: Hb < 12.0 g/dL

Pregnant Women:
- Normal: Hb ≥ 11.0 g/dL
- Mild: Hb 10.0–10.9 g/dL
- Moderate: Hb 7.0–9.9 g/dL → intensify IFA, refer for evaluation
- Severe: Hb < 7.0 g/dL → urgent referral for transfusion

Lactating Mothers:
- Normal: Hb ≥ 12.0 g/dL

Management:
- Mild/Moderate: Therapeutic IFA (double dose or as prescribed by MO).
- Severe: Refer to PHC/CHC. Do not delay treatment.
- Counsel on dietary iron: dark leafy vegetables, meat, eggs, pulses.
- Avoid iron inhibitors: tea, coffee, phytate-rich foods with IFA tablet.

National Anaemia target under Poshan 2.0: reduce anaemia prevalence in children/women by 3% per year.
        """,
    },

    # ── VITAMIN A ─────────────────────────────────────────────────────────────
    {
        "id": "vitamin_a",
        "title": "Vitamin A Supplementation Schedule",
        "content": """
Vitamin A Supplementation Programme (NHM Guidelines):

Schedule:
- Dose 1: 1,00,000 IU (1 lakh IU) at 6–11 months — given with first measles vaccine.
- Doses 2–9: 2,00,000 IU (2 lakh IU) every 6 months for children 1–5 years.
- Total: 9 doses by age 5.

Key rules:
- Never give Vitamin A to children under 6 months.
- Minimum 1-month gap between any two Vitamin A doses.
- Do NOT give Vitamin A within 4 weeks of a previous high-dose supplement.
- Record dose number, date, and batch number in Poshan Tracker.

Signs of Vitamin A deficiency (VAD):
- Night blindness (most common early sign).
- Bitot's spots (grey foamy patches on white of eye).
- Xerophthalmia (dry, rough eyes).
- Increased severity of infections.

Action for VAD:
- Give therapeutic Vitamin A immediately (same dose schedule above).
- Refer to medical officer if eye signs are present.
- Counsel mothers to include Vitamin A-rich foods: green leafy vegetables, yellow/orange fruits, eggs, liver.

VAS (Vitamin A Supplementation) rounds:
- Conducted during National Immunisation Days (NID) and routine immunisation sessions.
- AWW records each child's dose in Poshan Tracker vaccination module.
        """,
    },

    # ── SNP NUTRITION NORMS ───────────────────────────────────────────────────
    {
        "id": "snp_norms",
        "title": "Supplementary Nutrition Programme (SNP) — Norms Sept 2025",
        "content": """
SNP Nutrition Norms (Revised September 2025, Ministry of WCD):

Hot Cooked Meal (HCM) — for children 3–6 years at Anganwadi:
- Energy: 500 kcal per day
- Protein: 12–15 g per day
- Micronutrients: Iron 8 mg, Vitamin A 500 mcg, Folic acid 100 mcg
- Frequency: minimum 25 days per month

Take-Home Ration (THR) — for children 6 months–3 years:
- Energy: 500 kcal per day (600 kcal for severely undernourished)
- Protein: 12–15 g per day
- Micronutrients: as per age-appropriate fortified food
- Distribute monthly — record quantity in kg per child

THR for Pregnant & Lactating Women:
- Energy: 600 kcal per day (additional to normal diet)
- Protein: 18–20 g per day
- Iron, folic acid, calcium fortified

THR for Severely Malnourished children:
- Energy: 800 kcal per day
- Protein: 20–25 g per day
- Give RUTF (Ready-to-Use Therapeutic Food) if available

Food types in SNP:
- Fortified rice / wheat / millets (ragi, jowar, bajra preferred)
- Dal, pulses
- Seasonal vegetables, green leafy vegetables
- Eggs (2/week where available)
- Banana, amla (Vitamin C source)

Recording: AWW must record in Poshan Tracker — number of HCM beneficiaries daily, THR quantity distributed monthly.
        """,
    },

    # ── VACCINATION ───────────────────────────────────────────────────────────
    {
        "id": "vaccination",
        "title": "National Immunisation Schedule (UIP) — Poshan Tracker",
        "content": """
Universal Immunisation Programme (UIP) schedule tracked in Poshan Tracker:

At birth:
- BCG (1 dose)
- OPV 0 (birth dose)
- Hepatitis B (birth dose)

6 weeks:
- OPV 1 + DPT 1 (or Pentavalent 1) + Rotavirus 1 + PCV 1

10 weeks:
- OPV 2 + DPT 2 (or Pentavalent 2) + Rotavirus 2 + PCV 2

14 weeks:
- OPV 3 + DPT 3 (or Pentavalent 3) + Rotavirus 3 + PCV 3 + IPV 1

9 months:
- Measles / MR dose 1 + Vitamin A dose 1 (1 lakh IU) + JE (in endemic areas)

16–24 months (Booster):
- DPT booster + OPV booster + Measles / MR dose 2 + Vitamin A dose 2 (2 lakh IU)

5–6 years (DPT booster 2):
- DPT booster 2

10 years and 16 years:
- TT (Tetanus Toxoid)

Recording in Poshan Tracker:
- Vaccine name, date administered, next due date, batch number.
- Missed doses flagged as "Drop-Out" — AWW must follow up within 30 days.

Cold chain: Vaccines must be stored 2–8°C. AWW checks temperature at pickup from PHC.
        """,
    },

    # ── MUAC ──────────────────────────────────────────────────────────────────
    {
        "id": "muac_protocol",
        "title": "MUAC Measurement — Protocol and Classification",
        "content": """
MUAC (Mid-Upper Arm Circumference) — standard measurement protocol:

How to measure:
1. Child should be standing or sitting upright.
2. Measure left arm (non-dominant).
3. Find midpoint between shoulder tip (acromion) and elbow tip (olecranon).
4. Mark midpoint with a pen.
5. Wrap MUAC tape snugly (not tight) at the marked midpoint.
6. Read to nearest 0.1 cm.

Classification:
- Green: ≥ 12.5 cm → Normal — continue monthly monitoring.
- Yellow: 11.5–12.4 cm → MAM — provide RUTF, counsel on feeding, monthly follow-up.
- Red: < 11.5 cm → SAM — refer to NRC immediately.

Additional SAM indicators (any ONE is sufficient for SAM diagnosis):
- MUAC < 11.5 cm, OR
- Weight-for-Height Z-score < -3 SD, OR
- Bilateral pitting edema (press thumb on top of foot for 3 seconds; depression = edema).

MUAC for pregnant women: < 21 cm = Chronic Energy Deficiency (CED) — intensify nutrition support.

Equipment: Use colour-coded MUAC tape from the Poshan kit distributed by ICDS.
Recording: Enter MUAC reading in the child's growth monitoring record in Poshan Tracker.
        """,
    },

    # ── IMNCI PROTOCOLS ───────────────────────────────────────────────────────
    {
        "id": "imnci_protocols",
        "title": "IMNCI — Integrated Management of Neonatal and Childhood Illness",
        "content": """
IMNCI Clinical Assessment Protocols (NHM IMNCI Chart Booklet):

Danger signs — refer IMMEDIATELY (any one present):
- Not able to drink or breastfeed
- Vomits everything
- Convulsions (fits) now or in last 24 hours
- Lethargic or unconscious
- Stridor (noisy breathing when calm)

Pneumonia — breathing thresholds:
- Fast breathing: ≥ 60 breaths/min (0–2 months), ≥ 50 breaths/min (2–12 months), ≥ 40 breaths/min (1–5 years).
- Chest in-drawing: lower chest wall sucks in with each breath.
- Action: Treat with oral amoxicillin; refer if chest in-drawing or danger signs present.

Diarrhoea and dehydration:
- No dehydration: continue feeding, give ORS after each loose stool.
- Some dehydration: give ORS 75 ml/kg over 4 hours in health facility.
- Severe dehydration: IV fluids + immediate referral.
- Persistent diarrhoea (≥ 14 days): refer to PHC.

Low blood sugar (hypoglycaemia):
- Risk: small/premature baby, not feeding well, hypothermic, sick newborn.
- Action: give 10% dextrose IV 2 ml/kg OR expressed breastmilk / glucose water if no IV access.
- Keep warm: Kangaroo Mother Care for low-birth-weight babies.

Malaria (endemic areas): test with RDT; treat with ACT if positive; refer severe malaria immediately.

Malnutrition (IMNCI classification):
- SAM: MUAC < 11.5 cm or edema → refer NRC.
- MAM: MUAC 11.5–12.4 cm → RUTF + monthly follow-up.
- Anaemia: assess pallor on palm/conjunctiva; give IFA.

AWW role: IMNCI-trained AWW screens using the checklist during home visits and Anganwadi sessions.
        """,
    },

    # ── BREASTFEEDING ─────────────────────────────────────────────────────────
    {
        "id": "breastfeeding",
        "title": "Breastfeeding and Infant Feeding Guidelines",
        "content": """
Optimal Infant and Young Child Feeding (IYCF) practices:

Early initiation (within 1 hour of birth):
- Initiate breastfeeding within 60 minutes of birth.
- Colostrum (first yellow milk) is the 'first vaccine' — rich in antibodies and nutrients. NEVER discard.

Exclusive breastfeeding (0–6 months):
- Breastmilk only — no water, no other liquids, no solid food.
- Feed on demand — at least 8 times per day (day + night).
- No bottles or pacifiers; use cup and spoon if mother cannot breastfeed directly.

Complementary feeding (6–24 months):
- Start at exactly 6 months — not earlier, not later.
- Continue breastfeeding up to 2 years and beyond.
- Feed 2–3 times/day at 6–8 months; 3–4 times/day at 9–11 months; 3–4 meals + 2 snacks at 12–24 months.
- Portion sizes: 2–3 tablespoons at 6 months; increase gradually to 250 ml by 12 months.

Food diversity (minimum 5 of 8 food groups daily by 6–23 months):
1. Grains, roots, tubers  2. Legumes/nuts  3. Dairy  4. Meat/fish/eggs
5. Vitamin A-rich fruits/veg  6. Other fruits  7. Other vegetables  8. Breastmilk

Common myths to address:
- "Baby needs water" — NO, breastmilk is 88% water (0–6 months).
- "Thin milk is bad" — NO, foremilk quenches thirst, hindmilk provides fat.

Recording in Poshan Tracker: Record breastfeeding initiation (Y/N) at birth, exclusive breastfeeding status at 6 months.
        """,
    },

    # ── ANC (ANTENATAL CARE) ──────────────────────────────────────────────────
    {
        "id": "anc",
        "title": "Antenatal Care (ANC) — Schedule and Checks",
        "content": """
ANC Schedule (new guidelines — minimum 4 visits, target 8 visits):

ANC 1 (before 12 weeks — first trimester):
- Register in Poshan Tracker + Mother and Child Protection (MCP) card.
- Blood group, Hb, blood pressure, weight, urine albumin + sugar.
- Start IFA tablet + calcium supplement.
- TT vaccination if not previously immunised.

ANC 2 (14–16 weeks):
- Blood pressure, weight, Hb review.
- Ensure IFA compliance; counsel on diet and rest.

ANC 3 (28–30 weeks):
- Hb (flag if < 11 g/dL for treatment intensification).
- Abdominal examination: fundal height, foetal heart rate.
- Blood pressure (flag if ≥ 140/90 = refer for pre-eclampsia).

ANC 4 (36 weeks):
- Birth preparedness: identify delivery facility (institutional delivery mandatory).
- Review MCP card completeness.
- Danger signs counselling: bleeding, severe headache, reduced foetal movement.

TT Vaccination (Tetanus Toxoid):
- TT1: early pregnancy if not previously immunised.
- TT2: 4 weeks after TT1.
- TT Booster: if received TT2 in previous pregnancy within 3 years.

Recording in Poshan Tracker:
- LMP date, EDD (Expected Delivery Date), ANC visit dates, BP, Hb, weight, IFA tablets given, TT status, delivery outcome.
        """,
    },

    # ── POSHAN TRACKER Q&A ────────────────────────────────────────────────────
    {
        "id": "poshan_tracker_qa",
        "title": "Poshan Tracker — Common AWW Questions and Answers",
        "content": """
Frequently asked questions by Anganwadi Workers (from Poshan Tracker Q&A Expert 2025):

Q: What to do if Poshan Tracker app shows 'sync failed'?
A: Work offline — all data is saved locally. Once connectivity restored, app will auto-sync. Do not re-enter data.

Q: Child was absent on weighing day — what to enter?
A: Mark attendance as absent. Do not enter a fabricated weight. Follow up on next visit.

Q: Can I register a child without Aadhaar?
A: Yes — use birth certificate number or MCP card number as temporary ID. Link Aadhaar later.

Q: How often should I record weight?
A: Monthly for all children 0–6 years. Never skip — even if child appears healthy.

Q: My tablet is not working — what do I do?
A: Record data on paper registers (keep printed backup). Enter into Poshan Tracker as soon as device is repaired. Report to CDPO.

Q: A child's weight decreased from last month — is that normal?
A: Weight should never decrease significantly. Investigate: recent illness? Poor feeding? Refer if weight is below MAM threshold. Report to supervisor.

Q: How many children should I weigh per month?
A: 100% of registered 0–6 year children must be weighed every month. Strive for 100% coverage.

Q: How to record a child who shifted to another Anganwadi?
A: Mark as "migrated" in Poshan Tracker with the destination village/AWC. Record will transfer to destination AWC.

Q: What is the difference between THR and HCM?
A: HCM (Hot Cooked Meal) = cooked at AWC, given to 3–6 year children. THR (Take-Home Ration) = dry ration given to 6 month–3 year children and pregnant/lactating women to cook at home.

Q: How do I enter Vitamin A dose?
A: Go to Vaccination module → Select child → Add Vitamin A dose number and date.
        """,
    },

    # ── POSHAN 2.0 / MISSION ──────────────────────────────────────────────────
    {
        "id": "poshan_mission",
        "title": "Poshan 2.0 Mission — Objectives and Key Interventions",
        "content": """
Poshan 2.0 (Pradhan Mantri Poshan Shakti Nirman) — Ministry of WCD:

Merged schemes under Poshan 2.0 (from April 2021):
1. ICDS (Integrated Child Development Services)
2. POSHAN Abhiyaan (National Nutrition Mission)
3. Scheme for Adolescent Girls (SAG)
4. National Crèche Scheme

Four pillars of Poshan 2.0:
1. Supplementary Nutrition Programme (SNP)
2. Early Childhood Care and Education (ECCE / Pre-school)
3. Preventive Health and Nutrition Behaviour Change (SBCC)
4. Poshan Tracker — digital real-time monitoring

Key targets (2025):
- Reduce stunting by 2% per year
- Reduce wasting by 2% per year
- Reduce Low Birth Weight (LBW) by 2% per year
- Reduce anaemia in women and children by 3% per year

Saksham Anganwadi concept:
- Upgraded Anganwadi Centres with smart TVs, tablets, kitchen gardens.
- All AWCs to have functional toilets, water, and electricity.

Jan Andolan (People's Movement):
- Community mobilisation — village health and nutrition days (VHNDs).
- Poshan Maah (Nutrition Month) — September every year.
- Poshan Pakhwada — fortnight-long activities in March and April.

MoWCD guidelines mandate: every AWW must submit growth monitoring data for 100% beneficiaries monthly via Poshan Tracker.
        """,
    },

    # ── ICMR-NIN DIETARY GUIDELINES 2024 ─────────────────────────────────────
    {
        "id": "dietary_guidelines",
        "title": "ICMR-NIN Dietary Guidelines for Indians 2024",
        "content": """
ICMR-NIN Dietary Guidelines for Indians (2024) — 17 key guidelines:

1. Eat a variety of foods to ensure a balanced diet.
2. Ensure provision of adequate diet to infants and young children.
3. Eat plenty of vegetables and fruits.
4. Use whole grains, millets and pulses for most of daily food intake.
5. Be physically active — minimum 45 minutes moderate activity daily.
6. Restrict salt intake to < 5 g per day (WHO recommended < 2 g sodium/day).
7. Consume safe and clean foods.
8. Adopt healthy cooking methods — boil, steam, roast; avoid deep frying.
9. Drink adequate water — 2–3 litres per day.
10. Maintain healthy body weight — BMI 18.5–22.9 for Indians.
11. If you choose to eat meat, eat fish/poultry; limit red meat; avoid processed meat.
12. Limit consumption of high fat, high sugar, high salt (HFSS) ultra-processed foods.
13. Promote exclusive breastfeeding for first 6 months.
14. Restrict sugar and sugary beverages.
15. Reduce consumption of saturated fats; avoid trans-fats entirely.
16. Encourage consumption of traditional Indian foods and millets (Shree Anna).
17. Read food labels for ingredients and nutrition information.

10 food groups for dietary diversity (children 6–23 months):
Grains | Legumes/nuts | Dairy | Meat/fish/eggs | Vitamin A veg/fruit |
Other fruits | Other vegetables | Fortified foods | Breastmilk | Oils/fats

Recommended energy intake (approximate, adults):
- Sedentary man: 2,130 kcal | Sedentary woman: 1,660 kcal
- Pregnant +350 kcal | Lactating +500 kcal
        """,
    },

    # ── HBNC ─────────────────────────────────────────────────────────────────
    {
        "id": "hbnc",
        "title": "Home-Based Newborn Care (HBNC) — First 42 Days Protocol",
        "content": """
Home-Based Newborn Care (HBNC) — NHM Protocol (ASHA-led, AWW coordinates):

Visit schedule for INSTITUTIONAL deliveries (6 visits):
- Day 3, Day 7, Day 14, Day 21, Day 28, Day 42 after birth.

Visit schedule for HOME deliveries (7 visits):
- Day 1 (within 24 hours), then Day 3, 7, 14, 21, 28, 42.

Additional visits required for: pre-term / low-birth-weight / SNCU-discharged babies.

At each visit, assess:
1. Breastfeeding — initiated within 1 hour? Exclusive?
2. Temperature — keep baby warm (28°C room or Kangaroo Mother Care for LBW).
3. Umbilical cord — clean and dry; no redness/discharge.
4. Eyes — no discharge; wipe with clean cloth.
5. Danger signs (refer IMMEDIATELY if any):
   - Not feeding / not crying / very small.
   - Difficulty breathing, fast breathing.
   - Fever (temp > 37.5°C) or hypothermia (< 35.5°C).
   - Convulsions.
   - Jaundice spreading below chest.
   - Pus/discharge from umbilicus or eyes.

Kangaroo Mother Care (KMC) for LBW (< 2.5 kg):
- Skin-to-skin contact between mother and baby, at least 8 hours/day.
- Continue breastfeeding; monitor temperature.

ASHA incentive: Rs. 250 on completing all 6/7 scheduled visits.
AWW role: coordinate visits, ensure immunization at Day 42 (BCG, OPV-0, Hepatitis B).
        """,
    },

    # ── DIARRHOEA / ORS / ZINC ────────────────────────────────────────────────
    {
        "id": "diarrhoea_ors_zinc",
        "title": "Diarrhoea Management — ORS and Zinc Protocol (Under 5)",
        "content": """
Diarrhoea management for children under 5 (IMNCI / NHM / Government of India guidelines):

PROBLEM: Diarrhoea is the 2nd leading cause of under-5 deaths in India.
         Dehydration and electrolyte loss are the main killers.

ORS (Oral Rehydration Salts):
- Use low-osmolarity ORS (WHO 2004 formulation — 245 mOsm/L).
- Mix 1 ORS sachet in 1 litre clean water.
- Give after each loose stool:
    • Child < 2 years: 50–100 ml ORS per stool.
    • Child 2–10 years: 100–200 ml ORS per stool.
- Continue until diarrhoea stops.
- Do NOT use soft drinks, fruit juice, or salt-sugar home solutions instead of ORS.

Zinc supplementation (reduces duration and severity):
- Children 2–6 months: 10 mg elemental zinc per day × 14 days.
- Children 6 months–5 years: 20 mg elemental zinc per day × 14 days.
- Give as dispersible tablet dissolved in breast milk or ORS.
- Complete full 14-day course even if diarrhoea stops earlier.

Continued feeding:
- Do NOT stop breastfeeding or food during diarrhoea.
- Offer more frequent, energy-dense feeds during and after illness.

Referral — take child to PHC/CHC IMMEDIATELY if:
- Sunken eyes, dry mouth, unable to drink, skin pinch goes back slowly (dehydration).
- Blood in stool (dysentery — treat with cotrimoxazole or ciprofloxacin).
- Diarrhoea lasting ≥ 14 days (persistent diarrhoea).
- Any IMNCI danger sign present.

AWW actions:
1. Distribute ORS packets and zinc tablets from Anganwadi kit.
2. Demonstrate ORS preparation to mother.
3. Counsel on continued feeding and hand hygiene.
4. Record in Poshan Tracker — diarrhoea episode, ORS/zinc given, referral if any.
        """,
    },

    # ── PMMVY ─────────────────────────────────────────────────────────────────
    {
        "id": "pmmvy",
        "title": "Pradhan Mantri Matru Vandana Yojana (PMMVY) — Eligibility and Benefits",
        "content": """
PMMVY — Maternity benefit cash transfer scheme by Ministry of WCD (from 1 Jan 2017):

Eligibility:
- Pregnant women and lactating mothers aged 19 years and above.
- First two living children.
- Second benefit only if 2nd child is a girl (PMMVY 2.0 revision).
- Must belong to disadvantaged category: SC/ST, disabled (≥40%), BPL ration card,
  PMJAY beneficiary, e-Shram registered, MGNREGA job-card holder,
  OR net family income < Rs. 8 lakh per year.
- Excludes women in regular government employment.

Benefit amounts (via DBT — Direct Benefit Transfer to bank account):

First child — Rs. 5,000 in 2 instalments:
  Instalment 1 (Rs. 3,000): After early pregnancy registration + at least 1 ANC check-up.
  Instalment 2 (Rs. 2,000): After child birth + completion of BCG, OPV, DPT, Hepatitis B (14 weeks immunisation).

Second child (if girl) — Rs. 6,000 in 1 instalment:
  Single payment after child birth + completion of 14-week immunisation schedule.

Documents required:
- MCP (Mother and Child Protection) card
- Aadhaar of mother and husband
- Bank passbook
- Institutional delivery certificate (for 2nd instalment)

AWW actions:
1. Identify eligible pregnant women during early pregnancy registration.
2. Counsel on ANC completion and institutional delivery.
3. Support filling PMMVY application form (online via PMMVY-CAS portal).
4. Track immunisation completion for 2nd instalment processing.
5. Follow up on DBT credit to bank account.

Record in Poshan Tracker: PMMVY registration status, instalment dates, amount received.
        """,
    },

    # ── ANEMIA MUKT BHARAT ────────────────────────────────────────────────────
    {
        "id": "anemia_mukt_bharat",
        "title": "Anemia Mukt Bharat (AMB) — Intensified Strategy",
        "content": """
Anemia Mukt Bharat (AMB) — Intensified National Iron Plus Initiative (NHM):
Target: Reduce anaemia prevalence by 3% per year across all age groups.

Six target beneficiary groups with colour-coded IFA tablets:

1. Children 6–59 months — PINK syrup:
   - 20 mg elemental iron + 100 mcg folic acid, BIWEEKLY (2 × per week).
   - 52 doses per year.

2. Children 5–10 years — PINK tablet:
   - 45 mg iron + 400 mcg folic acid, WEEKLY (Mondays).
   - 52 tablets per year.

3. Adolescents 10–19 years — BLUE tablet (WIFS):
   - 100 mg iron + 500 mcg folic acid, WEEKLY (Mondays).
   - 52 tablets per year + biannual Albendazole 400 mg (August + February).

4. Pregnant women — RED tablet:
   - 100 mg iron + 500 mcg folic acid, DAILY from 1st trimester.
   - Minimum 180 tablets during pregnancy.

5. Lactating mothers — RED tablet:
   - 100 mg iron + 500 mcg folic acid, DAILY for 180 days postpartum.

6. Reproductive-age women (non-pregnant, 20–49 years) — WEEKLY blue tablet:
   - 1 tablet per week throughout the year.

"Monday" = National Anaemia Control Day (Somvar IFA Diwas):
- All school/Anganwadi IFA distribution happens on Mondays.
- AWW supervises ingestion; do not send tablets home without supervision.

Point-of-care Hb testing:
- Use digital Hb meter at Anganwadi for beneficiaries aged 6m–49y at each visit.
- Record Hb value in Poshan Tracker.

Side effect counselling:
- Dark/black stool = NORMAL (iron excretion).
- Nausea: take after meals or at bedtime.
- Avoid tea/coffee within 1 hour of IFA dose.
- Vitamin C foods (amla, lemon) improve iron absorption.
        """,
    },

    # ── COMPLEMENTARY FEEDING DETAILED ───────────────────────────────────────
    {
        "id": "complementary_feeding",
        "title": "Complementary Feeding — Detailed Guidelines (6–24 months)",
        "content": """
Complementary feeding (CF) — when to start, what to give, how much (ICDS / WHO / IYCF):

START: Exactly at 6 completed months. Not earlier (gut not ready), not later (growth faltering risk).
Continue breastfeeding until 2 years and beyond.

Feeding frequency by age:
- 6–8 months: 2–3 meals/day + breastfeeding on demand.
- 9–11 months: 3–4 meals/day + 1–2 snacks + breastfeeding.
- 12–24 months: 3–4 meals/day + 2 snacks + breastfeeding.

Portion size progression:
- 6 months: 2–3 tablespoons per meal.
- 9 months: ½ bowl (125 ml).
- 12 months: ¾ bowl (175 ml).
- 24 months: 1 full bowl (250 ml).

Texture progression:
- 6–8 months: mashed / pureed / thick porridge.
- 9–11 months: finely chopped / mashed lumpy food.
- 12+ months: family food, soft chopped pieces.

Food diversity — minimum 5 of 8 food groups daily (6–23 months):
1. Grains, roots, tubers (rice, roti, potato)
2. Legumes and nuts (dal, peas, groundnuts)
3. Dairy (milk, curd, paneer)
4. Meat, fish, eggs
5. Vitamin A-rich vegetables/fruits (carrot, green leafy veg, mango, papaya)
6. Other fruits
7. Other vegetables
8. Breastmilk (counts as one food group)

Minimum acceptable diet = minimum feeding frequency + minimum food diversity.

Energy-dense foods to encourage:
- Add 1 tsp ghee/oil to each meal (increases energy by 45 kcal).
- Include egg (protein, zinc, Vitamin B12) — 3–4 per week if available.
- Ragi (finger millet): excellent calcium and iron source for Indian infants.

What NOT to give before 1 year:
- Honey (risk of botulism).
- Cow's milk as main drink (can give in cooking after 6 months).
- Salt and sugar in excess.
- Processed/junk foods.

AWW action: counsel mother at each monthly session on food diversity, portion, texture, and responsive feeding (feed slowly, look for hunger/fullness cues).
        """,
    },

    # ── VOICE COMMANDS / FIELD EXTRACTION ────────────────────────────────────
    {
        "id": "voice_commands",
        "title": "Common Anganwadi Worker Voice Inputs — Expected Fields",
        "content": """
When an Anganwadi worker speaks, extract these key fields:

For child weight entry:
- Child name (string)
- Age in months (number)
- Weight in kg (decimal, e.g. 8.2)
- Height in cm (optional)
- Village name (optional)
- Any nutrition concern mentioned

For beneficiary registration:
- Name, age/DOB, mother's name, village, Aadhaar

For vaccination:
- Child name, vaccine name, date

For attendance:
- Number of children present
- Meal type provided (HCM/THR)

Example Hindi speech and expected extraction:
"Aarav ka wajan aath point do kilo hai, umar chaudah mahine" →
  {child_name: "Aarav", weight_kg: 8.2, age_months: 14}

"Priya ki umar teen saal hai, lambai pachchasi centimeter" →
  {child_name: "Priya", age_months: 36, height_cm: 85}

"Radha ka MUAC gyarah point do centimeter hai" →
  {child_name: "Radha", muac_cm: 11.2}  → MAM, provide RUTF

Hindi number words for voice extraction:
शून्य=0, एक=1, दो=2, तीन=3, चार=4, पांच=5, छह=6, सात=7, आठ=8, नौ=9, दस=10
ग्यारह=11, बारह=12, तेरह=13, चौदह=14, पंद्रह=15, सोलह=16, सत्रह=17, अठारह=18, उन्नीस=19, बीस=20
        """,
    },
]
