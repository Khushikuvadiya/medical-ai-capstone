# Threat Model and Misuse / Abuse Cases

## 1. System Context

This project is a research prototype for chest X-ray classification, explainability, counterfactual analysis, reviewer feedback, and audit logging.

The system accepts uploaded chest X-ray images and provides:

- NORMAL / PNEUMONIA prediction
- class probabilities
- Grad-CAM explanation
- experimental counterfactual explanation
- reviewer decision workflow
- audit logging

The prototype is not intended for real clinical diagnosis or treatment decisions.

---

## 2. Assets to Protect

Important assets include:

- model weights
- uploaded medical images
- prediction results
- explanation results
- audit logs
- reviewer feedback
- source code
- configuration files

---

## 3. Potential Threats

### 3.1 Invalid File Upload

A user may upload:

- text files
- executable files
- unsupported formats
- corrupted image files

Mitigation:

- only JPEG and PNG MIME types are accepted
- invalid requests return an HTTP error
- image decoding is handled inside controlled API functions

---

### 3.2 Non-Chest Images

A user may upload an image that is not a chest X-ray.

Risk:

The model may still return a confident prediction even though the input is outside the training distribution.

Mitigation:

- research-use warning
- reviewer oversight
- documented limitation
- future work should include out-of-distribution detection

---

### 3.3 Manipulated or Adversarial Images

An uploaded image may be intentionally modified to change the model prediction.

Examples include:

- extreme brightness changes
- contrast manipulation
- noise
- artificial perturbations

Mitigation:

- robustness testing was performed
- sensitivity to darker and low-contrast images is documented
- results must be reviewed by a human
- the system is not approved for clinical use

---

### 3.4 Over-Reliance on Model Confidence

A user may interpret a high probability as clinical certainty.

Risk:

Model confidence does not guarantee correctness or clinical validity.

Mitigation:

- confidence is presented as model probability only
- model limitations are displayed
- false positives and false negatives are documented
- human reviewer workflow is included

---

### 3.5 Misinterpretation of Grad-CAM

A user may treat highlighted regions as confirmed disease locations.

Risk:

Grad-CAM identifies regions influential to the model prediction, not clinically proven pathology.

Mitigation:

The interface displays a warning that Grad-CAM does not prove clinical causation or disease location.

---

### 3.6 Misinterpretation of Counterfactual Images

A user may interpret a generated counterfactual as a realistic medical outcome.

Risk:

Counterfactual images may contain artificial, non-physiological changes.

Mitigation:

The interface explicitly labels them as experimental model interpretation artifacts.

---

### 3.7 Patient Identifying Information

Uploaded files may contain identifying information in:

- filenames
- image metadata
- burned-in text

Risk:

Sensitive information could be exposed or stored unintentionally.

Mitigation:

- only de-identified research images should be used
- audit logs should avoid patient identifiers
- the prototype should not be used with real clinical patient data
- production systems should strip metadata and use secure storage

---

### 3.8 Audit Log Exposure

Audit logs contain prediction and review information.

Risk:

Logs could expose sensitive or operational data if shared improperly.

Mitigation:

- logs are stored locally
- no patient identifiers should be recorded
- audit output is excluded from Git using `.gitignore`
- production deployment should use access controls

---

### 3.9 Unauthorized API Access

The current prototype API does not implement authentication or authorization.

Risk:

Any user with local API access can call available endpoints.

Mitigation:

- prototype is intended for local academic demonstration only
- authentication is documented as a deployment limitation
- production deployment should implement authenticated and role-based access

---

### 3.10 Model Theft or Unauthorized Redistribution

Saved model weights may be copied.

Mitigation:

- model weights are excluded from Git
- access should be restricted in production environments
- licensing and dataset terms should be respected

---

## 4. Failure Modes

Known failure conditions include:

- low contrast images
- darker images
- domain shift
- non-chest images
- corrupted files
- poor image positioning
- unusual acquisition devices
- unseen patient populations
- artifacts
- adversarial manipulation

---

## 5. Human Oversight

The prototype includes reviewer options:

- Agree with AI
- Disagree with AI
- Needs further review

Reviewer responses are recorded in the audit log.

This workflow is intended to demonstrate human oversight rather than autonomous clinical decision making.

---

## 6. Privacy Considerations

The system should only use de-identified research data.

The prototype should not store:

- patient names
- medical record numbers
- addresses
- dates of birth
- other personally identifying information

---

## 7. Security Limitations

Current prototype limitations include:

- no authentication
- no role-based authorization
- no encrypted database
- no secure production secrets management
- no rate limiting
- no malware scanning
- no production-grade storage controls

These are acceptable only within the local academic research scope and must be addressed before any real-world deployment.

---

## 8. Residual Risk

Even with current safeguards, the following risks remain:

- incorrect predictions
- overconfidence
- domain shift
- misleading explanations
- unrealistic counterfactuals
- inappropriate use outside research settings

---

## 9. Recommended Future Controls

Future production development should include:

- authentication
- role-based access control
- rate limiting
- secure HTTPS deployment
- encrypted storage
- stronger file validation
- metadata removal
- out-of-distribution detection
- continuous model monitoring
- drift detection
- formal clinical validation
- independent security review

---

## 10. Risk Summary

This prototype is suitable for academic research and demonstration.

It is not suitable for real clinical deployment without substantial additional security, privacy, validation, monitoring, and regulatory controls.