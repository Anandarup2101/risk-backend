import json
from utils.global_explainability import get_shap_summary, get_shap_bar
from utils.individual_explainability import get_global_pdp_data

print("Computing SHAP + PDP locally...")

# ---- GLOBAL SHAP ----
shap_summary = get_shap_summary()
shap_bar = get_shap_bar()

# ---- PDP ----
pdp_data = get_global_pdp_data()

# ---- SAVE ----
with open("shap_summary.json", "w") as f:
    json.dump(shap_summary, f)

with open("shap_bar.json", "w") as f:
    json.dump(shap_bar, f)

with open("pdp_data.json", "w") as f:
    json.dump(pdp_data, f)

print("✅ JSON files generated")