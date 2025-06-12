import csv
import json

# Load leaders from JSON
with open("updated_leaders.json", encoding="utf-8") as f:
    leaders = json.load(f)

# Build regime dictionary from CSV
regime_dict = {}
with open("data/regime/political-regime.csv", newline='', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip header if present
    for row in reader:
        if len(row) < 4:
            continue
        country = row[0].strip()
        regime_dict[country] = row[3].strip()

# Map regime values to classifications
regime_mapping = {
    0: "Closed Autocracy",
    1: "Electoral Autocracy",
    2: "Electoral Democracy",
    3: "Liberal Democracy"
}

# Update leaders
for entry in leaders:
    entry["status"] = "current"
    country = entry["country"].strip()

    try:
        value = int(regime_dict[country])
        classification = regime_mapping.get(value, "NaN")
    except (KeyError, ValueError, TypeError):
        classification = "NaN"
        print("not found:", country)

    entry["classification"] = classification

# Save the updated file
with open("updated_leaders.json", "w", encoding="utf-8") as f:
    json.dump(leaders, f, indent=2, ensure_ascii=False)