import json
import csv

with open("data/current_leaders.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    leaders = data["leaders"]

regime_dict = {}
with open("data/regime/political-regime.csv", newline='', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) < 3:
            continue
        country = row[0].strip()
        regime_dict[country] = row[3].strip()

print(regime_dict)

for entry in leaders:
    entry["status"] = "current"
    country = entry["country"].strip()
    
    try:
        value = int(regime_dict[country])
        if value == 0:
            classification = "Dictator"
        elif value == 1:
            classification = "Authoritarian"
        elif value in [2, 3]:
            classification = "Democratic"
        else:
            classification = "NaN"
    except (KeyError, ValueError, TypeError):
        classification = "NaN"
        print(country)

    entry["classification"] = classification

with open("updated_leaders.json", "w", encoding="utf-8") as f:
    json.dump(leaders, f, indent=2, ensure_ascii=False)