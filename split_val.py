import json
import random

with open("vatex_validation_v1.0.json") as f:
    json_data = json.load(f)

random.shuffle(json_data)

with open("new_vatex_validation.json", 'w') as val_f:
    json.dump(json_data[:1500], val_f, ensure_ascii=False)

with open("new_vatex_test.json", 'w') as test_f:
    json.dump(json_data[1500:], test_f, ensure_ascii=False)