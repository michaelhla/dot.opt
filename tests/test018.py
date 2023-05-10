
import json

# edge case, file not found


def load_db_to_state(path):
    try:
        with open(path, 'r') as f:
            res_dictionary = json.load(f)
    except Exception as e:
        res_dictionary = {}
        print(e)

    return res_dictionary


if load_db_to_state('bad path') == {}:
    print('TEST 018 PASSED')
else:
    print('TEST 018 FAILED')
