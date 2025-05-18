import json
BASE_DIR = "/Users/niyantarana/Desktop/GENAI/Agentic-AI/"
def read_user():
    with open(f"{BASE_DIR}data/users.json") as stream:
        users = json.load(stream)
        return users


def read_alternatives(id: int):
    with open(f"{BASE_DIR}data/alternatives.json") as stream:
        alternatives = json.load(stream)

    for alternative in alternatives:
        if alternative['id'] == id:
            res = alternative.get('alternative')
            print(res)
            return res






if __name__ == "__main__":
    print(read_user())
    read_alternatives(1)





