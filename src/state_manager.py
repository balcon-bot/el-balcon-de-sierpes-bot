import requests
import json
import os

class StateManager:
    def __init__(self):
        self.gist_url = os.getenv('GIST_URL')
        self.token = os.getenv('GIST_TOKEN')
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def get_last_title(self):
        try:
            response = requests.get(self.gist_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                filename = list(data['files'].keys())[0]
                content = data['files'][filename]['content']
                state = json.loads(content)
                return state.get('last_title', '')
        except Exception as e:
            print(f"⚠️ Error leyendo estado: {e}")
        return ""

    def update_last_title(self, new_title):
        try:
            payload = {
                "files": {
                    "bot_state.json": {
                        "content": json.dumps({"last_title": new_title})
                    }
                }
            }
            response = requests.patch(self.gist_url, headers=self.headers, json=payload, timeout=10)
            if response.status_code == 200:
                print(f"✅ Memoria actualizada.")
        except Exception as e:
            print(f"⚠️ Error guardando estado: {e}")

    def is_duplicate(self, current_title):
        last_title = self.get_last_title()
        if last_title and last_title.strip().lower() == current_title.strip().lower():
            return True
        return False
